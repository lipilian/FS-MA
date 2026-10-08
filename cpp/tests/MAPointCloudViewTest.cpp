#include "widgets/MeshView.hpp"
#include <QApplication>
#include <QEventLoop>
#include <QOpenGLWidget>
#include <QTimer>
#include <QTransform>
#include <cmath>
#include <iostream>
#include <stdexcept>

namespace {
void require(bool ok, const char* message) { if (!ok) throw std::runtime_error(message); }
void checked(cudaError_t error) { if (error!=cudaSuccess) throw std::runtime_error(cudaGetErrorString(error)); }
struct Device {
    cudaStream_t stream{};
    float *dense{}, *input{}, *scale{};
    explicit Device(int n) {
        checked(cudaStreamCreateWithFlags(&stream,cudaStreamNonBlocking));
        checked(cudaMalloc(reinterpret_cast<void**>(&dense),n*6*sizeof(float)));
        checked(cudaMalloc(reinterpret_cast<void**>(&input),n*7*sizeof(float)));
        checked(cudaMalloc(reinterpret_cast<void**>(&scale),sizeof(float)));
    }
    ~Device() { cudaFree(dense); cudaFree(input); cudaFree(scale); cudaStreamDestroy(stream); }
};
SharedGPUMesh cloud(quint64 id, bool red, bool empty=false) {
    constexpr int w=31,h=23,n=w*h;
    auto device=std::make_shared<Device>(n);
    std::vector<float> dense(n*6),input(n*7);
    for (int y=0;y<h;++y) for (int x=0;x<w;++x) {
        const int i=y*w+x;
        const float px=(x-15)*.006f, py=(y-11)*.006f, pz=.4f;
        dense[i]=px; dense[n+i]=py; dense[2*n+i]=pz;
        dense[3*n+i]=std::log(std::sqrt(px*px+py*py+pz*pz));
        dense[5*n+i]=1;
        input[i]=((red ? 1.f : 0.f)-.485f)/.229f;
        input[n+i]=((red ? 0.f : 1.f)-.456f)/.224f;
        input[2*n+i]=-.406f/.225f;
        input[6*n+i]=empty ? 0.f : .4f;
    }
    float scale=0;
    checked(cudaMemcpyAsync(device->dense,dense.data(),dense.size()*sizeof(float),cudaMemcpyHostToDevice,device->stream));
    checked(cudaMemcpyAsync(device->input,input.data(),input.size()*sizeof(float),cudaMemcpyHostToDevice,device->stream));
    checked(cudaMemcpyAsync(device->scale,&scale,sizeof(float),cudaMemcpyHostToDevice,device->stream));
    auto buffer=std::make_shared<fs::MeshGPUBuffer>();
    auto frame=std::make_shared<GPUMeshFrame>();
    frame->stats=fs::build_ma_point_cloud_gpu({device->dense,device->input,device->scale,w,h,device->stream},*buffer);
    frame->buffer=buffer; frame->point_cloud=true; frame->stream=device->stream;
    frame->stream_owner=device; frame->image_id=id;
    checked(cudaGetDevice(&frame->device));
    return frame;
}
int pixels(const QImage& image, bool red) {
    int count=0;
    for (int y=0;y<image.height();++y) for (int x=0;x<image.width();++x) {
        const auto p=image.pixelColor(x,y);
        if (red ? p.red()>240 && p.green()<10 && p.blue()<10
                : p.green()>240 && p.red()<10 && p.blue()<10) ++count;
    }
    return count;
}
}

int main(int argc, char** argv) {
    if (qEnvironmentVariableIsEmpty("DISPLAY") && qEnvironmentVariableIsEmpty("WAYLAND_DISPLAY")) {
        std::cout<<"SKIP: an NVIDIA OpenGL display is required\n"; return 77;
    }
    QSurfaceFormat::setDefaultFormat(MeshView::surfaceFormat());
    QApplication app(argc,argv);
    try {
        MeshView view; view.resize(800,600);
        view.setAttribute(Qt::WA_ShowWithoutActivating);
        view.setWindowFlag(Qt::WindowDoesNotAcceptFocus);
        QString failure; int uploads=0;
        QObject::connect(&view,&MeshView::renderFailed,[&](QString message) { failure=message; });
        QObject::connect(&view,&MeshView::log,[&](QString message) { if (message.contains("→ OpenGL")) ++uploads; });
        const auto fs=cloud(1,true), ma=cloud(1,false);
        auto clouds=std::make_shared<std::vector<SharedGPUMesh>>(); clouds->push_back(ma);
        view.setCapturedClouds({fs}); view.setGPUMesh(fs); view.setMAClouds(clouds);
        // Nonidentity first pose, including negative world Z. No camera frustum
        // is needed: the ID still maps both clouds into the same world frame.
        auto poses=std::make_shared<std::vector<PredictedCameraFrame>>();
        poses->push_back({{},cv::Matx44d(0,-1,0,.1,1,0,0,-.1,0,0,1,-.5,0,0,0,1),1,{}});
        view.setPredictedCameras(poses); view.show(); view.resetView();
        const auto snapshot=[&] {
            QEventLoop loop; QTimer::singleShot(60,&loop,&QEventLoop::quit); loop.exec();
            auto* canvas=view.findChild<QOpenGLWidget*>();
            require(canvas && canvas->isValid(),"OpenGL canvas unavailable");
            const auto image=canvas->grabFramebuffer();
            if (!failure.isEmpty()) throw std::runtime_error(failure.toStdString());
            require(!image.isNull(),"Empty framebuffer");
            return image;
        };
        const auto fs_image=snapshot();
        require(pixels(fs_image,true)>500 && pixels(fs_image,false)==0,"FS selection drew MA or lost FS points");
        require(uploads==2,"Both GPU sources must be uploaded once");
        for (int i=0;i<3;++i) {
            view.setShowMAClouds(true);
            const auto ma_image=snapshot();
            require(pixels(ma_image,false)>500 && pixels(ma_image,true)==0,"MA selection drew FS or lost MA points");
            // Equal geometry and pose must occupy exactly the same pixel set.
            for (int y=0;y<fs_image.height();++y) for (int x=0;x<fs_image.width();++x)
                require((fs_image.pixelColor(x,y).red()>240)==(ma_image.pixelColor(x,y).green()>240),
                    "Source switch changed world pose or view framing");
            view.setShowMAClouds(false);
            require(snapshot()==fs_image,"Switching back changed the FS scene");
        }
        require(uploads==2,"Source selection reuploaded GPU geometry");
        view.setShowMAClouds(true);
        auto replacement=std::make_shared<std::vector<SharedGPUMesh>>(); replacement->push_back(cloud(2,false));
        auto new_poses=std::make_shared<std::vector<PredictedCameraFrame>>(*poses); new_poses->back().image_id=2;
        view.setMAClouds(replacement); view.setPredictedCameras(new_poses);
        require(pixels(snapshot(),false)>500 && uploads==3,"Retake failed to replace the MA cloud");
        auto empty=std::make_shared<std::vector<SharedGPUMesh>>(); empty->push_back(cloud(2,false,true));
        view.setMAClouds(empty);
        const auto no_points=snapshot();
        require(pixels(no_points,true)==0 && pixels(no_points,false)==0,"Empty MA result retained old geometry or fell back to FS");
        view.setMAClouds({}); view.setCapturedClouds({}); view.setGPUMesh({}); view.setPredictedCameras({});
        view.setShowMAClouds(false);
        require(snapshot()==no_points,"Clean retained point-cloud geometry");
        // Render five MA meshes at distinct poses, with negative world-space Z.
        // The red FS source must disappear while MA triangles are selected.
        auto surface=std::make_shared<GPUMeshFrame>(*ma);
        auto mesh_buffer=std::make_shared<fs::MeshGPUBuffer>();
        // This rendering fixture uses 6 mm spacing, independent of MA's production limit.
        surface->stats=fs::build_mesh_from_point_cloud_gpu(*ma->buffer,31,23,*mesh_buffer,ma->stream,.01);
        surface->buffer=mesh_buffer; surface->point_cloud=false;
        auto surfaces=std::make_shared<std::vector<SharedGPUMesh>>();
        auto mesh_poses=std::make_shared<std::vector<PredictedCameraFrame>>();
        auto mesh_clouds=std::make_shared<std::vector<SharedGPUMesh>>();
        for (int i=0;i<5;++i) {
            auto part=std::make_shared<GPUMeshFrame>(*surface); part->image_id=i+1; surfaces->push_back(part);
            auto points=std::make_shared<GPUMeshFrame>(*ma); points->image_id=i+1; mesh_clouds->push_back(points);
            auto pose=cv::Matx44d::eye(); pose(0,3)=i*.25; pose(2,3)=-.5;
            mesh_poses->push_back({{},pose,quint64(i+1),{}});
        }
        int presented_meshes=0;
        QObject::connect(&view,&MeshView::maMeshesPresented,[&](SharedGPUMeshes result) {
            require(result==surfaces,"Renderer presented an outdated MA mesh snapshot"); ++presented_meshes;
        });
        view.setCapturedClouds({fs}); view.setMAClouds(mesh_clouds); view.setMAMeshes(surfaces);
        view.setPredictedCameras(mesh_poses); view.setShowMAClouds(true); view.setMode(1); view.resetView();
        const auto mesh_image=snapshot();
        require(presented_meshes==1 && pixels(mesh_image,false)>500 && pixels(mesh_image,true)==0,
                "MA meshes were not presented, lost negative world Z, or overlapped FS geometry");
        view.setMode(2); const auto wire=snapshot(); require(wire!=mesh_image,"MA wireframe rendered filled triangles");
        view.setMode(0); require(snapshot()!=mesh_image,"MA point-cloud mode still draws mesh surfaces");
        auto updated_poses=std::make_shared<std::vector<PredictedCameraFrame>>(*mesh_poses);
        updated_poses->back().camera_to_world(1,3)=.2;
        view.setPredictedCameras(updated_poses); view.setMode(1);
        require(snapshot()!=mesh_image,"MA mesh ignored an updated camera pose");
        view.setMAClouds({}); view.setMAMeshes({}); view.setCapturedClouds({}); view.setPredictedCameras({});
        view.setShowMAClouds(false);
        require(snapshot()==no_points,"Clean retained MA mesh geometry");
        // Compare the animation destination against actual textured GL pixels,
        // including the top/bottom orientation, perspective and a resized view.
        view.setCamera(fs::MeshCamera{cv::Matx33d(240,0,120,0,255,85,0,0,1),{320,240}});
        QImage image(320,240,QImage::Format_RGB32);
        const QColor colors[] = {QColor(220,40,180),QColor(40,210,220),QColor(230,220,30),QColor(70,80,220)};
        for (int y=0;y<240;++y) for (int x=0;x<320;++x)
            image.setPixelColor(x,y,colors[(y>=120 ? 2 : 0)+(x>=160 ? 1 : 0)]);
        view.setCameraImage(image); view.resetCaptureView();
        for (const auto size : {QSize(800,600),QSize(640,720)}) {
            view.resize(size);
            const auto frame=snapshot();
            const auto quad=view.cameraImageQuad();
            require(quad.size()==4,"Active camera has no screen-space image corners");
            QTransform transform;
            require(QTransform::quadToQuad(QPolygonF{{0,0},{1,0},{1,1},{0,1}},quad,transform),"Degenerate camera projection");
            auto* canvas=view.findChild<QOpenGLWidget*>();
            for (int y=0;y<2;++y) for (int x=0;x<2;++x) {
                const auto logical=transform.map(QPointF(.25+.5*x,.25+.5*y))-canvas->mapTo(&view,QPoint());
                const QPoint pixel(qRound(logical.x()*frame.width()/canvas->width()),qRound(logical.y()*frame.height()/canvas->height()));
                require(frame.rect().contains(pixel),"Projected camera lies outside the framebuffer");
                const auto actual=frame.pixelColor(pixel), expected=colors[y*2+x];
                require(std::abs(actual.red()-expected.red())<4 && std::abs(actual.green()-expected.green())<4 &&
                        std::abs(actual.blue()-expected.blue())<4,"Camera projection does not match the rendered texture");
            }
        }
        view.setCameraImage({}); snapshot();
        require(view.cameraImageQuad().size()==4,"Hiding the texture removed the animation's wireframe destination");
        view.setCamera({}); snapshot();
        require(view.cameraImageQuad().isEmpty(),"Clearing the camera retained a stale animation destination");
        const fs::MeshCamera camera{cv::Matx33d(240,0,120,0,255,85,0,0,1),{320,240}};
        QImage dark(320,240,QImage::Format_RGB32); dark.fill(Qt::black);
        auto cameras=std::make_shared<std::vector<PredictedCameraFrame>>();
        auto left=cv::Matx44d::eye(), right=cv::Matx44d::eye(); left(0,3)=-.10; right(0,3)=.10;
        cameras->push_back({camera,left,10,dark}); cameras->push_back({camera,right,20,dark});
        view.setPredictedCameras(cameras); view.setCameraPresentation(20,0,20); view.resetCaptureView();
        const auto color_count=[](const QImage& frame,bool green) {
            int count=0;
            for (int y=0;y<frame.height();++y) for (int x=0;x<frame.width();++x) {
                const auto p=frame.pixelColor(x,y);
                if (green ? p.green()>230 && p.red()<50 && p.blue()<100 : p.blue()>230 && p.red()<50 && p.green()<150) ++count;
            }
            return count;
        };
        auto camera_image=snapshot();
        require(color_count(camera_image,true)==0 && color_count(camera_image,false)>20,"Camera frames must be blue before landing");
        const auto target=view.cameraImageQuad();
        require(target.size()==4,"Transfer does not target the active MA camera");
        view.setCameraPresentation(20,20); camera_image=snapshot();
        require(color_count(camera_image,true)>10 && color_count(camera_image,false)>10,"Active camera did not turn green while other cameras stayed blue");
        view.setCameraPresentation(10,10); const auto switched=snapshot();
        require(view.cameraImageQuad()!=target && switched!=camera_image && color_count(switched,true)>10 && color_count(switched,false)>10,
                "Highlight or transfer destination did not follow the active camera ID");
        auto overlapping=std::make_shared<std::vector<PredictedCameraFrame>>(*cameras);
        overlapping->back().camera_to_world=overlapping->front().camera_to_world;
        view.setPredictedCameras(overlapping); view.setCameraPresentation(20,20);
        require(color_count(snapshot(),true)>10,"An overlapping camera hid the active green frame");
        view.setPredictedCameras({}); view.setCameraPresentation(0,0);
        const auto cleared=snapshot();
        require(color_count(cleared,true)==0 && color_count(cleared,false)==0 && view.cameraImageQuad().isEmpty(),
                "Clean retained camera highlighting");
        std::cout<<"PASS: CUDA/OpenGL uploads, exclusive FS/MA rendering, stable view/poses, cached switches, retake, empty clouds and clean\n";
        std::cout<<"PASS: projected camera corners match textured GL pixels, perspective, orientation and resizing\n";
        std::cout<<"PASS: five MA meshes, negative world Z, exclusive source/mode rendering, pose updates and green/blue camera selection\n";
    } catch (const std::exception& error) {
        std::cerr<<error.what()<<'\n'; return 1;
    }
}
