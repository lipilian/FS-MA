#include "widgets/MeshView.hpp"
#include <QApplication>
#include <QEventLoop>
#include <QOpenGLWidget>
#include <QMouseEvent>
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
    unsigned char* mask{};
    explicit Device(int n) {
        checked(cudaStreamCreateWithFlags(&stream,cudaStreamNonBlocking));
        checked(cudaMalloc(reinterpret_cast<void**>(&dense),n*6*sizeof(float)));
        checked(cudaMalloc(reinterpret_cast<void**>(&input),n*7*sizeof(float)));
        checked(cudaMalloc(reinterpret_cast<void**>(&scale),sizeof(float)));
        checked(cudaMalloc(reinterpret_cast<void**>(&mask),n));
    }
    ~Device() { cudaFree(dense); cudaFree(input); cudaFree(scale); cudaFree(mask); cudaStreamDestroy(stream); }
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
SharedGPUMesh fsGeometry(quint64 id, bool surface, bool red=true, int w=31, int h=23) {
    const int n=w*h;
    auto device=std::make_shared<Device>(n);
    std::vector<float> xyz(n*3),rgb(n*3);
    for (int y=0;y<h;++y) for (int x=0;x<w;++x) {
        const int i=y*w+x;
        xyz[3*i]=(x-(w-1)*.5f)*.006f; xyz[3*i+1]=(y-(h-1)*.5f)*.006f; xyz[3*i+2]=.4f;
        rgb[i]=red ? 255 : 0; rgb[n+i]=red ? 0 : 255;
    }
    checked(cudaMemcpyAsync(device->dense,xyz.data(),xyz.size()*sizeof(float),cudaMemcpyHostToDevice,device->stream));
    checked(cudaMemcpyAsync(device->input,rgb.data(),rgb.size()*sizeof(float),cudaMemcpyHostToDevice,device->stream));
    checked(cudaMemsetAsync(device->mask,255,n,device->stream));
    auto buffer=std::make_shared<fs::MeshGPUBuffer>();
    auto frame=std::make_shared<GPUMeshFrame>();
    const fs::MeshGPUInputs input{device->dense,device->input,device->mask,w,h,device->stream};
    frame->stats=surface ? fs::build_mesh_gpu(input,*buffer) : fs::build_point_cloud_gpu(input,*buffer);
    frame->buffer=buffer; frame->point_cloud=!surface; frame->stream=device->stream;
    frame->stream_owner=device; frame->image_id=id;
    frame->camera={cv::Matx33d(.4/.006,0,(w-1)*.5,0,.4/.006,(h-1)*.5,0,0,1),{w,h}};
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
        // Render five MA clouds at distinct poses, with negative world-space Z.
        // Mesh/wireframe requests must keep MA in point-cloud mode.
        auto cloud_poses=std::make_shared<std::vector<PredictedCameraFrame>>();
        auto multi_clouds=std::make_shared<std::vector<SharedGPUMesh>>();
        for (int i=0;i<5;++i) {
            auto points=std::make_shared<GPUMeshFrame>(*ma); points->image_id=i+1; multi_clouds->push_back(points);
            auto pose=cv::Matx44d::eye(); pose(0,3)=i*.25; pose(2,3)=-.5;
            cloud_poses->push_back({{},pose,quint64(i+1),{}});
        }
        int presented_clouds=0;
        QObject::connect(&view,&MeshView::maCloudsPresented,[&](SharedGPUClouds result) {
            require(result==multi_clouds,"Renderer presented an outdated MA point-cloud snapshot"); ++presented_clouds;
        });
        const int uploads_before=uploads;
        view.setCapturedClouds({fs}); view.setMAClouds(multi_clouds);
        view.setPredictedCameras(cloud_poses); view.setShowMAClouds(true); view.setMode(0); view.resetView();
        const auto cloud_image=snapshot();
        require(presented_clouds==1 && pixels(cloud_image,false)>500 && pixels(cloud_image,true)==0,
                "MA clouds were not presented, lost negative world Z, or overlapped FS geometry");
        require(uploads==uploads_before+6,"Five MA clouds and one FS cloud should upload without extra mesh buffers");
        for (int mode : {1,2,0}) {
            view.setMode(mode);
            require(snapshot()==cloud_image,"A mesh/wireframe request changed MA point-cloud rendering");
        }
        view.setMAClouds(multi_clouds); snapshot();
        require(presented_clouds==1 && uploads==uploads_before+6,"Unchanged MA clouds were reuploaded or announced again");
        auto updated_poses=std::make_shared<std::vector<PredictedCameraFrame>>(*cloud_poses);
        updated_poses->back().camera_to_world(1,3)=.2;
        view.setPredictedCameras(updated_poses); view.setMode(1);
        require(snapshot()!=cloud_image,"MA point cloud ignored an updated camera pose");
        view.setMAClouds({}); view.setCapturedClouds({}); view.setPredictedCameras({});
        view.setShowMAClouds(false);
        require(snapshot()==no_points,"Clean retained MA point-cloud geometry");
        // TSDF is compact world-space geometry, including negative Z. It must
        // bypass FS masks and camera transforms and retain both extraction modes.
        const auto world_geometry=[&](bool surface) {
            const auto source=fsGeometry(1,surface,false);
            std::vector<fs::MeshGPUVertex> vertices(source->buffer->vertex_slots());
            checked(cudaMemcpy(vertices.data(),source->buffer->vertices(),vertices.size()*sizeof(vertices[0]),cudaMemcpyDeviceToHost));
            for (auto& vertex:vertices) vertex.z-=.5f;
            auto buffer=std::make_shared<fs::MeshGPUBuffer>();
            fs::upload_geometry_gpu(vertices.data(),vertices.size(),*buffer,source->stream);
            auto frame=std::make_shared<GPUMeshFrame>(*source);
            frame->buffer=buffer; frame->world_space=true;
            frame->stats.low[2]-=.5f; frame->stats.high[2]-=.5f;
            return frame;
        };
        auto tsdf=std::make_shared<TsdfScene>();
        tsdf->points=world_geometry(false); tsdf->mesh=world_geometry(true);
        view.setCapturedClouds({fs}); view.setTSDF(tsdf,true); view.setFSMasks({},true);
        view.setMode(0); view.resetView();
        const auto tsdf_points=snapshot();
        require(pixels(tsdf_points,false)>500 && pixels(tsdf_points,true)==0,
                "TSDF lost negative world Z, inherited FS transparency or mixed FS geometry");
        auto irrelevant_poses=std::make_shared<std::vector<PredictedCameraFrame>>();
        auto far_pose=cv::Matx44d::eye(); far_pose(0,3)=100;
        irrelevant_poses->push_back({{},far_pose,1,{}});
        view.setPredictedCameras(irrelevant_poses);
        require(snapshot()==tsdf_points,"TSDF world-space geometry was transformed a second time");
        const auto tsdf_uploads=uploads;
        view.setMode(1); const auto tsdf_mesh=snapshot();
        require(tsdf_mesh!=tsdf_points,"TSDF mesh mode retained the point cloud");
        const auto area_corner=tsdf_mesh.copy(tsdf_mesh.width()/2,tsdf_mesh.height()*3/4,
            tsdf_mesh.width()/2,tsdf_mesh.height()/4);
        int area_pixels=0;
        for (int y=0;y<area_corner.height();++y) for (int x=0;x<area_corner.width();++x) {
            const auto color=area_corner.pixelColor(x,y);
            if (color.red()>220 && color.green()>220 && color.blue()>220) ++area_pixels;
        }
        require(area_pixels>50,"TSDF mesh area is missing from the lower-right corner");
        const auto area_screenshot=qEnvironmentVariable("FS_TSDF_AREA_SCREENSHOT");
        if (!area_screenshot.isEmpty()) require(tsdf_mesh.save(area_screenshot),"Cannot save TSDF area screenshot");
        view.setMode(2); const auto tsdf_wire=snapshot();
        require(tsdf_wire!=tsdf_mesh && tsdf_wire!=tsdf_points,"TSDF wireframe mode did not draw mesh edges");
        view.setMode(0);
        require(snapshot()==tsdf_points && uploads==tsdf_uploads,"TSDF mode switches changed framing or reuploaded geometry");
        view.setTSDF({},false); view.setCapturedClouds({}); view.setPredictedCameras({}); view.setFSMasks({},false);
        require(snapshot()==no_points,"Invalidating TSDF left stale geometry visible");
        std::cout<<"PASS: TSDF mesh/points/wireframe, negative world Z, no double pose/masking, cached uploads and invalidation\n";
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
        // Pick using the rendered image planes, independent of FS/MA selection.
        auto* canvas=view.findChild<QOpenGLWidget*>();
        const auto mouse=[&](QEvent::Type type,QPointF point,Qt::MouseButton button,Qt::MouseButtons buttons) {
            QMouseEvent event(type,point,canvas->mapToGlobal(point.toPoint()),button,buttons,Qt::NoModifier);
            QApplication::sendEvent(canvas,&event);
        };
        const auto hover=[&](QPointF point) { mouse(QEvent::MouseMove,point,Qt::NoButton,Qt::NoButton); };
        const auto click=[&](QPointF point) {
            mouse(QEvent::MouseButtonPress,point,Qt::LeftButton,Qt::LeftButton);
            mouse(QEvent::MouseButtonRelease,point,Qt::LeftButton,Qt::NoButton);
        };
        const auto purple_count=[](const QImage& frame) {
            int count=0;
            for (int y=0;y<frame.height();++y) for (int x=0;x<frame.width();++x) {
                const auto p=frame.pixelColor(x,y);
                if (p.red()>150 && p.red()<180 && p.green()>60 && p.green()<95 && p.blue()>240) ++count;
            }
            return count;
        };
        std::vector<quint64> clicked;
        QObject::connect(&view,&MeshView::cameraClicked,[&](quint64 id) { clicked.push_back(id); });
        auto five=std::make_shared<std::vector<PredictedCameraFrame>>();
        for (int i=0;i<5;++i) {
            auto pose=cv::Matx44d::eye(); pose(0,3)=(i%3-1)*.15; pose(1,3)=(i/3)*.15;
            five->push_back({camera,pose,quint64(i+1),dark});
        }
        view.setPredictedCameras(five); view.resetCaptureView();
        std::vector<QPointF> centers;
        for (quint64 id=1;id<=5;++id) {
            view.setCameraPresentation(id,id); snapshot();
            const auto quad=view.cameraImageQuad();
            require(quad.size()==4,"Five-view camera projection missing");
            QPointF center;
            for (const auto& corner:quad) center+=corner*.25;
            centers.push_back(center-canvas->mapTo(&view,QPoint()));
        }
        const int before_hover_uploads=uploads;
        view.setCameraInteractionEnabled(true);
        for (bool show_ma : {false,true}) {
            view.setShowMAClouds(show_ma);
            for (int i=0;i<5;++i) {
                hover(centers[i]);
                require(purple_count(snapshot())>10 && canvas->cursor().shape()==Qt::PointingHandCursor,
                        "Hover did not turn a selectable camera purple");
                click(centers[i]);
                require(clicked.size()==size_t((show_ma ? 5 : 0)+i+1) && clicked.back()==quint64(i+1),
                        "Click selected the wrong camera or depended on the geometry source");
            }
        }
        require(uploads==before_hover_uploads,"Camera hover/click reuploaded point clouds");
        QEvent leave(QEvent::Leave); QApplication::sendEvent(canvas,&leave);
        require(purple_count(snapshot())==0 && canvas->cursor().shape()==Qt::ArrowCursor,"Leaving retained hover state");
        mouse(QEvent::MouseButtonPress,centers[0],Qt::LeftButton,Qt::LeftButton);
        mouse(QEvent::MouseMove,centers[0]+QPointF(30,0),Qt::NoButton,Qt::LeftButton);
        mouse(QEvent::MouseMove,centers[0],Qt::NoButton,Qt::LeftButton);
        mouse(QEvent::MouseButtonRelease,centers[0],Qt::LeftButton,Qt::NoButton);
        require(clicked.size()==10,"Rotating the scene triggered a camera click");
        view.setCameraInteractionEnabled(false); hover(centers[0]); click(centers[0]);
        require(clicked.size()==10 && purple_count(snapshot())==0,"Disabled camera interaction still picked or highlighted");
        auto retaken=std::make_shared<std::vector<PredictedCameraFrame>>(*five);
        retaken->back().image_id=99;
        view.setPredictedCameras(retaken); view.setCameraPresentation(99,99); snapshot();
        view.setCameraInteractionEnabled(true); hover(centers.back()); click(centers.back());
        require(clicked.size()==11 && clicked.back()==99,"Retake kept the previous capture ID in its hit target");
        // With coincident poses, click both non-overlapping numbered labels.
        view.setPredictedCameras(overlapping); view.setCameraPresentation(20,20); view.resetCaptureView();
        snapshot();
        const auto quad=view.cameraImageQuad().translated(-canvas->mapTo(&view,QPoint()));
        const auto scan=quad.boundingRect().adjusted(-80,-120,80,20).intersected(canvas->rect());
        bool label_one=false, label_two=false;
        // Probe above the image plane, where only the displaced labels can hit.
        for (int y=int(scan.top());y<int(quad.boundingRect().top());y+=3)
            for (int x=int(scan.left());x<int(scan.right());x+=3) {
                const auto before=clicked.size(); click(QPointF(x,y));
                if (clicked.size()>before) {
                    label_one|=clicked.back()==10; label_two|=clicked.back()==20;
                }
            }
        require(label_one && label_two,"Coincident camera poses cannot both be selected via their labels");
        view.setPredictedCameras({}); view.setCameraPresentation(0,0); snapshot();
        const auto before_clear=clicked.size(); hover(centers.back()); click(centers.back());
        require(clicked.size()==before_clear && purple_count(snapshot())==0,"Cleared cameras retained pick targets");
        view.setPredictedCameras(std::make_shared<std::vector<PredictedCameraFrame>>(1,five->front()));
        view.setCameraPresentation(1,1); snapshot();
        click(view.cameraImageQuad().boundingRect().center()-canvas->mapTo(&view,QPoint()));
        require(clicked.size()==before_clear+1 && clicked.back()==1,"Single predicted camera could not be selected");
        view.setPredictedCameras({}); view.setCamera(camera); view.setCameraImage(dark);
        view.setCameraPresentation(7,7); view.resetCaptureView(); snapshot();
        const auto single_center=view.cameraImageQuad().boundingRect().center()-canvas->mapTo(&view,QPoint());
        hover(single_center); snapshot(); click(single_center);
        require(clicked.size()==before_clear+2 && clicked.back()==7 && canvas->cursor().shape()==Qt::PointingHandCursor,
                "Single FS camera without MA poses could not be selected");
        // Hover must tint only the matching FS points and restore their original
        // RGB on leave, without uploading or changing the immutable GPU buffers.
        const auto fs_left=cloud(101,true), fs_right=cloud(202,false);
        auto fs_cameras=std::make_shared<std::vector<PredictedCameraFrame>>();
        auto fs_left_pose=cv::Matx44d::eye(), fs_right_pose=cv::Matx44d::eye();
        fs_left_pose(0,3)=-.18; fs_right_pose(0,3)=.18;
        fs_cameras->push_back({camera,fs_left_pose,101,dark});
        fs_cameras->push_back({camera,fs_right_pose,202,dark});
        view.setShowMAClouds(false); view.setCapturedClouds({fs_left,fs_right}); view.setGPUMesh(fs_right);
        view.setPredictedCameras(fs_cameras); view.resetView();
        std::vector<QPointF> fs_centers;
        for (const auto& frame:*fs_cameras) {
            view.setCameraPresentation(frame.image_id,0); snapshot();
            fs_centers.push_back(view.cameraImageQuad().boundingRect().center()-canvas->mapTo(&view,QPoint()));
        }
        const auto original_fs=snapshot();
        const int fs_uploads=uploads;
        require(pixels(original_fs,true)>300 && pixels(original_fs,false)>300,"FS hover fixture lost a view's points");
        for (int i=0;i<2;++i) {
            hover(fs_centers[i]); const auto highlighted=snapshot();
            int tinted=0, preserved=0;
            for (int y=0;y<original_fs.height();++y) for (int x=0;x<original_fs.width();++x) {
                const auto before=original_fs.pixelColor(x,y), after=highlighted.pixelColor(x,y);
                const bool red=before.red()>240 && before.green()<10 && before.blue()<10;
                const bool green=before.green()>240 && before.red()<10 && before.blue()<10;
                if (i==0 ? red : green) {
                    // Pale purple: high blue/red and a lifted green channel.
                    if (after.blue()>200 && after.red()>160 && after.green()>120 && after.green()<200) ++tinted;
                }
                if ((i==0 ? green : red) && after==before) ++preserved;
            }
            require(tinted>300 && preserved>300,"Hover did not tint exactly the matching FS point cloud pale purple");
            QApplication::sendEvent(canvas,&leave);
            require(snapshot()==original_fs,"Leaving did not restore the FS scene's exact original colors");
        }
        require(uploads==fs_uploads,"FS hover reuploaded point-cloud geometry");
        // Identical world points still highlight when the hovered view is drawn
        // after an unhovered view that has already written the same depth.
        auto coincident=std::make_shared<std::vector<PredictedCameraFrame>>(*fs_cameras);
        coincident->back().camera_to_world=coincident->front().camera_to_world;
        view.setPredictedCameras(coincident); view.setCameraPresentation(101,0); view.resetView(); snapshot();
        hover(view.cameraImageQuad().boundingRect().center()-canvas->mapTo(&view,QPoint()));
        const auto same_depth=snapshot();
        require(pixels(same_depth,true)==0 && pixels(same_depth,false)==0,
                "An overlapping view hid the hovered FS cloud at equal depth");
        QApplication::sendEvent(canvas,&leave);
        require(pixels(snapshot(),true)>300,"Overlapping FS points did not restore after hover");
        // Mask display is independent of MA, including matching camera IDs.
        view.setShowMAClouds(true);
        const auto ma_before_mask=snapshot();
        QImage full_mask(62,46,QImage::Format_Grayscale8); full_mask.fill(255);
        QImage empty_mask(full_mask.size(),QImage::Format_Grayscale8); empty_mask.fill(0);
        view.setFSMasks({{1,empty_mask}});
        require(snapshot()==ma_before_mask,"FS mask transparency changed the MA display");
        view.setShowMAClouds(false); view.setMAClouds({}); view.setPredictedCameras({});
        view.setCamera(fs::MeshCamera{}); view.setCameraInteractionEnabled(false);
        const auto masked_cloud=fsGeometry(501,false);
        view.setCameraPresentation(501,0); view.setCapturedClouds({masked_cloud}); view.setGPUMesh(masked_cloud);
        view.setFSMasks({},false); view.resetView();
        const auto solid=snapshot();
        const int mask_uploads=uploads;
        const auto check_blend=[&](const QImage& original,const QImage& actual,bool red) {
            const QColor background=original.pixelColor(0,0);
            int opaque=0,translucent=0;
            for (int y=0;y<original.height();++y) for (int x=0;x<original.width();++x) {
                const auto color=original.pixelColor(x,y), shown=actual.pixelColor(x,y);
                if (!(red ? color.red()>240 && color.green()<10 && color.blue()<10
                          : color.green()>240 && color.red()<10 && color.blue()<10)) continue;
                if (shown==color) { ++opaque; continue; }
                require(std::abs(shown.red()-(.3*color.red()+.7*background.red()))<2 &&
                        std::abs(shown.green()-(.3*color.green()+.7*background.green()))<2 &&
                        std::abs(shown.blue()-(.3*color.blue()+.7*background.blue()))<2,
                        "Unselected FS sample did not blend at alpha 0.3");
                ++translucent;
            }
            return std::pair(opaque,translucent);
        };
        view.setFSMasks({});
        const auto unmasked=snapshot();
        require(check_blend(solid,unmasked,true)==std::pair(0,713),"Undrawn FS view must be entirely translucent");
        QImage region=empty_mask.copy();
        for (int y=0;y<23;++y) for (int x=0;x<31;++x) region.scanLine(y)[x]=255;
        view.setFSMasks({{501,region}});
        require(check_blend(solid,snapshot(),true)==std::pair(192,521),
                "Full-resolution mask did not select the corresponding FS pixel slots");
        view.setFSMasks({{501,full_mask}});
        require(snapshot()==solid,"All-selected mask did not restore exact solid RGB");
        view.setFSMasks({{501,empty_mask}});
        require(snapshot()==unmasked,"Empty selection did not fade every point");
        view.setFSMasks({});
        require(snapshot()==unmasked && uploads==mask_uploads,"Clearing a mask changed geometry or reuploaded GPU vertices");
        // Separate masks follow local pixels even when camera poses rotate.
        auto mask_poses=std::make_shared<std::vector<PredictedCameraFrame>>();
        auto left_pose=cv::Matx44d::eye(), right_pose=cv::Matx44d::eye();
        left_pose(0,3)=-.15; right_pose(0,3)=.15;
        right_pose(0,0)=right_pose(1,1)=0; right_pose(0,1)=-1; right_pose(1,0)=1;
        mask_poses->push_back({{},left_pose,501,{}}); mask_poses->push_back({{},right_pose,502,{}});
        const auto other_cloud=fsGeometry(502,false,false);
        view.setCapturedClouds({masked_cloud,other_cloud}); view.setPredictedCameras(mask_poses);
        view.setFSMasks({},false); view.resetView(); const auto multi_solid=snapshot();
        const int multi_uploads=uploads;
        view.setFSMasks({{501,region},{502,full_mask}});
        const auto mixed=snapshot();
        require(check_blend(multi_solid,mixed,true)==std::pair(192,521) &&
                check_blend(multi_solid,mixed,false)==std::pair(713,0),"Masks leaked between FS captures");
        view.setFSMasks({{502,region}});
        const auto rotated=snapshot();
        require(check_blend(multi_solid,rotated,true)==std::pair(0,713) &&
                check_blend(multi_solid,rotated,false)==std::pair(192,521) && uploads==multi_uploads,
                "Mask coordinates changed with MA pose or caused a geometry upload");
        // The first imported view defaults to an FS triangle mesh.
        const auto surface=fsGeometry(503,true);
        view.setCameraPresentation(503,0);
        view.setCapturedClouds({}); view.setGPUMesh(surface); view.setPredictedCameras({});
        view.setMode(1); view.setFSMasks({},false); view.resetView(); const auto solid_surface=snapshot();
        view.setFSMasks({}); const auto faint_surface=snapshot();
        const auto surface_counts=check_blend(solid_surface,faint_surface,true);
        require(surface_counts.first==0 && surface_counts.second>10000,"Unmasked first-view FS mesh did not fade");
        view.setFSMasks({{503,region}});
        const auto selected_surface=check_blend(solid_surface,snapshot(),true);
        require(selected_surface.first>1000 && selected_surface.second>1000,
                "First-view FS mesh failed to project its mask in local rectified coordinates");
        view.setFSMasks({},false);
        require(snapshot()==solid_surface,"Leaving mask display did not restore the original FS mesh");
        // Match the first import: retain FS points while the mesh replaces them.
        // Point mode must use those original points, not repeated triangle vertices.
        const auto surface_points=fsGeometry(503,false);
        view.setCapturedClouds({surface_points},surface_points); snapshot();
        const int switch_uploads=uploads;
        view.setMode(0); const auto surface_as_points=snapshot();
        view.setFSMasks({});
        require(check_blend(surface_as_points,snapshot(),true)==std::pair(0,713),
                "Switching Mesh to Point cloud stacked transparency from repeated triangle vertices");
        view.setFSMasks({{503,region}});
        require(check_blend(surface_as_points,snapshot(),true)==std::pair(192,521),
                "Point cloud mode lost its opaque mask after switching from Mesh");
        view.setFSMasks({},false); view.setMode(1);
        require(snapshot()==solid_surface && uploads==switch_uploads,"Mesh/point switching changed geometry or reuploaded cached vertices");
        // Multiple FS samples land on one screen pixel at normal viewing sizes.
        // That pixel must still receive alpha 0.3 only once.
        const auto dense=fsGeometry(504,false,true,960,800);
        view.setCameraPresentation(504,0); view.setCapturedClouds({dense}); view.setGPUMesh(dense);
        view.setMode(0); view.resetView(); const auto dense_solid=snapshot();
        view.setFSMasks({});
        const auto dense_counts=check_blend(dense_solid,snapshot(),true);
        require(dense_counts.first==0 && dense_counts.second>10000,
                "Dense FS samples accumulated opacity instead of staying at alpha 0.3");
        std::cout<<"PASS: CUDA/OpenGL uploads, exclusive FS/MA rendering, stable view/poses, cached switches, retake, empty clouds and clean\n";
        std::cout<<"PASS: projected camera corners match textured GL pixels, perspective, orientation and resizing\n";
        std::cout<<"PASS: five MA point clouds, point-only modes, presentation notifications, negative world Z, pose updates and green/blue camera selection\n";
        std::cout<<"PASS: five clickable cameras, purple hover, FS/MA selection, drag rejection, disabled/cleared picks, retakes and overlapping labels\n";
        std::cout<<"PASS: matching FS cloud pale-purple hover, unchanged other views, exact RGB restoration, overlapping points and cached GPU geometry\n";
        std::cout<<"PASS: unmasked/cleared FS alpha 0.3, per-view solid masks, original RGB, rotated poses, first-view mesh and unchanged MA/GPU buffers\n";
        std::cout<<"PASS: Mesh/Point cloud mode switching and dense 960x800 FS points retain alpha 0.3 without repeated blending\n";
    } catch (const std::exception& error) {
        std::cerr<<error.what()<<'\n'; return 1;
    }
}
