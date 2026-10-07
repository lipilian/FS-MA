#include "widgets/MeshView.hpp"
#include <QApplication>
#include <QEventLoop>
#include <QOpenGLWidget>
#include <QTimer>
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
        view.setMAClouds({}); view.setCapturedClouds({}); view.setMesh({}); view.setPredictedCameras({});
        view.setShowMAClouds(false);
        require(snapshot()==no_points,"Clean retained point-cloud geometry");
        std::cout<<"PASS: CUDA/OpenGL uploads, exclusive FS/MA rendering, stable view/poses, cached switches, retake, empty clouds and clean\n";
    } catch (const std::exception& error) {
        std::cerr<<error.what()<<'\n'; return 1;
    }
}
