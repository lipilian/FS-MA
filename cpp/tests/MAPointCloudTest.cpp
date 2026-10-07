#include "fs/geometry/MeshBuilderGPU.hpp"
#include <algorithm>
#include <cmath>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <vector>

namespace {
void require(bool ok, const char* message) { if (!ok) throw std::runtime_error(message); }
void checked(cudaError_t error) {
    if (error!=cudaSuccess) throw std::runtime_error(cudaGetErrorString(error));
}
struct Device {
    cudaStream_t stream{};
    float *dense{}, *input{}, *scale{};
    explicit Device(int pixels) {
        checked(cudaStreamCreateWithFlags(&stream,cudaStreamNonBlocking));
        checked(cudaMalloc(reinterpret_cast<void**>(&dense),pixels*6*sizeof(float)));
        checked(cudaMalloc(reinterpret_cast<void**>(&input),pixels*7*sizeof(float)));
        checked(cudaMalloc(reinterpret_cast<void**>(&scale),sizeof(float)));
    }
    ~Device() { cudaFree(dense); cudaFree(input); cudaFree(scale); cudaStreamDestroy(stream); }
};
}

int main() {
    try {
        // Partial CUDA blocks, per-view offsets and very different input rays:
        // MA output rays, not the calibration rays, must determine the points.
        constexpr int width=17, height=19, n=width*height, views=2;
        Device device(n*views);
        std::vector<float> dense(views*6*n), input(views*7*n);
        const float means[3]={.485f,.456f,.406f}, deviations[3]={.229f,.224f,.225f};
        for (int view=0;view<views;++view) for (int i=0;i<n;++i) {
            float* d=dense.data()+view*6*n;
            d[i]=3; d[n+i]=0; d[2*n+i]=4; d[3*n+i]=std::log(float(view+2));
            d[4*n+i]=-10; // No arbitrary confidence threshold.
            d[5*n+i]=1;
            for (int c=0;c<3;++c)
                input[view*7*n+c*n+i]=(float((i+view*33+c*71)%256)/255.f-means[c])/deviations[c];
            input[view*7*n+5*n+i]=1;
            input[view*7*n+6*n+i]=.25f+view; // Validity only; keep MA's predicted distance.
        }
        dense[5*n+1]=0; dense[5*n+2]=-1; // Ambiguous samples.
        dense[3]=dense[n+3]=dense[2*n+3]=0; // Zero ray.
        dense[2*n+4]=-4; // Behind the camera.
        dense[5]=std::numeric_limits<float>::quiet_NaN();
        dense[3*n+6]=1000; // Overflow.
        dense[3*n+7]=-1000; // Zero distance.
        input[8]=std::numeric_limits<float>::infinity();
        dense[5*n+9]=std::numeric_limits<float>::quiet_NaN();
        // Valid MA predictions at missing/invalid input depths must remain zero.
        input[6*n+10]=0;
        input[6*n+11]=-1;
        input[6*n+12]=std::numeric_limits<float>::quiet_NaN();
        input[6*n+13]=std::numeric_limits<float>::infinity();
        input[7*n+6*n+n-1]=0; // Independent mask in the second view, last CUDA block.
        const auto upload=[&](float scale) {
            checked(cudaMemcpyAsync(device.dense,dense.data(),dense.size()*sizeof(float),cudaMemcpyHostToDevice,device.stream));
            checked(cudaMemcpyAsync(device.input,input.data(),input.size()*sizeof(float),cudaMemcpyHostToDevice,device.stream));
            checked(cudaMemcpyAsync(device.scale,&scale,sizeof(float),cudaMemcpyHostToDevice,device.stream));
            checked(cudaStreamSynchronize(device.stream));
        };
        upload(std::log(2.f));
        fs::MeshGPUBuffer buffer;
        for (int view=0;view<views;++view) {
            const auto stats=fs::build_ma_point_cloud_gpu({device.dense+view*6*n,device.input+view*7*n,
                device.scale,width,height,device.stream},buffer);
            require(stats.point_count==static_cast<unsigned>(n-(view==0 ? 13 : 1)),"Wrong valid MA point count");
            require(!stats.triangle_count && !stats.area_m2 && buffer.vertex_slots()==n,"Wrong MA buffer shape/statistics");
            const float distance=2.f*(view+2);
            require(std::abs(stats.low[0]-.6f*distance)<1e-5f && std::abs(stats.high[2]-.8f*distance)<1e-5f,
                "MA metric bounds are wrong");
            std::vector<fs::MeshGPUVertex> points(n);
            checked(cudaMemcpy(points.data(),buffer.vertices(),points.size()*sizeof(points[0]),cudaMemcpyDeviceToHost));
            for (int i=0;i<n;++i) {
                const auto& p=points[i];
                if ((view==0 && i>=1 && i<=13) || (view==1 && i==n-1)) {
                    require(p.x==0 && p.y==0 && p.z==0 && p.r==0 && p.g==0 && p.b==0,"Invalid MA vertex not cleared");
                } else {
                    require(std::abs(p.x-.6f*distance)<1e-5f && p.y==0 && std::abs(p.z-.8f*distance)<1e-5f,
                        "MA ray-distance decoding used wrong rays, depth or scale");
                    require(std::abs(p.r-float((i+view*33)%256)/255.f)<1e-6f &&
                        std::abs(p.g-float((i+view*33+71)%256)/255.f)<1e-6f &&
                        std::abs(p.b-float((i+view*33+142)%256)/255.f)<1e-6f,"MA RGB decoding differs from input image");
                }
            }
        }
        // All-zero input depth must clear a previous scene even with nonzero
        // MA predictions, including reduction state and bounds.
        std::fill(input.begin()+6*n,input.begin()+7*n,0.f);
        upload(0);
        auto stats=fs::build_ma_point_cloud_gpu({device.dense,device.input,device.scale,width,height,device.stream},buffer);
        require(!stats.point_count,"Empty MA cloud retained points");
        for (int c=0;c<3;++c) require(stats.low[c]==0 && stats.high[c]==0,"Empty MA cloud retained bounds");
        std::vector<fs::MeshGPUVertex> empty(n);
        checked(cudaMemcpy(empty.data(),buffer.vertices(),empty.size()*sizeof(empty[0]),cudaMemcpyDeviceToHost));
        for (const auto& p:empty) require(p.x==0 && p.y==0 && p.z==0 && p.r==0 && p.g==0 && p.b==0,
            "Zero input depth retained MA vertices");
        // Tiny rays and global scale clamps follow the upstream adaptors.
        dense[0]=0; dense[2*n]=1e-9f; dense[3*n]=0; dense[5*n]=1;
        input[6*n]=1;
        upload(-100.f);
        stats=fs::build_ma_point_cloud_gpu({device.dense,device.input,device.scale,width,height,device.stream},buffer);
        require(stats.point_count==1 && std::abs(stats.high[2]-1e-9f)<1e-14f,"MA adaptor epsilon/scale clamp mismatch");
        upload(std::numeric_limits<float>::quiet_NaN());
        stats=fs::build_ma_point_cloud_gpu({device.dense,device.input,device.scale,width,height,device.stream},buffer);
        require(!stats.point_count,"Invalid scale produced MA points");
        bool rejected=false;
        try { fs::build_ma_point_cloud_gpu({device.dense,device.input,nullptr,width,height,device.stream},buffer); }
        catch (const std::invalid_argument&) { rejected=true; }
        require(rejected && !buffer.vertex_slots(),"Invalid MA inputs retained usable output");
        std::cout<<"PASS: MA predicted rays, metric depth/scale, RGB, masks, zero/invalid input depth, per-view offsets, partial blocks and buffer reuse\n";
    } catch (const std::exception& error) {
        std::cerr<<error.what()<<'\n'; return 1;
    }
}
