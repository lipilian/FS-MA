#include "fs/geometry/MeshBuilderGPU.hpp"
#include <algorithm>
#include <cmath>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <vector>

namespace {
void require(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}
void checked(cudaError_t result) {
    if (result!=cudaSuccess) throw std::runtime_error(cudaGetErrorString(result));
}
struct Inputs {
    cudaStream_t stream{};
    float *xyz{}, *rgb{};
    unsigned char* mask{};
    explicit Inputs(std::size_t count) {
        checked(cudaStreamCreateWithFlags(&stream,cudaStreamNonBlocking));
        checked(cudaMalloc(reinterpret_cast<void**>(&xyz),count*3*sizeof(float)));
        checked(cudaMalloc(reinterpret_cast<void**>(&rgb),count*3*sizeof(float)));
        checked(cudaMalloc(reinterpret_cast<void**>(&mask),count));
    }
    ~Inputs() { cudaFree(xyz); cudaFree(rgb); cudaFree(mask); cudaStreamDestroy(stream); }
    void upload(const std::vector<float>& points, const std::vector<float>& colors,
                const std::vector<unsigned char>& selection) {
        checked(cudaMemcpyAsync(xyz,points.data(),points.size()*sizeof(float),cudaMemcpyHostToDevice,stream));
        checked(cudaMemcpyAsync(rgb,colors.data(),colors.size()*sizeof(float),cudaMemcpyHostToDevice,stream));
        checked(cudaMemcpyAsync(mask,selection.data(),selection.size(),cudaMemcpyHostToDevice,stream));
        checked(cudaStreamSynchronize(stream));
    }
};
std::vector<fs::MeshGPUVertex> download(const fs::MeshGPUBuffer& buffer) {
    // Validation only: the renderer keeps these vertices on the device.
    std::vector<fs::MeshGPUVertex> result(buffer.vertex_slots());
    checked(cudaMemcpy(result.data(),buffer.vertices(),result.size()*sizeof(result[0]),cudaMemcpyDeviceToHost));
    return result;
}
}
int main() {
    try {
        // Non-multiples of 16 exercise partial CUDA blocks and the last pixel.
        constexpr int width=17, height=19, count=width*height;
        Inputs device(count);
        std::vector<float> xyz(count*3,0), rgb(count*3);
        std::vector<unsigned char> mask(count,255);
        for (int i=0; i<count; ++i) {
            rgb[i]=i%256; rgb[count+i]=(i+51)%256; rgb[2*count+i]=(i+117)%256;
        }
        const std::vector<int> retained{0,100,count-1};
        for (int id : retained) { xyz[3*id]=float(id); xyz[3*id+1]=-float(id); xyz[3*id+2]=1.f; }
        xyz[3*4+2]=1; mask[4]=0; // A valid but unselected point.
        xyz[3*5+2]=-1;
        xyz[3*6]=std::numeric_limits<float>::quiet_NaN(); xyz[3*6+2]=1;
        xyz[3*7+1]=std::numeric_limits<float>::infinity(); xyz[3*7+2]=1;
        xyz[3*8+2]=std::numeric_limits<float>::infinity();
        device.upload(xyz,rgb,mask);
        fs::MeshGPUInputs inputs{device.xyz,device.rgb,device.mask,width,height,device.stream};
        fs::MeshGPUBuffer output;
        auto stats=fs::build_point_cloud_gpu(inputs,output);
        require(stats.point_count==retained.size() && stats.triangle_count==0 && stats.area_m2==0,"Wrong point statistics");
        require(output.vertex_slots()==count,"Point cloud must have one device slot per pixel");
        require(stats.low[0]==0 && stats.high[0]==count-1 && stats.low[1]==-(count-1) &&
                stats.high[1]==0 && stats.low[2]==1 && stats.high[2]==1,"Incorrect GPU point bounds");
        const auto vertices=download(output);
        for (int i=0; i<count; ++i) {
            const auto& v=vertices[i];
            if (std::find(retained.begin(),retained.end(),i)==retained.end()) {
                require(v.x==0 && v.y==0 && v.z==0 && v.r==0 && v.g==0 && v.b==0,"Invalid slot not cleared");
            } else {
                require(v.x==xyz[3*i] && v.y==xyz[3*i+1] && v.z==xyz[3*i+2],"Isolated valid point lost");
                require(std::abs(v.r-rgb[i]/255.f)<1e-6f && std::abs(v.g-rgb[count+i]/255.f)<1e-6f &&
                        std::abs(v.b-rgb[2*count+i]/255.f)<1e-6f,"RGB sample or normalization mismatch");
            }
        }
        // Reuse must erase old points, including reduction state and bounds.
        std::fill(mask.begin(),mask.end(),0);
        device.upload(xyz,rgb,mask);
        stats=fs::build_point_cloud_gpu(inputs,output);
        require(!stats.point_count && !stats.triangle_count,"Empty cloud retained old counts");
        for (int c=0; c<3; ++c) require(stats.low[c]==0 && stats.high[c]==0,"Empty cloud retained old bounds");
        for (auto v : download(output)) require(v.z==0,"Empty cloud retained old vertices");

        // Sharing the buffer and reduction implementation must preserve meshes.
        xyz={0,0,1, .01f,0,1, 0,.01f,1, .01f,.01f,1}; rgb.assign(12,128); mask.assign(4,255);
        device.upload(xyz,rgb,mask); inputs.width=inputs.height=2;
        stats=fs::build_mesh_gpu(inputs,output,.02,.01);
        require(stats.triangle_count==2 && stats.point_count==0 && output.vertex_slots()==6,"Mesh regression after point cloud reuse");
        stats=fs::build_point_cloud_gpu(inputs,output);
        require(stats.point_count==4 && stats.triangle_count==0 && output.vertex_slots()==4,"Point cloud regression after mesh reuse");
        inputs.width=inputs.height=1;
        stats=fs::build_point_cloud_gpu(inputs,output);
        require(stats.point_count==1 && output.vertex_slots()==1,"Single isolated point was rejected");
        inputs.stream=nullptr;
        bool rejected=false;
        try { fs::build_point_cloud_gpu(inputs,output); } catch (const std::invalid_argument&) { rejected=true; }
        require(rejected && output.vertex_slots()==0,"Invalid input retained a usable output");
        std::cout<<"PASS: GPU validity, masks, isolated points, RGB, partial blocks, bounds, empty clouds and mesh/buffer reuse\n";
    } catch (const std::exception& error) {
        std::cerr<<error.what()<<'\n'; return 1;
    }
}
