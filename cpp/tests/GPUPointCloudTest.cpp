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
void checkMeshEdgeFilter() {
    Inputs device(4);
    fs::MeshGPUInputs inputs{device.xyz,device.rgb,device.mask,2,2,device.stream};
    fs::MeshGPUBuffer output;
    const std::vector<float> rgb(12,128);
    const std::vector<unsigned char> mask{255,255,255,0};
    // A 10 mm edge and two shorter edges; the fourth cell corner is unselected.
    const std::vector<float> triangle{0,0,1, .01f,0,1, .005f,.004f,1, 0,0,0};
    device.upload(triangle,rgb,mask);
    auto stats=fs::build_mesh_gpu(inputs,output);
    require(stats.triangle_count==1 && stats.area_m2>0,"10 mm edge should survive the default mesh filter");

    auto points=triangle;
    points[3]=std::nextafter(.01f,std::numeric_limits<float>::infinity());
    device.upload(points,rgb,mask);
    stats=fs::build_mesh_gpu(inputs,output);
    require(stats.triangle_count==0 && stats.area_m2==0,"Edge just above 10 mm was not rejected");
    for (auto v : download(output)) require(v.z==0,"Rejected triangle left stale GPU vertices");

    // Even with 8 mm horizontal/vertical edges, the cell diagonal is too long.
    points={0,0,1, .008f,0,1, 0,.008f,1, .008f,.008f,1};
    device.upload(points,rgb,{255,255,255,255});
    require(fs::build_mesh_gpu(inputs,output).triangle_count==0,"Mesh filter did not check the diagonal edge");

    points=triangle; points[7]=0; // Three collinear points, all edges <= 10 mm.
    device.upload(points,rgb,mask);
    require(fs::build_mesh_gpu(inputs,output).triangle_count==0,"Zero-area triangle was retained");
    for (const float invalid : {std::numeric_limits<float>::quiet_NaN(),std::numeric_limits<float>::infinity()}) {
        points=triangle; points[0]=invalid;
        device.upload(points,rgb,mask);
        require(fs::build_mesh_gpu(inputs,output).triangle_count==0,"Nonfinite triangle was retained");
    }

    // A caller may choose a larger edge limit. A 12 mm Z span must not trigger
    // the removed 10 mm depth-jump filter when all three 3D edges fit the limit.
    points={0,0,1, .001f,0,1.012f, 0,.001f,1.006f, 0,0,0};
    device.upload(points,rgb,mask);
    require(fs::build_mesh_gpu(inputs,output,.02).triangle_count==1,"Mesh still applies an independent depth-jump filter");
    require(fs::build_mesh_gpu(inputs,output).triangle_count==0,"Edge filtering ignored the Z component of 3D distance");
    std::cout<<"PASS: 10 mm mesh edge boundary, diagonal/3D edges, zero area, nonfinite vertices and no depth-jump filter\n";
}
}
int main() {
    try {
        checkMeshEdgeFilter();
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
        stats=fs::build_mesh_gpu(inputs,output,.02);
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
