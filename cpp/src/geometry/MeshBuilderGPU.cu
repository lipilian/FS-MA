#include "fs/geometry/MeshBuilderGPU.hpp"
#include <cmath>
#include <limits>
#include <stdexcept>
#include <string>

namespace fs {
namespace {
constexpr int side = 16;
constexpr int threads = side * side;
void checked(cudaError_t error, const char* operation) {
    if (error != cudaSuccess)
        throw std::runtime_error(std::string(operation) + ": " + cudaGetErrorString(error));
}
template<class T> struct DeviceBuffer {
    T* data{nullptr};
    std::size_t capacity{0};
    ~DeviceBuffer() { if (data) cudaFree(data); }
    void reserve(std::size_t count) {
        if (count <= capacity) return;
        T* next = nullptr;
        checked(cudaMalloc(reinterpret_cast<void**>(&next), count * sizeof(T)), "allocate GPU mesh buffer");
        if (data) cudaFree(data);
        data = next; capacity = count;
    }
};
__device__ double3 subtract(float3 a, float3 b) {
    return make_double3(double(a.x)-b.x, double(a.y)-b.y, double(a.z)-b.z);
}
__device__ double squared(double3 p) { return p.x*p.x + p.y*p.y + p.z*p.z; }
__device__ bool valid(float3 p) {
    return isfinite(p.x) && isfinite(p.y) && isfinite(p.z) && p.z > 0;
}
__device__ double triangle(const MeshGPUInputs& in, const int* ids, const float3* p,
                           int a, int b, int c, double edge, double jump,
                           MeshGPUVertex* out) {
    const double low = fmin(double(p[a].z), fmin(double(p[b].z), double(p[c].z)));
    const double high = fmax(double(p[a].z), fmax(double(p[b].z), double(p[c].z)));
    if (high-low > jump) return 0;
    const auto ab = subtract(p[b],p[a]), ac = subtract(p[c],p[a]), bc = subtract(p[c],p[b]);
    if (sqrt(squared(ab)) > edge || sqrt(squared(ac)) > edge || sqrt(squared(bc)) > edge) return 0;
    const auto cross = make_double3(ab.y*ac.z-ab.z*ac.y, ab.z*ac.x-ab.x*ac.z, ab.x*ac.y-ab.y*ac.x);
    const double area = .5 * sqrt(squared(cross));
    if (!(area > 0) || !isfinite(area)) return 0;
    const int corners[3] = {a,b,c};
    const int plane = in.width * in.height;
    for (int k=0; k<3; ++k) {
        const int n = corners[k], id = ids[n];
        out[k] = {p[n].x, p[n].y, p[n].z,
                  in.rgb[id]/255.f, in.rgb[plane+id]/255.f, in.rgb[2*plane+id]/255.f};
    }
    return area;
}
__device__ MeshGPUStats empty_stats() {
    MeshGPUStats s{};
    for (int c=0;c<3;++c) { s.low[c]=INFINITY; s.high[c]=-INFINITY; }
    return s;
}
__device__ void merge_stats(MeshGPUStats& a, const MeshGPUStats& b) {
    a.area_m2+=b.area_m2; a.triangle_count+=b.triangle_count;
    a.point_count+=b.point_count;
    for (int c=0;c<3;++c) { a.low[c]=fminf(a.low[c],b.low[c]); a.high[c]=fmaxf(a.high[c],b.high[c]); }
}
__global__ void build_cells(MeshGPUInputs in, MeshGPUVertex* vertices, MeshGPUStats* partials,
                            double edge, double jump) {
    const int x = blockIdx.x*blockDim.x+threadIdx.x, y = blockIdx.y*blockDim.y+threadIdx.y;
    const int tid = threadIdx.y*side+threadIdx.x;
    double area = 0;
    unsigned long long count = 0;
    MeshGPUStats stats = empty_stats();
    if (x < in.width-1 && y < in.height-1) {
        auto* out = vertices + (std::size_t(y)*(in.width-1)+x)*6;
        for (int i=0; i<6; ++i) out[i] = {};
        // Cyclic order A,B,D,C gives consistent winding, also for three corners.
        const int ids[4] = {y*in.width+x, y*in.width+x+1, (y+1)*in.width+x+1, (y+1)*in.width+x};
        float3 p[4]; int keep[4], n = 0;
        for (int i=0; i<4; ++i) {
            const int id = ids[i];
            p[i] = make_float3(in.xyz[3*id],in.xyz[3*id+1],in.xyz[3*id+2]);
            if (in.mask[id] && valid(p[i])) keep[n++] = i;
        }
        if (n == 4) {
            const double first = triangle(in,ids,p,0,1,2,edge,jump,out);
            const double second = triangle(in,ids,p,0,2,3,edge,jump,out+3);
            area = first+second; count = (first>0)+(second>0);
        } else if (n == 3) {
            area = triangle(in,ids,p,keep[0],keep[1],keep[2],edge,jump,out);
            count = area>0;
        }
        if (count) for (int i=0;i<6;++i) if (out[i].z>0) {
            const float p[3]={out[i].x,out[i].y,out[i].z};
            for (int c=0;c<3;++c) { stats.low[c]=fminf(stats.low[c],p[c]); stats.high[c]=fmaxf(stats.high[c],p[c]); }
        }
    }
    // All threads participate, including those outside the image.
    __shared__ MeshGPUStats values[threads];
    stats.area_m2=area; stats.triangle_count=count;
    values[tid]=stats;
    __syncthreads();
    for (int stride=threads/2; stride>0; stride/=2) {
        if (tid<stride) merge_stats(values[tid],values[tid+stride]);
        __syncthreads();
    }
    if (tid==0) partials[blockIdx.y*gridDim.x+blockIdx.x]=values[0];
}
__global__ void build_points(MeshGPUInputs in, MeshGPUVertex* vertices, MeshGPUStats* partials) {
    const int x=blockIdx.x*blockDim.x+threadIdx.x, y=blockIdx.y*blockDim.y+threadIdx.y;
    const int tid=threadIdx.y*side+threadIdx.x;
    MeshGPUStats stats=empty_stats();
    if (x<in.width && y<in.height) {
        const int id=y*in.width+x, plane=in.width*in.height;
        const float3 p=make_float3(in.xyz[3*id],in.xyz[3*id+1],in.xyz[3*id+2]);
        vertices[id]={};
        if (in.mask[id] && valid(p)) {
            vertices[id]={p.x,p.y,p.z,in.rgb[id]/255.f,in.rgb[plane+id]/255.f,in.rgb[2*plane+id]/255.f};
            stats.point_count=1;
            stats.low[0]=stats.high[0]=p.x;
            stats.low[1]=stats.high[1]=p.y;
            stats.low[2]=stats.high[2]=p.z;
        }
    }
    __shared__ MeshGPUStats values[threads];
    values[tid]=stats;
    __syncthreads();
    for (int stride=threads/2; stride>0; stride/=2) {
        if (tid<stride) merge_stats(values[tid],values[tid+stride]);
        __syncthreads();
    }
    if (tid==0) partials[blockIdx.y*gridDim.x+blockIdx.x]=values[0];
}
__global__ void build_ma_points(MAPointCloudGPUInputs in, MeshGPUVertex* vertices, MeshGPUStats* partials) {
    const int x=blockIdx.x*blockDim.x+threadIdx.x, y=blockIdx.y*blockDim.y+threadIdx.y;
    const int tid=threadIdx.y*side+threadIdx.x;
    MeshGPUStats stats=empty_stats();
    if (x<in.width && y<in.height) {
        const int id=y*in.width+x, plane=in.width*in.height;
        vertices[id]={};
        const float rx=in.dense[id], ry=in.dense[plane+id], rz=in.dense[2*plane+id];
        const float norm=sqrtf(rx*rx+ry*ry+rz*rz);
        const float raw_depth=in.dense[3*plane+id], raw_scale=*in.scale;
        const float scale=fmaxf(expf(raw_scale),1e-8f);
        const float distance=expf(raw_depth)*scale;
        const float mask=in.dense[5*plane+id];
        // Channel 6 is the actual MA input ray distance on this 518x434 grid.
        // An invalid input pixel must stay empty even if MA predicts a surface.
        const float input_depth=in.input[6*plane+id];
        const float r=in.input[id]*.229f+.485f;
        const float g=in.input[plane+id]*.224f+.456f;
        const float b=in.input[2*plane+id]*.225f+.406f;
        // Match the unit-sphere adaptor, including its epsilon for tiny rays.
        const float divisor=fmaxf(norm,1e-8f);
        const float3 p=make_float3(rx/divisor*distance,ry/divisor*distance,rz/divisor*distance);
        if (isfinite(input_depth) && input_depth>0 &&
            isfinite(norm) && norm>0 && isfinite(raw_depth) && isfinite(raw_scale) &&
            isfinite(scale) && isfinite(distance) && distance>0 &&
            isfinite(mask) && mask>0 && valid(p) && isfinite(r) && isfinite(g) && isfinite(b)) {
            vertices[id]={p.x,p.y,p.z,fminf(1.f,fmaxf(0.f,r)),fminf(1.f,fmaxf(0.f,g)),fminf(1.f,fmaxf(0.f,b))};
            stats.point_count=1;
            stats.low[0]=stats.high[0]=p.x;
            stats.low[1]=stats.high[1]=p.y;
            stats.low[2]=stats.high[2]=p.z;
        }
    }
    __shared__ MeshGPUStats values[threads];
    values[tid]=stats;
    __syncthreads();
    for (int stride=threads/2; stride>0; stride/=2) {
        if (tid<stride) merge_stats(values[tid],values[tid+stride]);
        __syncthreads();
    }
    if (tid==0) partials[blockIdx.y*gridDim.x+blockIdx.x]=values[0];
}
__global__ void reduce_stats(const MeshGPUStats* partials, int size, MeshGPUStats* total) {
    const int tid=threadIdx.x;
    MeshGPUStats stats=empty_stats();
    for (int i=tid;i<size;i+=threads) merge_stats(stats,partials[i]);
    __shared__ MeshGPUStats values[threads];
    values[tid]=stats;
    __syncthreads();
    for (int stride=threads/2; stride>0; stride/=2) {
        if (tid<stride) merge_stats(values[tid],values[tid+stride]);
        __syncthreads();
    }
    if (tid==0) *total=(values[0].triangle_count || values[0].point_count) ? values[0] : MeshGPUStats{};
}
} // namespace
struct MeshGPUBuffer::Impl {
    DeviceBuffer<MeshGPUVertex> vertices;
    DeviceBuffer<MeshGPUStats> partials, total;
    std::size_t slots{0};
};
MeshGPUBuffer::MeshGPUBuffer() : impl_(std::make_unique<Impl>()) {}
MeshGPUBuffer::~MeshGPUBuffer() = default;
const MeshGPUVertex* MeshGPUBuffer::vertices() const { return impl_->vertices.data; }
std::size_t MeshGPUBuffer::vertex_slots() const { return impl_->slots; }

MeshGPUStats build_point_cloud_gpu(const MeshGPUInputs& inputs, MeshGPUBuffer& output) {
    output.impl_->slots=0;
    if (!inputs.xyz || !inputs.rgb || !inputs.mask || !inputs.stream || inputs.width<1 || inputs.height<1 ||
        std::size_t(inputs.width)*inputs.height>std::size_t(std::numeric_limits<int>::max()/3))
        throw std::invalid_argument("GPU point cloud needs aligned device XYZ, RGB and mask, positive dimensions and the FS stream");
    const dim3 block(side,side), grid((inputs.width+side-1)/side,(inputs.height+side-1)/side);
    auto& storage=*output.impl_;
    const std::size_t slots=std::size_t(inputs.width)*inputs.height;
    storage.vertices.reserve(slots);
    storage.partials.reserve(grid.x*grid.y);
    storage.total.reserve(1);
    MeshGPUStats result;
    try {
        build_points<<<grid,block,0,inputs.stream>>>(inputs,storage.vertices.data,storage.partials.data);
        checked(cudaGetLastError(), "launch GPU point cloud");
        reduce_stats<<<1,threads,0,inputs.stream>>>(storage.partials.data,grid.x*grid.y,storage.total.data);
        checked(cudaGetLastError(), "launch GPU point cloud reduction");
        checked(cudaMemcpyAsync(&result,storage.total.data,sizeof(result),cudaMemcpyDeviceToHost,inputs.stream), "download GPU point cloud statistics");
        checked(cudaStreamSynchronize(inputs.stream), "complete GPU point cloud");
    } catch (...) {
        cudaStreamSynchronize(inputs.stream);
        throw;
    }
    storage.slots=slots;
    return result;
}

MeshGPUStats build_ma_point_cloud_gpu(const MAPointCloudGPUInputs& inputs, MeshGPUBuffer& output) {
    output.impl_->slots=0;
    if (!inputs.dense || !inputs.input || !inputs.scale || !inputs.stream || inputs.width<1 || inputs.height<1 ||
        std::size_t(inputs.width)*inputs.height>std::size_t(std::numeric_limits<int>::max()/7))
        throw std::invalid_argument("MA point cloud needs device dense output, RGB input, scale, positive dimensions and the MA stream");
    const dim3 block(side,side), grid((inputs.width+side-1)/side,(inputs.height+side-1)/side);
    auto& storage=*output.impl_;
    const std::size_t slots=std::size_t(inputs.width)*inputs.height;
    storage.vertices.reserve(slots);
    storage.partials.reserve(grid.x*grid.y);
    storage.total.reserve(1);
    MeshGPUStats result;
    try {
        build_ma_points<<<grid,block,0,inputs.stream>>>(inputs,storage.vertices.data,storage.partials.data);
        checked(cudaGetLastError(), "decode MA GPU point cloud");
        reduce_stats<<<1,threads,0,inputs.stream>>>(storage.partials.data,grid.x*grid.y,storage.total.data);
        checked(cudaGetLastError(), "reduce MA GPU point cloud");
        checked(cudaMemcpyAsync(&result,storage.total.data,sizeof(result),cudaMemcpyDeviceToHost,inputs.stream), "download MA point cloud statistics");
        checked(cudaStreamSynchronize(inputs.stream), "complete MA GPU point cloud");
    } catch (...) {
        cudaStreamSynchronize(inputs.stream);
        throw;
    }
    storage.slots=slots;
    return result;
}

MeshGPUStats build_mesh_gpu(const MeshGPUInputs& inputs, MeshGPUBuffer& output,
                            double max_edge_m, double max_depth_jump_m) {
    output.impl_->slots = 0;
    if (!inputs.xyz || !inputs.rgb || !inputs.mask || !inputs.stream || inputs.width < 2 || inputs.height < 2 ||
        std::size_t(inputs.width)*inputs.height > std::size_t(std::numeric_limits<int>::max()/3) ||
        !std::isfinite(max_edge_m) || max_edge_m <= 0 ||
        !std::isfinite(max_depth_jump_m) || max_depth_jump_m <= 0)
        throw std::invalid_argument("GPU mesh needs aligned device XYZ, RGB and mask, dimensions >=2, the FS stream and positive thresholds");
    const dim3 block(side,side), grid((inputs.width+side-1)/side,(inputs.height+side-1)/side);
    const std::size_t slots = std::size_t(inputs.width-1)*(inputs.height-1)*6;
    auto& storage = *output.impl_;
    storage.vertices.reserve(slots);
    storage.partials.reserve(grid.x*grid.y);
    storage.total.reserve(1);
    MeshGPUStats result;
    try {
        build_cells<<<grid,block,0,inputs.stream>>>(inputs,storage.vertices.data,storage.partials.data,max_edge_m,max_depth_jump_m);
        checked(cudaGetLastError(), "launch GPU mesh cells");
        reduce_stats<<<1,threads,0,inputs.stream>>>(storage.partials.data,grid.x*grid.y,storage.total.data);
        checked(cudaGetLastError(), "launch GPU mesh area reduction");
        checked(cudaMemcpyAsync(&result,storage.total.data,sizeof(result),cudaMemcpyDeviceToHost,inputs.stream), "download GPU mesh statistics");
        checked(cudaStreamSynchronize(inputs.stream), "complete GPU mesh");
    } catch (...) {
        // Do not let an exceptional return leave kernels using the caller's inputs.
        cudaStreamSynchronize(inputs.stream);
        throw;
    }
    storage.slots = slots;
    return result;
}
} // namespace fs
