#include "fs/geometry/TsdfFusion.hpp"
#include <open3d/core/Tensor.h>
#include <open3d/t/geometry/VoxelBlockGrid.h>
#include <opencv2/imgproc.hpp>
#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>

namespace fs {
namespace {
using namespace open3d;
template<class T>
core::Tensor tensor(const T* data, const core::SizeVector& shape, core::Dtype dtype) {
    return core::Tensor(data,shape,dtype,core::Device("CPU:0"));
}
void summarize(TsdfGeometry& geometry, bool points) {
    auto& s=geometry.stats;
    if (geometry.vertices.empty()) return;
    for (int k=0;k<3;++k) { s.low[k]=std::numeric_limits<float>::max(); s.high[k]=-s.low[k]; }
    for (const auto& v:geometry.vertices) {
        const float p[]{v.x,v.y,v.z};
        for (int k=0;k<3;++k) {
            if (!std::isfinite(p[k])) throw std::runtime_error("TSDF produced non-finite geometry");
            s.low[k]=std::min(s.low[k],p[k]); s.high[k]=std::max(s.high[k],p[k]);
        }
    }
    if (points) s.point_count=geometry.vertices.size();
    else {
        s.triangle_count=geometry.vertices.size()/3;
        for (size_t i=0;i<geometry.vertices.size();i+=3) {
            const auto& a=geometry.vertices[i]; const auto& b=geometry.vertices[i+1]; const auto& c=geometry.vertices[i+2];
            s.area_m2+=.5*cv::norm(cv::Vec3d(b.x-a.x,b.y-a.y,b.z-a.z).cross(cv::Vec3d(c.x-a.x,c.y-a.y,c.z-a.z)));
        }
    }
}
TsdfGeometry extract(const core::Tensor& positions, const core::Tensor& colors, const core::Tensor* triangles=nullptr) {
    TsdfGeometry result;
    const auto p=positions.To(core::Device("CPU:0"),core::Float32).Contiguous();
    const auto c=colors.To(core::Device("CPU:0"),core::Float32).Contiguous();
    const auto* xyz=p.GetDataPtr<float>(); const auto* rgb=c.GetDataPtr<float>();
    // Match the reference SDK's float RGB 0..255 convention. Its float color
    // integration/extraction preserves that range; the GL renderer needs 0..1.
    const auto vertex=[&](int64_t i) {
        if (i<0 || i>=p.GetShape(0)) throw std::runtime_error("Invalid TSDF triangle index");
        return MeshGPUVertex{xyz[3*i],xyz[3*i+1],xyz[3*i+2],
            std::clamp(rgb[3*i]/255.f,0.f,1.f),std::clamp(rgb[3*i+1]/255.f,0.f,1.f),std::clamp(rgb[3*i+2]/255.f,0.f,1.f)};
    };
    if (triangles) {
        const auto indices=triangles->To(core::Device("CPU:0"),core::Int64).Contiguous();
        const auto count=indices.NumElements();
        result.vertices.reserve(count);
        for (int64_t i=0;i<count;++i) result.vertices.push_back(vertex(indices.GetDataPtr<int64_t>()[i]));
    } else {
        result.vertices.reserve(p.GetShape(0));
        for (int64_t i=0;i<p.GetShape(0);++i) result.vertices.push_back(vertex(i));
    }
    summarize(result,!triangles);
    return result;
}
}
void TsdfOptions::validate() const {
    if (!std::isfinite(voxel_size_mm) || voxel_size_mm<.1 || voxel_size_mm>10 ||
        !std::isfinite(truncation_voxels) || truncation_voxels<=0 ||
        block_resolution<1 || block_count<1 ||
        !std::isfinite(mesh_weight_threshold) || mesh_weight_threshold<0 ||
        !std::isfinite(point_weight_threshold) || point_weight_threshold<0)
        throw std::invalid_argument("Invalid TSDF settings (voxel size must be 0.1–10 mm)");
}
TsdfResult fuse_tsdf(const std::vector<TsdfView>& views, const TsdfOptions& options,
                     const std::function<void(int,const char*)>& progress) {
    options.validate();
    if (views.empty()) throw std::invalid_argument("TSDF needs captured depth and camera poses");
    const auto report=[&](int percent,const char* message) { if (progress) progress(percent,message); };
    report(0,"Allocating TSDF on CUDA:0…");
    const core::Device device("CUDA:0");
    t::geometry::VoxelBlockGrid grid({"tsdf","weight","color"},
        {core::Float32,core::Float32,core::Float32},{{1},{1},{3}},
        float(options.voxel_size_mm*.001),options.block_resolution,options.block_count,device);
    TsdfResult result;
    for (size_t i=0;i<views.size();++i) {
        report(5+int(65*i/views.size()),"Integrating masked RGB-D into TSDF…");
        const auto& v=views[i];
        if (v.depth.empty() || v.depth.type()!=CV_32FC1 || v.rgb.type()!=CV_8UC3 || v.rgb.size()!=v.depth.size() ||
            !cv::checkRange(cv::Mat(v.intrinsics)) || v.intrinsics(0,0)<=0 || v.intrinsics(1,1)<=0 ||
            !cv::checkRange(cv::Mat(v.camera_to_world)))
            throw std::invalid_argument("TSDF needs aligned RGB/depth, finite intrinsics and camera poses");
        const auto& pose=v.camera_to_world;
        const cv::Matx33d rotation(pose(0,0),pose(0,1),pose(0,2),pose(1,0),pose(1,1),pose(1,2),pose(2,0),pose(2,1),pose(2,2));
        if (cv::norm(rotation.t()*rotation-cv::Matx33d::eye())>1e-3 ||
            std::abs(cv::determinant(rotation)-1)>1e-3 ||
            cv::norm(cv::Vec4d(pose(3,0),pose(3,1),pose(3,2),pose(3,3))-cv::Vec4d(0,0,0,1))>1e-6)
            throw std::invalid_argument("TSDF camera-to-world pose must be a rigid transform in metres");
        cv::Mat mask;
        if (!v.mask.empty()) {
            if (v.mask.type()!=CV_8UC1) throw std::invalid_argument("TSDF mask must be uint8 grayscale");
            if (cv::countNonZero(v.mask)) cv::resize(v.mask,mask,v.depth.size(),0,0,cv::INTER_NEAREST);
        }
        cv::Mat depth=v.depth.clone(), color;
        float max_depth=0;
        for (int y=0;y<depth.rows;++y) for (int x=0;x<depth.cols;++x) {
            float& z=depth.at<float>(y,x);
            if (!std::isfinite(z) || z<=0 || (!mask.empty() && !mask.at<unsigned char>(y,x))) z=0;
            max_depth=std::max(max_depth,z);
        }
        if (!(max_depth>0)) continue;
        // Preserve the entire already-filtered FS depth range. Units remain metres.
        max_depth=std::nextafter(max_depth,std::numeric_limits<float>::infinity());
        v.rgb.convertTo(color,CV_32FC3);
        const auto k=tensor(v.intrinsics.val,{3,3},core::Float64);
        const cv::Matx44d world_to_camera=v.camera_to_world.inv();
        const auto extrinsic=tensor(world_to_camera.val,{4,4},core::Float64);
        const t::geometry::Image d(tensor(depth.ptr<float>(),{depth.rows,depth.cols,1},core::Float32).To(device));
        const t::geometry::Image rgb(tensor(color.ptr<float>(),{color.rows,color.cols,3},core::Float32).To(device));
        const auto blocks=grid.GetUniqueBlockCoordinates(d,k,extrinsic,1.f,max_depth,float(options.truncation_voxels));
        if (!blocks.GetShape(0)) continue;
        grid.Integrate(blocks,d,rgb,k,extrinsic,1.f,max_depth,float(options.truncation_voxels));
        ++result.integrated_views;
    }
    result.active_blocks=grid.GetHashMap().Size();
    if (!result.integrated_views || !result.active_blocks)
        throw std::runtime_error("TSDF: no valid depth remains inside the selected masks");
    report(75,"Extracting TSDF mesh…");
    const auto mesh=grid.ExtractTriangleMesh(float(options.mesh_weight_threshold),-1);
    if (mesh.HasTriangleIndices() && mesh.HasVertexPositions())
        result.mesh=extract(mesh.GetVertexPositions(),mesh.GetVertexColors(),&mesh.GetTriangleIndices());
    report(88,"Extracting TSDF point cloud…");
    const auto points=grid.ExtractPointCloud(float(options.point_weight_threshold),-1);
    if (points.HasPointPositions()) result.points=extract(points.GetPointPositions(),points.GetPointColors());
    report(96,"Preparing TSDF display…");
    return result;
}
} // namespace fs
