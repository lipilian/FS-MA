#include "fs/geometry/ICPPoseRefinement.hpp"
#include <opencv2/core.hpp>
#include <open3d/t/geometry/PointCloud.h>
#include <open3d/t/pipelines/registration/Registration.h>
#include <open3d/t/pipelines/registration/RobustKernel.h>
#include <open3d/t/pipelines/registration/TransformationEstimation.h>
#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>

namespace fs {
namespace {
namespace core = open3d::core;
namespace reg = open3d::t::pipelines::registration;
using Cloud = open3d::t::geometry::PointCloud;
constexpr double fine_voxel = .001;

core::Tensor tensor(const cv::Matx44d& pose) {
    return core::Tensor(pose.val, {4,4}, core::Float64);
}
cv::Matx44d matrix(const core::Tensor& pose) {
    const auto cpu=pose.To(core::Device("CPU:0"),core::Float64).Contiguous();
    if (cpu.GetShape()!=core::SizeVector{4,4}) throw std::runtime_error("Invalid ICP pose shape");
    cv::Matx44d result;
    std::copy_n(cpu.GetDataPtr<double>(),16,result.val);
    return result;
}
bool rigid(const cv::Matx44d& pose) {
    for (double v:pose.val) if (!std::isfinite(v)) return false;
    if (std::abs(pose(3,0))+std::abs(pose(3,1))+std::abs(pose(3,2))+std::abs(pose(3,3)-1)>1e-6) return false;
    const cv::Matx33d r(pose(0,0),pose(0,1),pose(0,2),pose(1,0),pose(1,1),pose(1,2),pose(2,0),pose(2,1),pose(2,2));
    return cv::norm(r*r.t()-cv::Matx33d::eye())<1e-4 && std::abs(cv::determinant(r)-1)<1e-4;
}
Cloud cloud(const ICPView& view) {
    const core::Device device(view.device<0 ? "CPU:0" : "CUDA:"+std::to_string(view.device));
    if (!view.selection_mask.empty() && view.selection_mask.size()!=view.vertex_count)
        throw std::runtime_error("ICP selection mask does not match the point buffer");
    if (!view.vertex_count) return Cloud(device);
    if (!view.vertices || view.vertex_count>std::size_t(std::numeric_limits<int64_t>::max()))
        throw std::runtime_error("Invalid ICP point buffer");
    if (view.device>=0) {
        const auto selected=cudaSetDevice(view.device);
        if (selected!=cudaSuccess) throw std::runtime_error(cudaGetErrorString(selected));
        const auto synced=cudaStreamSynchronize(view.stream);
        if (synced!=cudaSuccess) throw std::runtime_error(cudaGetErrorString(synced));
    }
    static_assert(sizeof(MeshGPUVertex)==6*sizeof(float));
    const auto count=static_cast<int64_t>(view.vertex_count);
    // A read-only strided XYZ view into the interleaved FS point buffer.
    // SelectByMask creates owned storage before normals/transforms are applied.
    const core::Tensor positions(const_cast<MeshGPUVertex*>(view.vertices),core::Float32,{count,3},{6,1},device);
    auto valid=positions.IsFinite().All(core::SizeVector{1}).LogicalAnd(positions.IndexExtract(1,2).Gt(0));
    if (!view.selection_mask.empty()) {
        const core::Tensor selection(view.selection_mask,{count},core::UInt8,device);
        valid=valid.LogicalAnd(selection.Ne(0));
    }
    return Cloud(positions).SelectByMask(valid);
}
}

std::vector<ICPViewResult> refine_camera_poses_icp(
    const std::vector<ICPView>& views, const std::function<void()>& checkpoint,
    const std::function<void(std::size_t, const ICPViewResult&)>& on_view) {
    const auto check=[&] { if (checkpoint) checkpoint(); };
    check();
    if (views.size()<2) throw std::runtime_error("ICP requires at least two completed views");
    std::vector<Cloud> sources;
    std::vector<ICPViewResult> results(views.size());
    for (std::size_t i=0;i<views.size();++i) {
        check();
        if (views[i].device!=views.front().device || !rigid(views[i].camera_to_world))
            throw std::runtime_error("ICP requires rigid camera-to-world poses and points on one device");
        results[i].camera_to_world=views[i].camera_to_world;
        sources.push_back(cloud(views[i]));
        results[i].point_count=sources.back().IsEmpty() ? 0 : sources.back().GetPointPositions().GetLength();
    }
    if (sources.front().IsEmpty() || sources.front().GetPointPositions().GetLength()<30)
        throw std::runtime_error("ICP reference view has too few valid FS points");
    results.front().reference=true;
    results.front().message="Reference view; pose unchanged";
    if (on_view) on_view(0,results.front());
    const reg::RobustKernel kernel(reg::RobustKernelMethod::TukeyLoss,.01);
    const reg::TransformationEstimationPointToPlane estimation(kernel);
    const std::vector<double> voxels{.004,.002,.001}, distances{.025,.010,.004};
    const std::vector<reg::ICPConvergenceCriteria> criteria{{1e-6,1e-6,50},{1e-6,1e-6,30},{1e-6,1e-6,14}};
    auto merged=sources.front().Clone();
    merged.Transform(tensor(results.front().camera_to_world));
    for (std::size_t i=1;i<views.size();++i) {
        check();
        auto& result=results[i];
        try {
            auto target=merged.VoxelDownSample(fine_voxel);
            if (sources[i].IsEmpty() || sources[i].GetPointPositions().GetLength()<30 ||
                target.IsEmpty() || target.GetPointPositions().GetLength()<30) {
                result.message="Too few valid FS points; MA pose retained";
            } else {
                target.EstimateNormals(30,.006);
                check();
                const auto fit=reg::MultiScaleICP(sources[i],target,voxels,criteria,distances,
                    tensor(views[i].camera_to_world),estimation,[&](const auto&) { check(); });
                check();
                result.computed=true;
                result.fitness=fit.fitness_;
                result.rmse_m=fit.inlier_rmse_;
                const auto refined=matrix(fit.transformation_);
                const auto& initial=views[i].camera_to_world;
                double rotation_trace=0, translation_squared=0;
                for (int r=0;r<3;++r) {
                    translation_squared+=std::pow(refined(r,3)-initial(r,3),2);
                    for (int c=0;c<3;++c) rotation_trace+=refined(r,c)*initial(r,c);
                }
                result.translation_m=std::sqrt(translation_squared);
                result.rotation_deg=std::acos(std::clamp((rotation_trace-1)*.5,-1.,1.))*180/std::acos(-1.);
                result.accepted=rigid(refined) && std::isfinite(result.fitness) && std::isfinite(result.rmse_m) &&
                    result.fitness>=.3 && result.rmse_m<=.004 && result.translation_m<=.03 && result.rotation_deg<=10;
                if (result.accepted) result.camera_to_world=refined;
                result.message=result.accepted ? "Accepted" : "Rejected; MA pose retained";
            }
        } catch (const std::runtime_error& e) {
            check(); // Propagate cancellation rather than treating it as rejected ICP.
            result.message=std::string("ICP failed; MA pose retained: ")+e.what();
        }
        check();
        if (on_view) on_view(i,result);
        // Match tbsa_panorama's RegisteredViews target, including rejected views
        // at their original MA pose. Never transform the borrowed input in place.
        if (i+1<views.size() && !sources[i].IsEmpty()) {
            auto world=sources[i].Clone();
            world.Transform(tensor(result.camera_to_world));
            merged=merged.Append(world);
        }
    }
    check();
    return results;
}
} // namespace fs
