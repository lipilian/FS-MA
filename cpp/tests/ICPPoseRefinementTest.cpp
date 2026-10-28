#include "fs/geometry/ICPPoseRefinement.hpp"
#include <opencv2/core.hpp>
#include <cmath>
#include <cstring>
#include <iostream>
#include <limits>
#include <stdexcept>

namespace {
void require(bool ok,const char* message) { if (!ok) throw std::runtime_error(message); }
void checked(cudaError_t error) { if (error!=cudaSuccess) throw std::runtime_error(cudaGetErrorString(error)); }
cv::Matx44d pose(double angle,double x,double y,double z) {
    const auto c=std::cos(angle),s=std::sin(angle);
    return {c,0,s,x, 0,1,0,y, -s,0,c,z, 0,0,0,1};
}
std::vector<fs::MeshGPUVertex> surface(const cv::Matx44d& camera,const cv::Matx44d& anchor) {
    std::vector<fs::MeshGPUVertex> points;
    const auto transform=camera.inv()*anchor;
    for (int y=0;y<81;++y) for (int x=0;x<101;++x) {
        const double px=(x-50)*.0015, py=(y-40)*.0015;
        const double z=.5+.018*std::sin(px*71)+.012*std::cos(py*83)+.008*std::sin((px+py)*53);
        const auto p=transform*cv::Vec4d(px,py,z,1);
        points.push_back({float(p[0]),float(p[1]),float(p[2]),.2f,.4f,.8f});
    }
    points.push_back({0,0,0,0,0,0});
    points.push_back({std::numeric_limits<float>::quiet_NaN(),0,1,0,0,0});
    return points;
}
struct DevicePoints {
    fs::MeshGPUVertex* ptr{nullptr};
    ~DevicePoints() { if (ptr) cudaFree(ptr); }
};
void run(bool gpu) {
    const auto anchor=pose(.23,.21,-.12,.31);
    const auto truth=anchor*pose(-.04,.018,.006,-.012);
    const auto third=anchor*pose(.035,-.014,-.004,.008);
    auto a=surface(anchor,anchor),b=surface(truth,anchor),c=surface(third,anchor);
    const auto original=b;
    std::vector<fs::ICPView> views(4);
    views[0].camera_to_world=anchor;
    views[1].camera_to_world=truth*pose(.012,.004,-.002,.003);
    views[2].camera_to_world=third*pose(-.009,-.003,.002,-.002);
    views[3].camera_to_world=pose(0,2,0,0); // No overlap, must retain this MA pose.
    const std::vector<const std::vector<fs::MeshGPUVertex>*> data{&a,&b,&c,&b};
    DevicePoints buffers[4];
    for (std::size_t i=0;i<views.size();++i) {
        views[i].vertex_count=data[i]->size();
        views[i].vertices=data[i]->data();
        if (gpu) {
            checked(cudaMalloc(reinterpret_cast<void**>(&buffers[i].ptr),data[i]->size()*sizeof(fs::MeshGPUVertex)));
            checked(cudaMemcpy(buffers[i].ptr,data[i]->data(),data[i]->size()*sizeof(fs::MeshGPUVertex),cudaMemcpyHostToDevice));
            views[i].vertices=buffers[i].ptr; views[i].device=0;
        }
    }
    const auto results=fs::refine_camera_poses_icp(views);
    require(results[0].reference && results[0].camera_to_world==anchor,"ICP moved the reference camera");
    for (int i=1;i<=2;++i) {
        std::cout<<(gpu ? "GPU" : "CPU")<<" view "<<i+1<<": "<<results[i].message<<", fitness "<<results[i].fitness
                 <<", RMSE "<<results[i].rmse_m<<"\n";
        require(results[i].accepted,"ICP rejected a recoverable pose perturbation");
        require(cv::norm(results[i].camera_to_world-(i==1 ? truth : third))<.003,"ICP returned the wrong camera-to-world transform");
    }
    require(!results[3].accepted && results[3].camera_to_world==views[3].camera_to_world,
            "ICP replaced a non-overlapping view's MA pose");
    auto limited_views=views;
    limited_views.resize(2);
    const auto strict=fs::refine_camera_poses_icp(limited_views,.001);
    require(strict[1].computed && strict[1].translation_m>.001 && !strict[1].accepted &&
            strict[1].camera_to_world==limited_views[1].camera_to_world,
            "A tighter translation limit did not reject the correction and retain the MA pose");
    const auto relaxed=fs::refine_camera_poses_icp(limited_views,.02);
    require(relaxed[1].accepted && cv::norm(relaxed[1].camera_to_world-truth)<.003,
            "Increasing the translation limit did not accept the same recoverable correction");
    for (double invalid : {-1.,std::numeric_limits<double>::infinity(),std::numeric_limits<double>::quiet_NaN()}) {
        bool refused=false;
        try { fs::refine_camera_poses_icp(limited_views,invalid); }
        catch (const std::invalid_argument&) { refused=true; }
        require(refused,"ICP accepted a negative or nonfinite translation limit");
    }
    // Imported scans retain full point buffers; their current masks are applied
    // only when ICP starts, including to the reference/target cloud.
    auto masked_views=views;
    masked_views.resize(2);
    std::size_t selected_counts[2]{};
    for (int i=0;i<2;++i) {
        auto& mask=masked_views[i].selection_mask;
        mask.assign(masked_views[i].vertex_count,0);
        for (std::size_t j=i;j<mask.size();j+=2) {
            mask[j]=i ? 255 : 1;
            const auto& p=(*data[i])[j];
            selected_counts[i]+=std::isfinite(p.x) && std::isfinite(p.y) && std::isfinite(p.z) && p.z>0;
        }
    }
    const auto masked=fs::refine_camera_poses_icp(masked_views);
    require(masked[0].point_count==selected_counts[0] && masked[1].point_count==selected_counts[1],
            "ICP did not intersect each view's current mask with valid XYZ");
    require(masked[1].accepted && cv::norm(masked[1].camera_to_world-truth)<.003,
            "Masked ICP failed to recover the camera pose");
    std::fill(masked_views[1].selection_mask.begin(),masked_views[1].selection_mask.end(),0);
    const auto empty_source=fs::refine_camera_poses_icp(masked_views);
    require(empty_source[1].point_count==0 && !empty_source[1].accepted &&
            empty_source[1].camera_to_world==masked_views[1].camera_to_world,
            "An empty source mask fell back to all valid points");
    std::fill(masked_views[0].selection_mask.begin(),masked_views[0].selection_mask.end(),0);
    bool empty_reference_rejected=false;
    try { fs::refine_camera_poses_icp(masked_views); }
    catch (const std::runtime_error&) { empty_reference_rejected=true; }
    require(empty_reference_rejected,"An empty reference mask fell back to all valid points");
    masked_views[0].selection_mask.pop_back();
    bool bad_mask_rejected=false;
    try { fs::refine_camera_poses_icp(masked_views); }
    catch (const std::runtime_error&) { bad_mask_rejected=true; }
    require(bad_mask_rejected,"ICP accepted a mask with a different point count");
    if (gpu) checked(cudaMemcpy(b.data(),buffers[1].ptr,b.size()*sizeof(fs::MeshGPUVertex),cudaMemcpyDeviceToHost));
    require(std::memcmp(b.data(),original.data(),b.size()*sizeof(fs::MeshGPUVertex))==0,"ICP mutated the input point buffer");
    // A reference containing only zero/NaN points must fail on CPU and CUDA.
    const auto reference=views[0];
    views[0].vertices+=views[0].vertex_count-2;
    views[0].vertex_count=2;
    bool rejected=false;
    try { fs::refine_camera_poses_icp(views); } catch (const std::runtime_error&) { rejected=true; }
    require(rejected,"ICP accepted a reference with no valid points");
    views[0]=reference;
    bool cancelled=false, stop=false;
    try {
        fs::refine_camera_poses_icp(views,fs::kDefaultICPTranslationLimitM,[&] { if (stop) throw std::runtime_error("cancelled"); },
            [&](std::size_t,const fs::ICPViewResult&) { stop=true; });
    } catch (const std::runtime_error& e) { cancelled=std::string(e.what())=="cancelled"; }
    require(cancelled,"ICP swallowed cancellation and returned partially refined poses");
}
}
int main() {
    try {
        run(false);
        int devices=0;
        if (cudaGetDeviceCount(&devices)==cudaSuccess && devices>0) run(true);
        else std::cout<<"GPU verification skipped: no CUDA device\n";
        std::cout<<"PASS: anchored multiview ICP, adjustable translation limit, current per-view masks, empty/misaligned masks, pose convention, rejection, immutable points and cancellation\n";
        return 0;
    } catch (const std::exception& e) { std::cerr<<e.what()<<'\n'; return 1; }
}
