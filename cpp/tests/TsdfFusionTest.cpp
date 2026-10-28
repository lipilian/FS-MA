#include "fs/geometry/TsdfFusion.hpp"
#include <cmath>
#include <iostream>
#include <limits>
#include <stdexcept>

namespace {
void require(bool ok,const char* message) { if (!ok) throw std::runtime_error(message); }
}
int main() {
    try {
        fs::TsdfOptions options;
        require(options.voxel_size_mm==1 && options.truncation_mm()==8 && options.block_resolution==8 && options.block_count==30000 &&
                options.mesh_weight_threshold==0 && options.point_weight_threshold==0,"Incorrect TSDF defaults");
        options.block_count=1000;
        for (double mm : {.1,10.}) { auto o=options; o.voxel_size_mm=mm; o.validate(); }
        for (double mm : {0.,.09,10.01,std::numeric_limits<double>::quiet_NaN()}) {
            auto o=options; o.voxel_size_mm=mm; bool rejected=false;
            try { o.validate(); } catch (const std::invalid_argument&) { rejected=true; }
            require(rejected,"Invalid voxel size was accepted");
        }
        fs::TsdfView view;
        view.rgb=cv::Mat(80,96,CV_8UC3,cv::Scalar(255,32,0));
        view.depth=cv::Mat(80,96,CV_32FC1,cv::Scalar(.4f));
        view.intrinsics={200,0,48,0,200,40,0,0,1};
        view.camera_to_world=cv::Matx44d::eye();
        view.camera_to_world(0,3)=.1; view.camera_to_world(2,3)=-1.;
        view.mask=cv::Mat::zeros(160,192,CV_8UC1);
        view.mask(cv::Rect(0,0,96,160)).setTo(255);
        view.depth.at<float>(20,20)=std::numeric_limits<float>::quiet_NaN();
        view.depth.at<float>(20,21)=std::numeric_limits<float>::infinity();
        view.depth.at<float>(20,22)=-1;
        int progress=0;
        const auto masked=fs::fuse_tsdf({view,view},options,[&](int p,const char*) { require(p>=progress,"Fusion progress regressed"); progress=p; });
        require(masked.integrated_views==2 && masked.active_blocks>0 && masked.mesh.stats.triangle_count>100 &&
                masked.points.stats.point_count>100,"CUDA TSDF did not extract plane geometry");
        for (const auto* geometry:{&masked.mesh,&masked.points}) {
            require(std::abs(geometry->stats.low[2]+.6f)<.003f && std::abs(geometry->stats.high[2]+.6f)<.003f,
                    "TSDF depth scale or camera-to-world inversion is incorrect");
            require(geometry->stats.high[0]<.105 && geometry->stats.low[0]>-.005,"Full-resolution mask or pose translation was ignored");
            size_t bright=0;
            for (const auto& v:geometry->vertices) {
                require(std::abs(v.g-v.r*32.f/255)<.01 && v.b<.01,"TSDF RGB channels or normalization are wrong");
                if (v.r>.95) ++bright;
            }
            // Raw extraction retains edge samples whose interpolated color can
            // include an unobserved neighbour. Interior surface must keep RGB.
            require(bright>geometry->vertices.size()*.95,"TSDF surface colors are not normalized RGB");

        }
        require(std::isnan(view.depth.at<float>(20,20)) && view.depth.at<float>(20,22)==-1 &&
                view.depth.at<float>(30,70)==.4f,"Fusion modified retained FS depth");
        view.mask.setTo(0); // Undrawn masks retain full depth, as in the reference tool.
        options.mesh_weight_threshold=100;
        options.truncation_voxels=4;
        const auto point_only=fs::fuse_tsdf({view,view},options);
        require(point_only.mesh.vertices.empty() && point_only.points.stats.point_count>masked.points.stats.point_count &&
                point_only.points.stats.high[0]>.18,"Independent extraction weights or undrawn-mask fallback failed");
        options.mesh_weight_threshold=0; options.point_weight_threshold=100;
        options.block_resolution=16;
        const auto mesh_only=fs::fuse_tsdf({view,view},options);
        require(mesh_only.points.vertices.empty() && mesh_only.mesh.stats.triangle_count>100,
                "Point weight threshold affected mesh extraction or block resolution failed");
        std::cout<<"PASS: CUDA TSDF metric scale, pose inversion, masks, RGB, truncation, block resolution, independent extraction weights and immutable depth\n";
        return 0;
    } catch (const std::exception& error) { std::cerr<<error.what()<<'\n'; return 1; }
}
