#include <nvblox/mapper/mapper.h>
#include <nvblox/rays/sphere_tracer.h>
#include <cmath>
#include <iostream>
#include <stdexcept>
#include <vector>

int main() {
    try {
        constexpr int width=960, height=800;
        constexpr float voxel_size=.005f, plane_z=.4f, focal=644.f;
        const nvblox::Camera camera(focal,focal,width*.5f,height*.5f,width,height);
        nvblox::DepthImage depth(height,width,nvblox::MemoryType::kDevice);
        const std::vector<float> input(width*height,plane_z);
        depth.copyFrom(height,width,input.data());
        nvblox::Mapper mapper(voxel_size);
        for (const float x : {-.025f,.025f}) {
            auto pose=nvblox::Transform::Identity();
            pose.translation().x()=x;
            mapper.integrateDepth(depth,pose,camera);
        }
        auto target=nvblox::Transform::Identity();
        target.translation().z()=.05f;
        target.linear()=Eigen::AngleAxisf(.1f,nvblox::Vector3f::UnitY()).toRotationMatrix();
        nvblox::SphereTracer tracer;
        nvblox::DepthImage rendered(nvblox::MemoryType::kDevice);
        tracer.renderImageOnGPU(camera,target,mapper.tsdf_layer(),
            mapper.tsdf_integrator().truncation_distance_vox()*voxel_size,&rendered);
        if (rendered.rows()!=height || rendered.cols()!=width)
            throw std::runtime_error("nvblox rendered the wrong depth resolution");
        std::vector<float> output(width*height);
        rendered.copyTo(output.data());
        int checked=0;
        for (int y=height/4;y<height*3/4;y+=16) for (int x=width/4;x<width*3/4;x+=16) {
            // A world-Z plane viewed from a translated, rotated camera. Check
            // camera Z in metres, including off-axis pixels, rather than range.
            const nvblox::Vector3f ray((x+.5f-width*.5f)/focal,(y+.5f-height*.5f)/focal,1.f);
            const float expected=(plane_z-target.translation().z())/(target.linear()*ray).z();
            const float actual=output[y*width+x];
            if (!std::isfinite(actual) || actual<=0 || std::abs(actual-expected)>2*voxel_size)
                throw std::runtime_error("nvblox synthetic depth failed metric Z / camera pose validation");
            ++checked;
        }
        std::cout<<"PASS: local nvblox library, GPU TSDF fusion and 960x800 metric Z rendering from a new camera pose ("
                 <<checked<<" samples)\n";
    } catch (const std::exception& error) {
        std::cerr<<error.what()<<'\n'; return 1;
    }
}
