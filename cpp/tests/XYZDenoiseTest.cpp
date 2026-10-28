#include "fs/inference/PostProcessing.hpp"

#include <opencv2/core.hpp>
#include <algorithm>
#include <cmath>
#include <cstring>
#include <iostream>
#include <limits>
#include <random>
#include <stdexcept>
#include <vector>

namespace {
constexpr float distance_limit=0.003F;
constexpr double pi=3.14159265358979323846;
void require(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}
void checked(cudaError_t result) {
    if (result!=cudaSuccess) throw std::runtime_error(cudaGetErrorString(result));
}
struct Device {
    cudaStream_t stream{};
    float *input{}, *output{};
    unsigned char* mask{};
    explicit Device(std::size_t count) {
        checked(cudaStreamCreateWithFlags(&stream,cudaStreamNonBlocking));
        checked(cudaMalloc(reinterpret_cast<void**>(&input),count*3*sizeof(float)));
        checked(cudaMalloc(reinterpret_cast<void**>(&output),count*3*sizeof(float)));
        checked(cudaMalloc(reinterpret_cast<void**>(&mask),count));
    }
    ~Device() { cudaFree(input); cudaFree(output); cudaFree(mask); cudaStreamDestroy(stream); }
    std::vector<cv::Vec3f> run(const std::vector<cv::Vec3f>& points, int width, int height,
                             const std::vector<unsigned char>& selection={}) {
        const auto bytes=points.size()*sizeof(points[0]);
        checked(cudaMemcpyAsync(input,points.data(),bytes,cudaMemcpyHostToDevice,stream));
        if (!selection.empty()) checked(cudaMemcpyAsync(mask,selection.data(),selection.size(),cudaMemcpyHostToDevice,stream));
        checked(fs::postprocessing::denoise_xyz_map(input,output,selection.empty()?nullptr:mask,
                                                   width,height,distance_limit,7,60.0F,stream));
        std::vector<cv::Vec3f> result(points.size()), original(points.size());
        checked(cudaMemcpyAsync(result.data(),output,bytes,cudaMemcpyDeviceToHost,stream));
        checked(cudaMemcpyAsync(original.data(),input,bytes,cudaMemcpyDeviceToHost,stream));
        checked(cudaStreamSynchronize(stream));
        require(std::memcmp(original.data(),points.data(),bytes)==0,"Denoising changed the input XYZ");
        return result;
    }
};
bool valid(const cv::Vec3f& p) {
    return std::isfinite(p[0]) && std::isfinite(p[1]) && std::isfinite(p[2]) && p[2]>0;
}
bool retained(const std::vector<cv::Vec3f>& points, int i) { return points[i][2]>0; }

// Independent double-precision CPU oracle using OpenCV's eigensolver. All
// neighbours come from the original input, including ones that will be removed.
std::vector<cv::Vec3f> reference(const std::vector<cv::Vec3f>& points, int width, int height,
                               const std::vector<unsigned char>& mask) {
    auto result=points;
    for (int v=0;v<height;++v) for (int u=0;u<width;++u) {
        const int i=v*width+u;
        if (!mask.empty() && !mask[i]) continue;
        const cv::Vec3d center=points[i];
        std::vector<cv::Vec3d> neighbors;
        if (valid(points[i])) {
            for (int y=std::max(0,v-2);y<std::min(height,v+3);++y)
                for (int x=std::max(0,u-2);x<std::min(width,u+3);++x) {
                    const int j=y*width+x;
                    if (j==i || (!mask.empty() && !mask[j]) || !valid(points[j])) continue;
                    const cv::Vec3d delta=cv::Vec3d(points[j])-center;
                    if (cv::norm(delta)<double(distance_limit)) neighbors.push_back(delta);
                }
        }
        bool keep=neighbors.size()>=7;
        if (keep) {
            cv::Vec3d mean{};
            for (const auto& n : neighbors) mean+=n;
            mean/=double(neighbors.size());
            cv::Matx33d covariance=cv::Matx33d::zeros();
            for (const auto& n : neighbors) { const cv::Vec3d delta=n-mean; covariance+=delta*delta.t(); }
            cv::Vec3d values;
            cv::Matx33d vectors;
            require(cv::eigen(covariance,values,vectors),"CPU plane fit failed");
            const cv::Vec3d normal(vectors(2,0),vectors(2,1),vectors(2,2));
            keep=values[1]>1e-6*values[0] && values[1]-values[2]>1e-6*values[0] &&
                 std::abs(normal.dot(center))/cv::norm(center)>0.5;
        }
        if (!keep) result[i]={0,0,0};
    }
    return result;
}
void compare(const std::vector<cv::Vec3f>& actual, const std::vector<cv::Vec3f>& expected) {
    require(actual.size()==expected.size(),"Wrong output size");
    for (std::size_t i=0;i<actual.size();++i) {
        if (std::memcmp(&actual[i],&expected[i],sizeof(actual[i]))!=0)
            throw std::runtime_error("GPU/CPU result mismatch at pixel "+std::to_string(i));
    }
}

std::vector<cv::Vec3f> plane(int width, int height, double tilt, double azimuth=0,
                           cv::Vec3d center={0,0,.4}, double spacing=.0006) {
    tilt*=pi/180; azimuth*=pi/180;
    const cv::Vec3d a(std::cos(tilt)*std::cos(azimuth),std::cos(tilt)*std::sin(azimuth),-std::sin(tilt));
    const cv::Vec3d b(-std::sin(azimuth),std::cos(azimuth),0);
    std::vector<cv::Vec3f> points(width*height);
    for (int y=0;y<height;++y) for (int x=0;x<width;++x)
        points[y*width+x]=center+spacing*((x-width/2)*a+(y-height/2)*b);
    return points;
}

void checkCountsAndDistance() {
    constexpr int w=7, h=7, center=3*w+3;
    Device device(w*h);
    auto full=plane(w,h,0);
    std::vector<cv::Vec3f> points(w*h,cv::Vec3f(0,0,0));
    points[center]=full[center];
    // Seven neighbours, four in the outer ring of the 5x5 window.
    const std::vector<int> neighbors{center-2*w-2,center-2*w+2,center+2*w-2,center+2*w+2,
                                     center-1,center+1,center-w};
    for (const auto i : neighbors) points[i]=full[i];
    require(retained(device.run(points,w,h),center),"5x5 outer-ring neighbours must count; exactly seven should pass");
    auto six=points; six[neighbors.back()]={0,0,0};
    require(!retained(device.run(six,w,h),center),"Six neighbours plus the centre must fail");
    points[neighbors.back()]={distance_limit,0,.4F};
    require(!retained(device.run(points,w,h),center),"A neighbour at exactly 3 mm must not count");
    points[neighbors.back()][0]=std::nextafter(distance_limit,0.0F);
    require(retained(device.run(points,w,h),center),"A neighbour just inside 3 mm must count");
    points[neighbors.back()][0]=std::nextafter(distance_limit,1.0F);
    require(!retained(device.run(points,w,h),center),"A neighbour just outside 3 mm must not count");
    points[neighbors.back()]={0,0,.404F};
    require(!retained(device.run(points,w,h),center),"Distance must include depth, not just XY");
    for (const cv::Vec3f invalid : {cv::Vec3f(0,0,0),cv::Vec3f(0,0,-1),
             cv::Vec3f(std::numeric_limits<float>::quiet_NaN(),0,.4F),
             cv::Vec3f(0,std::numeric_limits<float>::infinity(),.4F)}) {
        points[neighbors.back()]=invalid;
        require(!retained(device.run(points,w,h),center),"Invalid neighbours must not count");
        auto invalid_center=full; invalid_center[center]=invalid;
        const auto result=device.run(invalid_center,w,h);
        require(result[center]==cv::Vec3f(0,0,0),"Invalid centre must be zeroed");
    }
    std::cout<<"PASS: 5x5 support, seven/six neighbours, strict 3 mm, invalid points and unchanged input\n";
}

void checkNormals() {
    constexpr int w=9,h=9,center=4*w+4;
    Device device(w*h);
    for (const double azimuth : {0.0,37.0,90.0,143.0,235.0})
        for (const double tilt : {0.0,40.0,59.9,60.1,75.0,90.0}) {
            auto points=plane(w,h,tilt,azimuth);
            const auto result=device.run(points,w,h);
            require(retained(result,center)==(tilt<60),"Incorrect plane-normal viewing-angle rejection");
            compare(result,reference(points,w,h,{}));
        }
    const double off_axis_x=.4*std::tan(35*pi/180);
    auto points=plane(w,h,75,0,{off_axis_x,0,.4});
    require(retained(device.run(points,w,h),center),"Off-axis view ray should accept a 40 degree angle to a 75 degree normal");
    points=plane(w,h,35,0,{-off_axis_x,0,.4});
    require(!retained(device.run(points,w,h),center),"Off-axis view ray should reject a 70 degree angle to a 35 degree normal");
    // Exactly representable coordinates: normal (1,1,0), view (1,0,1).
    // Their angle is exactly 60 degrees, so strict inequality must reject it.
    for (int y=0;y<h;++y) for (int x=0;x<w;++x) {
        const float a=(x-w/2)/2048.0F, b=(y-h/2)/2048.0F;
        points[y*w+x]={.5F+a,-a,.5F+b};
    }
    require(!retained(device.run(points,w,h),center),"A normal at exactly 60 degrees must be rejected");
    points=plane(w,h,0);
    for (auto& p : points) p[1]=0;
    require(!retained(device.run(points,w,h),center),"Collinear neighbours have no plane normal");
    std::fill(points.begin(),points.end(),cv::Vec3f(0,0,.4F));
    require(!retained(device.run(points,w,h),center),"Coincident neighbours have no plane normal");
    std::fill(points.begin(),points.end(),cv::Vec3f(0,0,0));
    points[center]={0,0,.5F};
    // Cube corners have equal covariance eigenvalues: no preferred plane.
    for (int i=0;i<8;++i) points[center-2*w-2+(i/4)*w+i%4]=
        {((i&1)?1:-1)/1024.0F,((i&2)?1:-1)/1024.0F,.5F+((i&4)?1:-1)/1024.0F};
    require(!retained(device.run(points,w,h),center),"Isotropic neighbours must not produce an arbitrary normal");
    // Distant outliers must affect neither support nor the fitted normal.
    points=plane(w,h,0);
    for (int y=2;y<=6;++y) for (int x=2;x<=6;++x)
        if (x==2 || x==6 || y==2 || y==6) points[y*w+x][2]=.42F;
    require(retained(device.run(points,w,h),center),"Out-of-distance neighbours contaminated the normal fit");
    std::cout<<"PASS: normal angles around 60 degrees, arbitrary normal directions, local viewing ray and degenerate planes\n";
}

void checkMasksAndReference() {
    // Partial CUDA blocks, edges and corners, at 40 cm with a 644.26 px focal length.
    constexpr int w=17,h=19,center=9*w+8;
    Device device(w*h);
    auto points=plane(w,h,0,0,{0,0,.4},.4/644.259460449219);
    const auto full=device.run(points,w,h);
    compare(full,reference(points,w,h,{}));
    require(retained(full,0) && retained(full,w*h-1),"Flat 40 cm surface should survive, including image corners");
    std::vector<unsigned char> mask(w*h,0);
    // The centre is on a selection edge with exactly six selected neighbours.
    const std::vector<int> selected{center,center-1,center+1,center-w,center+w,center-w-1,center+w+1};
    for (const int i : selected) mask[i]=255;
    auto result=device.run(points,w,h,mask);
    require(!retained(result,center),"Selection edges must not relax the seven-neighbour requirement");
    compare(result,reference(points,w,h,mask));
    mask[center-w+1]=255;
    result=device.run(points,w,h,mask);
    require(retained(result,center),"Exactly seven selected neighbours should pass");
    compare(result,reference(points,w,h,mask));
    for (int i=0;i<w*h;++i) if (!mask[i]) require(result[i]==points[i],"Unselected XYZ must remain unchanged");
    std::fill(mask.begin(),mask.end(),0);
    compare(device.run(points,w,h,mask),points);
    std::fill(mask.begin(),mask.end(),255);
    compare(device.run(points,w,h,mask),full);
    // Non-planar noisy surfaces, holes, outliers and irregular selections; the
    // independent CPU eigensolver also detects any cascading removal in a pass.
    std::mt19937 random(42);
    std::uniform_real_distribution<float> jitter(-.00008F,.00008F);
    for (int trial=0;trial<40;++trial) {
        points=plane(w,h,(trial%10)*8,trial*17,{.04,-.02,.4});
        for (int i=0;i<w*h;++i) {
            points[i][2]+=jitter(random);
            if (random()%17==0) points[i]={0,0,0};
            if (random()%23==0) points[i][2]+=.01F;
            mask[i]=random()%5 ? 255 : 0;
        }
        compare(device.run(points,w,h,mask),reference(points,w,h,mask));
    }
    // A two-row strip cannot supply seven neighbours at its image corners.
    points=plane(w,2,0);
    result=device.run(points,w,2);
    require(!retained(result,0) && !retained(result,2*w-1),"Image edges must not relax the support requirement");
    compare(result,reference(points,w,2,{}));
    std::cout<<"PASS: masks, image edges, partial blocks, 40 cm spacing and 40 noisy CPU-reference cases\n";
}

void checkValidationAndFullImage() {
    constexpr int w=960,h=800;
    Device device(w*h);
    auto invoke=[&](float distance, int count, float angle) {
        return fs::postprocessing::denoise_xyz_map(device.input,device.output,nullptr,w,h,distance,count,angle,device.stream);
    };
    for (float distance : {0.0F,-1.0F,std::numeric_limits<float>::infinity(),std::numeric_limits<float>::quiet_NaN()})
        require(invoke(distance,7,60)==cudaErrorInvalidValue,"Invalid distance accepted");
    for (int count : {0,2,25}) require(invoke(distance_limit,count,60)==cudaErrorInvalidValue,"Invalid neighbour count accepted");
    for (float angle : {0.0F,90.0F,-1.0F,std::numeric_limits<float>::quiet_NaN()})
        require(invoke(distance_limit,7,angle)==cudaErrorInvalidValue,"Invalid angle accepted");
    require(fs::postprocessing::denoise_xyz_map(device.input,device.input,nullptr,w,h,distance_limit,7,60,device.stream)==cudaErrorInvalidValue,
            "In-place filtering must be rejected");
    const auto points=plane(w,h,0,0,{0,0,.4},.4/644.259460449219);
    compare(device.run(points,w,h),points);
    cudaEvent_t start{},end{};
    checked(cudaEventCreate(&start)); checked(cudaEventCreate(&end));
    checked(cudaEventRecord(start,device.stream));
    for (int i=0;i<10;++i) checked(invoke(distance_limit,7,60));
    checked(cudaEventRecord(end,device.stream)); checked(cudaEventSynchronize(end));
    float ms=0; checked(cudaEventElapsedTime(&ms,start,end));
    cudaEventDestroy(start); cudaEventDestroy(end);
    std::cout<<"PASS: invalid parameters and 960x800 surface; GPU filter average "<<ms/10<<" ms (10 passes)\n";
}
}

int main() {
    try {
        checkCountsAndDistance();
        checkNormals();
        checkMasksAndReference();
        checkValidationAndFullImage();
        return 0;
    } catch (const std::exception& error) {
        std::cerr<<"XYZ denoise test failed: "<<error.what()<<'\n';
        return 1;
    }
}
