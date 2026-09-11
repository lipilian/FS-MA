#include "fs/inference/SamSegmenter.hpp"
#include <NvInfer.h>
#include <cuda_runtime_api.h>
#include <opencv2/imgproc.hpp>
#include <array>
#include <algorithm>
#include <cmath>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>

namespace fs {
namespace {
class Logger final : public nvinfer1::ILogger {
    void log(Severity severity, const char* message) noexcept override {
        if (severity <= Severity::kWARNING) std::cerr << "[SAM TensorRT] " << message << '\n';
    }
} logger;
void cudaCheck(cudaError_t error, const char* action) {
    if (error != cudaSuccess) throw std::runtime_error(std::string(action) + ": " + cudaGetErrorString(error));
}
void require(bool condition, const std::string& message) {
    if (!condition) throw std::runtime_error("SAM: " + message);
}
struct DeviceFree { void operator()(float* p) const noexcept { if (p) cudaFree(p); } };
struct HostFree { void operator()(float* p) const noexcept { if (p) cudaFreeHost(p); } };
struct Buffer {
    std::unique_ptr<float, DeviceFree> device;
    std::unique_ptr<float, HostFree> host;
    size_t count{};
    void allocate(size_t n, bool staging = false) {
        count = n; void* p = nullptr;
        cudaCheck(cudaMalloc(&p, n*sizeof(float)), "allocate SAM GPU buffer"); device.reset(static_cast<float*>(p));
        if (staging) { p = nullptr; cudaCheck(cudaMallocHost(&p, n*sizeof(float)), "allocate SAM pinned buffer"); host.reset(static_cast<float*>(p)); }
    }
    void upload(cudaStream_t stream, size_t n) {
        cudaCheck(cudaMemcpyAsync(device.get(), host.get(), n*sizeof(float), cudaMemcpyHostToDevice, stream), "upload SAM input");
    }
    void download(cudaStream_t stream) {
        cudaCheck(cudaMemcpyAsync(host.get(), device.get(), count*sizeof(float), cudaMemcpyDeviceToHost, stream), "download SAM output");
    }
};
nvinfer1::Dims dims(std::initializer_list<int64_t> values) {
    nvinfer1::Dims d{}; d.nbDims = values.size(); std::copy(values.begin(), values.end(), d.d); return d;
}
bool equal(nvinfer1::Dims a, nvinfer1::Dims b) {
    return a.nbDims == b.nbDims && std::equal(a.d, a.d+a.nbDims, b.d);
}
struct Model {
    std::unique_ptr<nvinfer1::ICudaEngine> engine;
    std::unique_ptr<nvinfer1::IExecutionContext> context;
    void load(nvinfer1::IRuntime& runtime, const std::filesystem::path& path) {
        std::ifstream file(path, std::ios::binary | std::ios::ate);
        require(bool(file), "cannot open engine: " + path.string());
        const auto size = file.tellg(); require(size > 0, "empty engine: " + path.string());
        std::vector<char> bytes(static_cast<size_t>(size)); file.seekg(0);
        require(bool(file.read(bytes.data(), bytes.size())), "cannot read engine: " + path.string());
        engine.reset(runtime.deserializeCudaEngine(bytes.data(), bytes.size()));
        require(bool(engine), "cannot deserialize engine: " + path.string());
        context.reset(engine->createExecutionContext()); // Allocate TensorRT context memory during splash.
        require(bool(context), "cannot create context: " + path.string());
    }
    void tensor(const char* name, nvinfer1::Dims shape, nvinfer1::TensorIOMode mode) {
        require(engine->getTensorIOMode(name) == mode && engine->getTensorDataType(name) == nvinfer1::DataType::kFLOAT &&
                engine->getTensorLocation(name) == nvinfer1::TensorLocation::kDEVICE &&
                engine->getTensorFormat(name) == nvinfer1::TensorFormat::kLINEAR && equal(engine->getTensorShape(name),shape),
                std::string("incompatible tensor: ") + name);
    }
    void bind(const char* name, Buffer& buffer) { require(context->setTensorAddress(name,buffer.device.get()),std::string("cannot bind ")+name); }
    void run(cudaStream_t stream) { require(context->enqueueV3(stream), "inference enqueue failed"); }
};
}
struct SamSegmenter::Impl {
    cudaStream_t stream{};
    std::unique_ptr<nvinfer1::IRuntime> runtime;
    Model encoder, decoder;
    Buffer image, feat0, feat1, embed, coords, labels, mask_input, has_mask, masks, scores;
    cv::Mat resized;
    cv::Size original;
    bool loaded{false}, encoded{false};
    float best_score{};
    Impl() { cudaCheck(cudaStreamCreateWithFlags(&stream,cudaStreamNonBlocking),"create SAM stream"); }
    ~Impl() { if (stream) { cudaStreamSynchronize(stream); cudaStreamDestroy(stream); } }
    void sync() { cudaCheck(cudaStreamSynchronize(stream),"complete SAM work"); }
    void shapes(int n) {
        require(decoder.context->setInputShape("point_coords",dims({1,n,2})) &&
                decoder.context->setInputShape("point_labels",dims({1,n})) &&
                decoder.context->setInputShape("mask_input",dims({1,1,256,256})) &&
                decoder.context->setInputShape("has_mask_input",dims({1})), "cannot set decoder shapes");
        require(decoder.context->inferShapes(0,nullptr)==0, "decoder shapes are incomplete");
        require(equal(decoder.context->getTensorShape("masks"),dims({1,3,256,256})) &&
                equal(decoder.context->getTensorShape("iou_predictions"),dims({1,3})), "unexpected decoder output shapes");
    }
};
SamSegmenter::SamSegmenter() = default;
SamSegmenter::~SamSegmenter() = default;
bool SamSegmenter::isLoaded() const noexcept { return impl_ && impl_->loaded; }
bool SamSegmenter::hasImage() const noexcept { return isLoaded() && impl_->encoded; }
void SamSegmenter::clearImage() noexcept { if (impl_) { impl_->encoded=false; impl_->original={}; } }
float SamSegmenter::score() const noexcept { return impl_ ? impl_->best_score : 0.0F; }
void SamSegmenter::loadEngines(const std::filesystem::path& encoder_path, const std::filesystem::path& decoder_path) {
    auto next = std::make_unique<Impl>(); auto& s = *next;
    auto* active_logger = getLogger();
    s.runtime.reset(nvinfer1::createInferRuntime(active_logger ? *active_logger : logger)); require(bool(s.runtime), "cannot create runtime");
    s.encoder.load(*s.runtime,encoder_path); s.decoder.load(*s.runtime,decoder_path);
    using Mode = nvinfer1::TensorIOMode;
    require(s.encoder.engine->getNbIOTensors()==4 && s.decoder.engine->getNbIOTensors()==9, "unexpected engine interface");
    s.encoder.tensor("image",dims({1,3,1024,1024}),Mode::kINPUT);
    for (auto* model : {&s.encoder,&s.decoder}) {
        const auto mode = model==&s.encoder ? Mode::kOUTPUT : Mode::kINPUT;
        model->tensor("high_res_feats_0",dims({1,32,256,256}),mode);
        model->tensor("high_res_feats_1",dims({1,64,128,128}),mode);
        model->tensor("image_embed",dims({1,256,64,64}),mode);
    }
    s.decoder.tensor("point_coords",dims({1,-1,2}),Mode::kINPUT);
    s.decoder.tensor("point_labels",dims({1,-1}),Mode::kINPUT);
    s.decoder.tensor("mask_input",dims({1,1,256,256}),Mode::kINPUT);
    s.decoder.tensor("has_mask_input",dims({1}),Mode::kINPUT);
    s.decoder.tensor("masks",dims({1,3,256,256}),Mode::kOUTPUT);
    s.decoder.tensor("iou_predictions",dims({1,3}),Mode::kOUTPUT);
    for (const auto& item : std::array<std::pair<const char*,nvinfer1::Dims>,2>{{
            {"point_coords",dims({1,kMaxPrompts,2})},{"point_labels",dims({1,kMaxPrompts})}}}) {
        const auto low=s.decoder.engine->getProfileShape(item.first,0,nvinfer1::OptProfileSelector::kMIN);
        const auto high=s.decoder.engine->getProfileShape(item.first,0,nvinfer1::OptProfileSelector::kMAX);
        require(low.nbDims==item.second.nbDims && high.nbDims==item.second.nbDims,"invalid decoder profile");
        for(int i=0;i<low.nbDims;++i) {
            const int64_t minimum = (i==1 && (std::string(item.first)=="point_coords" || std::string(item.first)=="point_labels")) ? 1 : item.second.d[i];
            require(low.d[i]<=minimum && high.d[i]>=item.second.d[i],"decoder profile must support one target and 1–64 prompt points");
        }
    }
    s.image.allocate(3*1024*1024,true); s.feat0.allocate(32*256*256); s.feat1.allocate(64*128*128); s.embed.allocate(256*64*64);
    s.coords.allocate(kMaxPrompts*2,true); s.labels.allocate(kMaxPrompts,true);
    s.mask_input.allocate(256*256); s.has_mask.allocate(1); s.masks.allocate(3*256*256,true); s.scores.allocate(3,true);
    s.resized.create(1024,1024,CV_8UC3);
    s.encoder.bind("image",s.image);
    for(auto* model : {&s.encoder,&s.decoder}) {
        model->bind("high_res_feats_0",s.feat0); model->bind("high_res_feats_1",s.feat1); model->bind("image_embed",s.embed);
    }
    s.decoder.bind("point_coords",s.coords); s.decoder.bind("point_labels",s.labels);
    s.decoder.bind("mask_input",s.mask_input); s.decoder.bind("has_mask_input",s.has_mask);
    s.decoder.bind("masks",s.masks); s.decoder.bind("iou_predictions",s.scores);
    // AnyLabeling uses no previous-mask feedback: all edits decode the complete prompt list.
    cudaCheck(cudaMemsetAsync(s.mask_input.device.get(),0,s.mask_input.count*sizeof(float),s.stream),"clear SAM mask input");
    cudaCheck(cudaMemsetAsync(s.has_mask.device.get(),0,sizeof(float),s.stream),"clear SAM mask flag");
    s.shapes(kMaxPrompts); s.sync(); s.loaded=true; impl_=std::move(next);
}
void SamSegmenter::setImage(const cv::Mat& rgb) {
    require(isLoaded(),"load engines before encoding"); auto& s=*impl_; s.encoded=false;
    require(!rgb.empty() && rgb.type()==CV_8UC3,"image must be RGB uint8"); s.sync();
    cv::resize(rgb,s.resized,{1024,1024},0,0,cv::INTER_LINEAR);
    constexpr std::array<double,3> mean{0.485,0.456,0.406}, stddev{0.229,0.224,0.225};
    for(int y=0;y<1024;++y) {
        const auto* row=s.resized.ptr<cv::Vec3b>(y);
        for(int x=0;x<1024;++x) for(int c=0;c<3;++c)
            s.image.host.get()[c*1024*1024+y*1024+x]=static_cast<float>((row[x][c]/255.0-mean[c])/stddev[c]);
    }
    s.image.upload(s.stream,s.image.count); s.encoder.run(s.stream); s.sync(); s.original=rgb.size(); s.encoded=true;
}
cv::Mat SamSegmenter::predict(const std::vector<SamPrompt>& prompts) {
    require(hasImage(),"encode an image before predicting"); auto& s=*impl_;
    require(!prompts.empty() && prompts.size()<=kMaxPrompts,"use 1–64 prompt points (a box uses two)");
    s.sync();
    for(size_t i=0;i<prompts.size();++i) {
        const auto& p=prompts[i];
        require(p.label>=0 && p.label<=3 && std::isfinite(p.point.x) && std::isfinite(p.point.y) &&
                p.point.x>=0 && p.point.y>=0 && p.point.x<s.original.width && p.point.y<s.original.height,"invalid prompt");
        s.coords.host.get()[i*2]=p.point.x/static_cast<float>(s.original.width)*1024.0F;
        s.coords.host.get()[i*2+1]=p.point.y/static_cast<float>(s.original.height)*1024.0F;
        s.labels.host.get()[i]=p.label;
    }
    s.shapes(prompts.size()); s.coords.upload(s.stream,prompts.size()*2); s.labels.upload(s.stream,prompts.size());
    s.decoder.run(s.stream); s.masks.download(s.stream); s.scores.download(s.stream); s.sync();
    for(int i=0;i<3;++i) require(std::isfinite(s.scores.host.get()[i]),"non-finite mask score");
    const auto* scores=s.scores.host.get(); const int best=std::max_element(scores,scores+3)-scores; s.best_score=scores[best];
    cv::Mat logits(256,256,CV_32F,s.masks.host.get()+best*256*256), resized;
    require(cv::checkRange(logits),"non-finite mask output");
    cv::resize(logits,resized,s.original,0,0,cv::INTER_LINEAR);
    return resized > 0.0F;
}
}
