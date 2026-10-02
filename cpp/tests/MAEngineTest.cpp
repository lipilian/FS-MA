#include "fs/inference/FS.hpp"
#include "fs/inference/MA-VGGT.hpp"

#include <chrono>
#include <array>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>

namespace {
void require(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}
void checked(cudaError_t error) {
    if (error != cudaSuccess) throw std::runtime_error(cudaGetErrorString(error));
}
std::array<float*, 4> devicePointers(const fs::MA_VGGT& model) {
    return {model.inputDevice(), model.depthsDevice(), model.posesDevice(), model.scaleDevice()};
}
bool deviceAllocation(const float* pointer) {
    cudaPointerAttributes attributes{};
    const auto error = cudaPointerGetAttributes(&attributes, pointer);
    if (error == cudaErrorInvalidValue) { cudaGetLastError(); return false; }
    checked(error);
    return attributes.type == cudaMemoryTypeDevice;
}
template<class Action> void rejects(Action action, const char* expected) {
    try { action(); }
    catch (const std::runtime_error& error) {
        require(std::string(error.what()).find(expected) != std::string::npos, "Unexpected load error");
        return;
    }
    throw std::runtime_error("Invalid model load unexpectedly succeeded");
}
struct TempDirectory {
    std::filesystem::path path = std::filesystem::temp_directory_path() /
        ("fs_ma_engine_test_" + std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
    TempDirectory() { require(std::filesystem::create_directory(path), "Cannot create test directory"); }
    ~TempDirectory() { std::error_code error; std::filesystem::remove_all(path, error); }
};
} // namespace

int main(int argc, char** argv) {
    try {
        require(argc == 3, "Usage: fs_ma_engine_test MA.engine SAM_encoder.engine");
        rejects([] { fs::MA_VGGT model(nullptr); }, "FS-owned CUDA stream");
        TempDirectory temp;
        const auto missing = temp.path / "missing.engine", empty = temp.path / "empty.engine";
        std::ofstream(empty).close();
        auto owner = std::make_shared<FS>();
        const auto shared_stream = owner->stream();
        std::weak_ptr<FS> lifetime = owner;
        auto model = std::make_unique<fs::MA_VGGT>(owner);
        require(!model->isLoaded() && model->stream() == shared_stream, "Incorrect initial model or stream");
        const auto pointers = devicePointers(*model);
        constexpr std::array<std::size_t, 4> elements{5*7*434*518, 5*6*434*518, 5*7, 1};
        for (std::size_t i = 0; i < pointers.size(); ++i) {
            require(pointers[i] && deviceAllocation(pointers[i]), "Constructor did not allocate GPU I/O");
            // Exercise the full V=5 extent with a different byte pattern per tensor.
            checked(cudaMemsetAsync(pointers[i], int(i)+1, elements[i]*sizeof(float), shared_stream));
        }
        const auto checkBuffers = [&] {
            require(devicePointers(*model) == pointers, "Engine loading replaced persistent I/O addresses");
            checked(cudaStreamSynchronize(shared_stream));
            for (std::size_t i = 0; i < pointers.size(); ++i) {
                std::uint32_t first{}, last{};
                checked(cudaMemcpy(&first, pointers[i], sizeof(first), cudaMemcpyDeviceToHost));
                checked(cudaMemcpy(&last, pointers[i]+elements[i]-1, sizeof(last), cudaMemcpyDeviceToHost));
                const auto expected = std::uint32_t(i+1)*0x01010101U;
                require(first == expected && last == expected, "I/O capacity, independence or contents changed");
            }
        };
        checkBuffers();
        rejects([&] { model->loadEngine(missing); }, "cannot open engine");
        rejects([&] { model->loadEngine(empty); }, "empty engine");
        require(!model->isLoaded(), "Failed initial load marked the model ready");
        checkBuffers();

        model->loadEngine(argv[1]); // Includes runtime output-shape checks for every V=2..5.
        require(model->isLoaded() && model->stream() == shared_stream, "MA did not load on the FS stream");
        checkBuffers();
        rejects([&] { model->loadEngine(argv[2]); }, "incompatible tensor");
        rejects([&] { model->loadEngine(missing); }, "cannot open engine");
        require(model->isLoaded(), "Failed reload discarded the working model");
        checkBuffers();
        model->loadEngine(argv[1]);
        require(model->isLoaded() && model->stream() == shared_stream, "Successful reload changed the stream");
        checkBuffers();
        model.reset();
        for (const auto* pointer : pointers)
            require(!deviceAllocation(pointer), "Destructor retained a GPU I/O allocation");
        unsigned int flags{};
        require(cudaStreamGetFlags(shared_stream, &flags) == cudaSuccess, "MA destroyed FS's stream");
        owner->synchronize();

        // Even if the pipeline releases FS first, MA's reference protects its stream.
        model = std::make_unique<fs::MA_VGGT>(owner);
        const auto final_pointers = devicePointers(*model);
        checked(cudaMemsetAsync(model->inputDevice(), 0, elements[0]*sizeof(float), shared_stream));
        owner.reset();
        require(!lifetime.expired() && model->stream() == shared_stream, "MA lost its stream owner");
        require(cudaStreamSynchronize(model->stream()) == cudaSuccess, "Borrowed stream was released early");
        model.reset();
        require(lifetime.expired(), "Stream owner leaked after MA release");
        for (const auto* pointer : final_pointers)
            require(!deviceAllocation(pointer), "GPU I/O survived the shared stream owner");
        std::cout << "PASS: constructor V=5 GPU I/O capacity, stable addresses/data through load and reload, dynamic shapes, load errors, GPU buffer release, shared FS stream and lifetime; no inference\n";
    } catch (const std::exception& error) {
        std::cerr << "FAIL: " << error.what() << '\n';
        return 1;
    }
}
