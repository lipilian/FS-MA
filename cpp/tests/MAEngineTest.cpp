#include "fs/inference/FS.hpp"
#include "fs/inference/MA-VGGT.hpp"

#include <chrono>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>

namespace {
void require(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
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
        rejects([&] { model->loadEngine(missing); }, "cannot open engine");
        rejects([&] { model->loadEngine(empty); }, "empty engine");
        require(!model->isLoaded(), "Failed initial load marked the model ready");

        model->loadEngine(argv[1]); // Includes runtime output-shape checks for every V=2..5.
        require(model->isLoaded() && model->stream() == shared_stream, "MA did not load on the FS stream");
        rejects([&] { model->loadEngine(argv[2]); }, "incompatible tensor");
        rejects([&] { model->loadEngine(missing); }, "cannot open engine");
        require(model->isLoaded(), "Failed reload discarded the working model");
        model->loadEngine(argv[1]);
        require(model->isLoaded() && model->stream() == shared_stream, "Successful reload changed the stream");
        model.reset();
        unsigned int flags{};
        require(cudaStreamGetFlags(shared_stream, &flags) == cudaSuccess, "MA destroyed FS's stream");
        owner->synchronize();

        // Even if the pipeline releases FS first, MA's reference protects its stream.
        model = std::make_unique<fs::MA_VGGT>(owner);
        owner.reset();
        require(!lifetime.expired() && model->stream() == shared_stream, "MA lost its stream owner");
        require(cudaStreamSynchronize(model->stream()) == cudaSuccess, "Borrowed stream was released early");
        model.reset();
        require(lifetime.expired(), "Stream owner leaked after MA release");
        std::cout << "PASS: MA load, dynamic shapes, wrong-engine/missing/empty errors, reload, shared FS stream and lifetime; no inference\n";
    } catch (const std::exception& error) {
        std::cerr << "FAIL: " << error.what() << '\n';
        return 1;
    }
}
