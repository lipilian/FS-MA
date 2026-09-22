#include <cmath>
#include <exception>
#include <filesystem>
#include <iomanip>
#include <sstream>
#include <stdexcept>
#include <string>

#include <opencv2/imgcodecs.hpp>

#include "fs/geometry/MeshBuilderGPU.hpp"
#include "fs/inference/FS.hpp"
#include "fs/core/Logger.hpp"
#include "fs/stereo/StereoFrame.hpp"

namespace {

void print_usage(const char* executable) {
    Logger::error(std::string("Usage: ") + executable +
                  " <capture-directory> [--measure]\n\nThe directory must contain left.png, right.png, calibration.json, and masks/left_mask.png.\nSaves raw float32 disparity.tiff in the capture directory and prints the selected GPU mesh area.");
}

}  // namespace

int main(int argc, char* argv[]) {
    const bool measure = argc == 3 && std::string(argv[2]) == "--measure";
    if ((argc != 2 && argc != 3) || (argc == 3 && !measure)) {
        print_usage(argv[0]);
        return 2;
    }

    const std::filesystem::path capture_directory = argv[1];
    const auto left_path = capture_directory / "left.png";
    const auto right_path = capture_directory / "right.png";
    const auto calibration_path = capture_directory / "calibration.json";
    const auto mask_path = capture_directory / "masks" / "left_mask.png";
    const auto disparity_path = capture_directory / "disparity.tiff";

    try {
        FS fs;
        fs.loadEngine("onnx/foundationstereo_800x960_gwc_plugin_cu129.engine");

        Logger::log("Loading stereo capture.");
        StereoFrame frame(left_path, right_path, calibration_path);

        Logger::log("Rectifying RGB stereo pair.");
        frame.rectify();

        const cv::Mat raw_mask = cv::imread(mask_path.string(), cv::IMREAD_GRAYSCALE);
        if (raw_mask.empty()) {
            throw std::runtime_error("Unable to read selection mask: " + mask_path.string());
        }
        const cv::Mat rectified_mask = frame.rectify_left_mask(raw_mask);
        if (cv::countNonZero(rectified_mask) == 0) {
            throw std::runtime_error("Selection mask is empty after rectification");
        }

        fs.set_model_camera_parameters(
            frame.rectified_camera_parameters(), frame.rectified_left().size());
        fs.prepare_stereo_images(frame.rectified_left(), frame.rectified_right());
        fs.set_selection_mask(rectified_mask);
        if (measure) {
            fs.inference_time_measure();
        } else {
            fs.inference();
        }

        const cv::Mat disparity = fs.download_disparity();
        if (!cv::imwrite(disparity_path.string(), disparity)) {
            throw std::runtime_error("Unable to save disparity TIFF: " + disparity_path.string());
        }
        Logger::log("Saved raw float32 disparity (800 x 960, pixels): " + disparity_path.string());

        size_t finite_count = 0, positive_count = 0;
        for (int row = 0; row < disparity.rows; ++row) {
            const float* values = disparity.ptr<float>(row);
            for (int col = 0; col < disparity.cols; ++col) {
                if (std::isfinite(values[col])) {
                    ++finite_count;
                    if (values[col] > 0.0F) ++positive_count;
                }
            }
        }
        Logger::log("Disparity: " + std::to_string(finite_count) + "/" +
                    std::to_string(disparity.total()) + " finite values, " +
                    std::to_string(positive_count) + " positive finite values.");
        if (positive_count == 0) {
            throw std::runtime_error("Area unavailable: engine produced no finite positive disparity; "
                                     "raw output is preserved in " + disparity_path.string());
        }

        // Match PipelineWorker::reconstruct and the Qt default filter settings.
        Logger::log("Computing XYZ map (0 < depth <= 1 m).");
        fs.compute_xyz_map(0.0F, 1.0F);
        fs.denoise_xyz_map(0.01F, 3, 2);
        Logger::log("Selected-region 3 x 3 denoising: 0.01 m, interior 3 / boundary up to 2 neighbours.");

        // Match PipelineWorker::buildMeshGPU; no PLY export is needed for area.
        const auto inputs = fs.prepare_gpu_mesh_inputs(rectified_mask);
        fs::MeshGPUBuffer mesh;
        const auto stats = fs::build_mesh_gpu(inputs, mesh, 0.02, 0.01);
        if (stats.triangle_count == 0) {
            throw std::runtime_error("GPU mesh has no triangles after filtering");
        }
        std::ostringstream summary;
        summary << "GPU mesh: " << stats.triangle_count << " triangles\n"
                << "Area: " << std::fixed << std::setprecision(8) << stats.area_m2 << " m^2 ("
                << std::setprecision(2) << stats.area_m2 * 1e4 << " cm^2)";
        Logger::log(summary.str());
        return 0;
    } catch (const std::exception& error) {
        Logger::error(std::string("TSFS: ") + error.what());
        return 1;
    }
}
