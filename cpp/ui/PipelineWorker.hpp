#pragma once
#include "CalibrationWorker.hpp"
#include "fs/inference/FS.hpp"
#include "fs/inference/MA-VGGT.hpp"
#include "fs/inference/SamSegmenter.hpp"
#include <atomic>
#include <functional>
#include <optional>
#include "GPUMeshFrame.hpp"

struct ReconstructionSaveOptions {
    bool images{true}, calibration{true}, mask{true}, mesh{true}, depth{true};
};

enum class CameraMode { Sentech, RealSenseD435 };

// Camera/texture snapshots and completed GPU output retained for multi-view work.
// Full stereo images are needed only for the current pair; the UI owns review images.
struct CapturedStereoPair {
    quint64 image_id{0};
    fs::MeshCamera image_camera, ma_camera;
    SharedGPUMesh point_cloud;
    QImage ma_image;
};

struct PipelineState {
    static constexpr int kMaxCaptures = 5;
    int capture_count{0};
    QString import_directory;
    int import_index{0}, import_total{0}; // Number of pairs imported successfully, and pairs in the folder.
    bool import_sequence_started{false};
    CameraMode camera_mode{CameraMode::Sentech};
    bool connected{false}, live{false}, has_rectified{false}, gpu_ready{false}, depth_ready{false}, engine_ready{false};
    QString input{"No stereo pair loaded"}, calibration, status{"Import a capture directory or preview the cameras."};
    QString calibration_filename{"calibration.json"};
    bool sam_ready{false}, ma_ready{false};
    int progress{0};
    QString progress_stage{"Ready to capture"};
    bool action_failed{false};
    quint64 image_id{0};
    std::optional<fs::MeshCamera> image_camera;
    SharedPredictedCameras predicted_cameras;
    SharedGPUClouds ma_clouds;
};
Q_DECLARE_METATYPE(PipelineState)

// Serialized source, CPU and GPU work. The GUI receives owned images and immutable mesh results.
class PipelineWorker : public QObject {
    Q_OBJECT
public:
    PipelineWorker(ConfirmedCalibration calibration, QString path, SharedStereoSource source,
                   std::shared_ptr<std::atomic_bool> cancel, double exposure_us = SentechStereoOptions{}.exposure_us);
    void execute(const std::function<void(PipelineWorker&)>& action);
    void initialize(const QString& engine_path, const QString& sam_encoder = {}, const QString& sam_decoder = {},
                    const QString& ma_engine = {});
    void setImportDirectory(const QString& directory);
    void importNextPair(bool use_capture_calibration);
    void connectCameras(CameraMode mode = CameraMode::Sentech);
    void disconnectCameras();
    void setLive(bool enabled);
    void startCapturePreview(bool append);
    void cleanCaptures();
    void freeze();
    void captureAndReconstruct(float minimum, float maximum, bool denoise = true,
                               float max_neighbor_distance_m = 0.003F);
    void reconstruct(float minimum, float maximum, const QImage& selection_mask,
                     bool denoise = true, float max_neighbor_distance_m = 0.003F);
    void segment(quint64 image_id, quint64 request_id, const std::vector<fs::SamPrompt>& prompts,
                 const std::shared_ptr<std::atomic_uint64_t>& current_request, const QImage& image = {});
    // Triangle meshes are available only for single-view FS geometry.
    void buildMeshGPU(const QImage& selection_mask, double max_edge_m = .01);
    void saveResults(const QString& directory, ReconstructionSaveOptions options,
                     const QImage& selection, SharedGPUMesh gpu_mesh = {}, bool overwrite = false,
                     SharedGPUClouds ma_geometry = {});
    void shutdown();
signals:
    void stateChanged(PipelineState state);
    void images(QImage rectified_left, QImage rectified_right);
    void preview(QImage rectified_left, QImage rectified_right);
    void depthImage(QImage depth_rgb, float minimum, float maximum);
    void gpuMeshReady(SharedGPUMesh mesh);
    void maskReady(quint64 image_id, quint64 request_id, QImage mask, QString message);
    void log(QString message);
    void actionFinished();
    void stopped();
private:
    void poll();
    void publish();
    QString liveStatus() const;
    void emitPreview();
    void checkpoint() const;
    void reportProgress(int percent, const QString& stage);
    void buildPointCloud();
    void retainCapture(SharedGPUMesh point_cloud);
    void loadCapture(const QString& directory, const QString& left_name, const QString& right_name,
                     bool use_capture_calibration, bool append);
    void prepare(std::unique_ptr<StereoFrame> frame, QString input, QString calibration,
                 QString calibration_filename, QByteArray calibration_json, StereoRectificationCache* maps = nullptr);
    ConfirmedCalibration confirmed_;
    QString confirmed_path_;
    SharedStereoSource source_;
    StereoCalibration camera_calibration_;
    cv::Size camera_size_;
    QString camera_calibration_description_, camera_calibration_filename_;
    QByteArray camera_calibration_json_;
    const double exposure_us_; // Requested setting handed off by calibration; also used on reconnect.
    std::shared_ptr<std::atomic_bool> cancel_;
    QTimer* timer_;
    QElapsedTimer last_pair_;
    std::optional<StereoCameraPair> latest_;
    cv::Mat preview_left_map_x_, preview_left_map_y_, preview_right_map_x_, preview_right_map_y_;
    cv::Mat preview_left_, preview_right_;
    std::unique_ptr<StereoFrame> frame_;
    std::vector<CapturedStereoPair> captures_;
    std::vector<std::pair<QString,QString>> import_pairs_;
    struct ImportCalibration {
        QString directory, description, filename;
        bool own{true}, compact{false};
        StereoCalibration calibration;
        std::optional<cv::Size> image_size;
        QByteArray json;
        StereoRectificationCache maps;
    };
    std::optional<ImportCalibration> import_calibration_;
    int capture_slot_{0}; // Retake replaces this slot; Capture more selects the next slot.
    std::shared_ptr<FS> fs_;
    std::unique_ptr<fs::SamSegmenter> sam_;
    quint64 sam_image_id_{0};
    std::shared_ptr<fs::MA_VGGT> ma_;
    // Filtered Z depth retained for TIFF export; geometry stays on the GPU.
    cv::Mat depth_;
    QByteArray calibration_json_;
    std::weak_ptr<const GPUMeshFrame> latest_gpu_mesh_;
    std::shared_ptr<fs::MeshGPUBuffer> gpu_mesh_;
    PipelineState state_;
};
