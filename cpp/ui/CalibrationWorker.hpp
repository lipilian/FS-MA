#pragma once
#include "fs/calibration/CharucoCalibration.hpp"
#include "fs/capture/SentechStereoSource.hpp"
#include <QObject>
#include <QImage>
#include <QElapsedTimer>
#include <QStringList>
#include <QTimer>
#include <functional>
#include <memory>
#include <optional>

struct CalibrationState {
    fs::calibration::BoardConfig board;
    bool connected{false}, busy{false}, has_result{false}, can_finish{false};
    int candidates{0}, selected{-1};
    QStringList samples;
    QString status{"Connect the stereo cameras to begin."}, quality{"No calibration yet."}, saved_path;
};
using ConfirmedCalibration = std::shared_ptr<const fs::calibration::SessionResult>;
Q_DECLARE_METATYPE(CalibrationState)
Q_DECLARE_METATYPE(ConfirmedCalibration)

// All source lifecycle, detection and solving run exclusively on this object's thread.
class CalibrationWorker : public QObject {
    Q_OBJECT
public:
    using SourceFactory = std::function<std::unique_ptr<IStereoSource>(double exposure_us)>;
    explicit CalibrationWorker(SourceFactory factory = {}, QObject* parent = nullptr);
    void execute(const std::function<void(CalibrationWorker&)>& action);
    void connectCameras(double exposure_us);
    void disconnectCameras();
    void applyBoard(fs::calibration::BoardConfig board);
    void capture(bool for_check);
    void cancelCapture();
    void compute();
    void selectSample(int index);
    void deleteSample(int index);
    void setDisplay(bool detect_corners, bool rectified);
    void setThreshold(double pixels);
    void save(const QString& path);
    void load(const QString& path);
    void finish();
    void shutdown();
signals:
    void stateChanged(CalibrationState state);
    void preview(QImage left, QImage right, QString caption);
    void failed(QString message);
    void actionFinished();
    void completed(ConfirmedCalibration result, QString path);
    void stopped();
private:
    void poll();
    void publish();
    void invalidate();
    void show(const fs::calibration::Sample& sample, const QString& caption);
    void buildMaps();
    bool ready() const;
    CalibrationState state_;
    SourceFactory source_factory_;
    std::unique_ptr<IStereoSource> source_;
    QTimer* timer_;
    QElapsedTimer capture_clock_, last_frame_clock_;
    cv::Ptr<cv::aruco::CharucoBoard> board_;
    std::vector<fs::calibration::Sample> samples_;
    std::optional<fs::calibration::Sample> candidate_, latest_, check_sample_;
    std::optional<fs::calibration::SessionResult> result_;
    cv::Size image_size_;
    cv::Mat lx_, ly_, rx_, ry_;
    enum class Capture { None, Sample, Check } capture_{Capture::None};
    bool detection_{true}, rectified_{false}, shutting_down_{false};
    double threshold_{1.0};
};
