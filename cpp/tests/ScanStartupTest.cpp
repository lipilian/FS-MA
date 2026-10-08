#include "DesktopController.hpp"
#include "widgets/MeshView.hpp"
#include <QApplication>
#include <QCheckBox>
#include <QComboBox>
#include <QDoubleSpinBox>
#include <QLabel>
#include <QOpenGLWidget>
#include <QPointer>
#include <QPushButton>
#include <QTemporaryDir>
#include <QTimer>
#include <opencv2/imgcodecs.hpp>
#include <iostream>
#include <stdexcept>

namespace {
void require(bool ok, const char* message) { if (!ok) throw std::runtime_error(message); }

void checkUncalibratedPipeline() {
    PipelineWorker worker({}, {}, {}, std::make_shared<std::atomic_bool>(false));
    PipelineState state;
    QObject::connect(&worker, &PipelineWorker::stateChanged, [&](PipelineState update) { state=std::move(update); });
    worker.execute([](auto&) {});
    require(state.camera_mode==CameraMode::RealSenseD435 && !state.live_camera,
            "Calibration-free startup must select RealSense without invented intrinsics");
    worker.execute([](auto& w) { w.connectCameras(CameraMode::Sentech); });
    require(state.action_failed && state.status.contains("require confirmed calibration"),
            "Sentech must reject missing calibration without dereferencing it");
    worker.execute([](auto& w) { w.importCapture("", false); });
    require(state.action_failed && state.status.contains("No confirmed calibration"),
            "Import must not fall back to absent calibration");
    worker.execute([](auto& w) { w.initialize(""); });
    require(state.action_failed && !state.engine_ready && !state.connected,
            "Missing model must fail before connecting cameras");

    // Saved captures remain usable without a live camera or a ChArUco session.
    QTemporaryDir dir;
    require(dir.isValid(), "Cannot create capture fixture");
    const cv::Mat rgb(24,32,CV_8UC3,cv::Scalar(80,120,160));
    for (const auto* name : {"left.png","right.png"})
        require(cv::imwrite(dir.filePath(name).toStdString(),rgb), "Cannot write capture image");
    const cv::Mat k=(cv::Mat_<double>(3,3)<<100,0,16,0,100,12,0,0,1);
    const cv::Mat distortion=cv::Mat::zeros(1,5,CV_64F), rotation=cv::Mat::eye(3,3,CV_64F);
    const cv::Mat translation=(cv::Mat_<double>(3,1)<<-.05,0,0);
    {
        cv::FileStorage json(dir.filePath("calibration.json").toStdString(),cv::FileStorage::WRITE);
        json<<"image_width"<<32<<"image_height"<<24
            <<"left_camera_matrix"<<k<<"right_camera_matrix"<<k
            <<"left_distortion"<<distortion<<"right_distortion"<<distortion
            <<"right_to_left_rotation"<<rotation<<"right_to_left_translation"<<translation;
    }
    worker.execute([&](auto& w) { w.importCapture(dir.path(),true); });
    require(!state.action_failed && state.has_rectified && state.image_camera.has_value(),
            "Capture import with its own calibration failed without a calibration session");
    worker.shutdown();
    std::cout<<"PASS: startup without calibration, guarded Sentech/import, missing model, saved capture import\n";
}

// Explicit opt-in: needs the actual engines, D435 and an NVIDIA OpenGL display.
int checkLiveScan(QApplication& app) {
    DesktopController desktop(DesktopController::StartupMode::RealSenseScan);
    QPointer<ReconstructionWindow> window;
    bool finished=false, passed=false;
    QString failure;
    int phase=0, ready_ticks=0;
    const auto finish=[&](QString error) {
        if (finished) return;
        finished=true; failure=std::move(error); passed=failure.isEmpty();
        for (auto* widget : QApplication::topLevelWidgets())
            if (qobject_cast<ReconstructionWindow*>(widget) || qobject_cast<InferenceSplashWindow*>(widget))
                widget->close();
    };
    QTimer poll;
    QObject::connect(&poll,&QTimer::timeout,[&] {
        if (finished) return;
        try {
            for (auto* widget : QApplication::topLevelWidgets()) {
                require(!qobject_cast<CalibrationWindow*>(widget), "scan_gui opened a calibration window");
                if (auto* splash=qobject_cast<InferenceSplashWindow*>(widget)) {
                    require(!splash->findChild<QLabel*>("step")->text().contains("CALIBRATION"),
                            "Scan loading screen claims a completed calibration");
                    if (splash->findChild<QPushButton*>("retryInitialization")->isVisible())
                        throw std::runtime_error(splash->findChild<QLabel*>("loadingStatus")->text().toStdString());
                }
                if (!window) {
                    window=qobject_cast<ReconstructionWindow*>(widget);
                    if (!window) continue;
                    const auto* mode=window->findChild<QComboBox*>("cameraMode");
                    require(mode && mode->count()==1 && mode->currentData().toInt()==int(CameraMode::RealSenseD435),
                            "Scan workspace must default to RealSense only");
                    const auto* calibration=window->findChild<QCheckBox*>("captureCalibration");
                    require(calibration && calibration->isChecked() && !calibration->isEnabled(),
                            "Capture import must use the capture's calibration");
                    auto* view=window->findChild<MeshView*>();
                    require(view, "No OpenGL mesh view");
                    QObject::connect(view,&MeshView::renderFailed,&app,[&](QString error) { finish(error); });
                    QObject::connect(view,&MeshView::gpuMeshPresented,&app,[&](SharedGPUMesh mesh) {
                        if (finished || !mesh || !mesh->point_cloud) return;
                        if (mesh->stats.point_count==0) { finish("RealSense capture produced no points"); return; }
                        std::cout<<"RealSense points displayed: "<<mesh->stats.point_count<<'\n';
                        // Inspect the framebuffer after the presentation callback returns.
                        QTimer::singleShot(250,&app,[&] {
                            auto* canvas=window ? window->findChild<QOpenGLWidget*>() : nullptr;
                            if (!canvas || !canvas->isValid() || canvas->grabFramebuffer().isNull())
                                finish("No valid OpenGL framebuffer after capture");
                            else finish({});
                        });
                    });
                }
            }
            if (!window) return;
            auto* camera=window->findChild<QPushButton*>("connectCameras");
            auto* capture=window->findChild<QPushButton*>("capturePair");
            const bool connected=camera->isEnabled() && camera->text()=="Disconnect cameras" && capture->isEnabled();
            if ((phase==0 || phase==2) && connected) {
                if (++ready_ticks<8) return; // Allow fresh camera frames to arrive.
                ready_ticks=0;
                if (phase==0) { camera->click(); phase=1; }
                else {
                    window->findChild<QDoubleSpinBox*>("maximumDepth")->setValue(5.0);
                    capture->click(); phase=3;
                }
            } else if (phase==1 && camera->isEnabled() && camera->text()=="Connect cameras") {
                camera->click(); phase=2;
            } else if (phase==0 && camera->isEnabled() && camera->text()=="Connect cameras") {
                throw std::runtime_error(window->findChild<QLabel*>("status")->text().toStdString());
            }
        } catch (const std::exception& error) { finish(QString::fromUtf8(error.what())); }
    });
    QTimer::singleShot(180000,&app,[&] { finish("Timed out waiting for RealSense startup / capture / display"); });
    desktop.show(); poll.start(200);
    app.exec();
    require(passed, failure.isEmpty() ? "Scan closed before producing a point cloud" : failure.toUtf8().constData());
    std::cout<<"PASS: skip calibration, load FS/SAM/MA, auto-connect D435, reconnect, capture, CUDA/OpenGL display and clean shutdown\n";
    return 0;
}
}

int main(int argc, char** argv) {
    const bool live=argc==2 && std::string(argv[1])=="--live";
    if (!live) qputenv("QT_QPA_PLATFORM","offscreen");
    QSurfaceFormat::setDefaultFormat(MeshView::surfaceFormat());
    QApplication app(argc,argv);
    cv::setNumThreads(4);
    try {
        if (live) return checkLiveScan(app);
        checkUncalibratedPipeline();
        return 0;
    } catch (const std::exception& error) {
        std::cerr<<error.what()<<'\n'; return 1;
    }
}
