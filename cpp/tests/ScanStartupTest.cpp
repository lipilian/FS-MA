#include "DesktopController.hpp"
#include "widgets/MeshView.hpp"
#include "widgets/MaskEditor.hpp"
#include <QApplication>
#include <QCheckBox>
#include <QComboBox>
#include <QLabel>
#include <QDialog>
#include <QDataStream>
#include <QFile>
#include <QLineEdit>
#include <QMouseEvent>
#include <QOpenGLWidget>
#include <QPointer>
#include <QPushButton>
#include <QTemporaryDir>
#include <QTabWidget>
#include <QTimer>
#include <opencv2/imgcodecs.hpp>
#include <cmath>
#include <iostream>
#include <stdexcept>

namespace {
void require(bool ok, const char* message) { if (!ok) throw std::runtime_error(message); }

void checkGeometryExport(const QString& directory, const GPUMeshFrame& source) {
    QFile ply(directory+"/mesh.ply");
    require(ply.open(QIODevice::ReadOnly), "GPU geometry export did not produce mesh.ply");
    quint64 vertices=0, faces=0;
    QByteArray line;
    while (!(line=ply.readLine()).isEmpty() && line!="end_header\n") {
        if (line.startsWith("element vertex ")) vertices=line.mid(15).trimmed().toULongLong();
        if (line.startsWith("element face ")) faces=line.mid(13).trimmed().toULongLong();
    }
    require(line=="end_header\n" && vertices>0, "Invalid PLY header");
    require(source.point_cloud ? vertices==source.stats.point_count && faces==0
                               : faces==source.stats.triangle_count && faces>0,
            "PLY geometry counts differ from the displayed GPU result");
    require(ply.size()-ply.pos()==qint64(vertices*15+faces*13), "Incorrect PLY payload size");
    QDataStream stream(&ply); stream.setByteOrder(QDataStream::LittleEndian);
    stream.setFloatingPointPrecision(QDataStream::SinglePrecision);
    for (quint64 i=0;i<vertices;++i) {
        float x,y,z; quint8 r,g,b;
        stream>>x>>y>>z>>r>>g>>b;
        require(std::isfinite(x) && std::isfinite(y) && std::isfinite(z) && z>0, "Invalid exported vertex");
    }
    for (quint64 i=0;i<faces;++i) {
        quint8 size; qint32 a,b,c; stream>>size>>a>>b>>c;
        require(size==3 && a>=0 && b>=0 && c>=0 && quint64(a)<vertices && quint64(b)<vertices && quint64(c)<vertices,
                "Invalid exported triangle indices");
    }
    require(stream.status()==QDataStream::Ok, "Truncated PLY data");
    const auto depth=cv::imread((directory+"/depth.tiff").toStdString(),cv::IMREAD_UNCHANGED);
    require(depth.type()==CV_32FC1 && depth.size()==cv::Size(960,800) && cv::checkRange(depth), "Invalid depth TIFF");
    if (source.point_cloud)
        require(quint64(cv::countNonZero(depth))==vertices, "Saved depth does not match the captured point cloud");
}

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
    int phase=0, ready_ticks=0, presented=0;
    QTemporaryDir point_export, mesh_export;
    require(point_export.isValid() && mesh_export.isValid(), "Cannot create export test directories");
    SharedGPUMesh exported_geometry;
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
                    require(!window->findChild<QPushButton*>("buildMeshCPU"), "CPU mesh button still exists");
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
                        if (finished || !mesh) return;
                        if (!(mesh->point_cloud ? mesh->stats.point_count : mesh->stats.triangle_count)) {
                            finish("RealSense reconstruction produced empty geometry"); return;
                        }
                        ++presented;
                        std::cout<<"RealSense geometry displayed: "<<mesh->stats.point_count<<" points, "<<mesh->stats.triangle_count<<" triangles\n";
                        // Inspect the framebuffer after the presentation callback returns.
                        QTimer::singleShot(250,&app,[&,mesh] {
                            auto* canvas=window ? window->findChild<QOpenGLWidget*>() : nullptr;
                            try {
                                require(canvas && canvas->isValid() && !canvas->grabFramebuffer().isNull(),
                                        "No valid OpenGL framebuffer after capture");
                                require(!window->findChild<QWidget*>("workspaceTools")->isVisible(),
                                        "Scan settings sidebar reappeared after capture");
                                require(!window->findChild<QPushButton*>("run"),
                                        "Scan still creates an unused manual reconstruction button");
                                auto* tabs=window->findChild<QTabWidget*>("workspaceTabs");
                                if (presented==1) {
                                    tabs->setCurrentIndex(1);
                                    auto* editor=window->findChild<MaskEditor*>();
                                    window->findChild<QPushButton*>("maskBrush")->click();
                                    const QPointF center=editor->rect().center();
                                    const QPointF global=editor->mapToGlobal(center.toPoint());
                                    QMouseEvent press(QEvent::MouseButtonPress,center,global,Qt::LeftButton,Qt::LeftButton,Qt::NoModifier);
                                    QMouseEvent release(QEvent::MouseButtonRelease,center,global,Qt::LeftButton,Qt::NoButton,Qt::NoModifier);
                                    QApplication::sendEvent(editor,&press); QApplication::sendEvent(editor,&release);
                                    auto* confirm=window->findChild<QPushButton*>("finishDraw");
                                    require(confirm->isEnabled(), "Region cannot be confirmed after brush input");
                                    confirm->click(); // Must rebuild without any manual reconstruction button.
                                } else if (presented==2) {
                                    tabs->setCurrentIndex(1);
                                    window->findChild<QPushButton*>("clearMask")->click();
                                } else {
                                    if (!mesh->point_cloud) {
                                        auto* mode=window->findChild<QComboBox*>("meshMode");
                                        const auto surface=canvas->grabFramebuffer();
                                        mode->setCurrentIndex(2);
                                        const auto wireframe=canvas->grabFramebuffer();
                                        require(!wireframe.isNull() && wireframe!=surface, "GPU wireframe did not change the rendered mesh");
                                        mode->setCurrentIndex(1);
                                    }
                                    auto* save=window->findChild<QPushButton*>("openSaveResults");
                                    require(save && save->isVisible() && save->isEnabled(), "Saving is inaccessible without the sidebar");
                                    save->click();
                                    auto* dialog=window->findChild<QDialog*>("saveResultsDialog");
                                    require(dialog && dialog->isVisible() && dialog->findChild<QPushButton*>("browseSaveDirectory")->isVisible(),
                                            "Save dialog did not expose the existing export controls");
                                    exported_geometry=mesh;
                                    phase=mesh->point_cloud ? 4 : 6;
                                    dialog->findChild<QLineEdit*>("saveDirectory")->setText(mesh->point_cloud ? point_export.path() : mesh_export.path());
                                    auto* save_selected=dialog->findChild<QPushButton*>("saveSelected");
                                    require(save_selected->isEnabled(), "Cannot save displayed GPU geometry");
                                    save_selected->click();
                                    dialog->close();
                                }
                            } catch (const std::exception& error) { finish(QString::fromUtf8(error.what())); }
                        });
                    });
                }
            }
            if (!window) return;
            if ((phase==4 || phase==6) && window->findChild<QPushButton*>("saveSelected")->isEnabled()) {
                const auto status=window->findChild<QLabel*>("status")->text();
                if (!status.startsWith("Saved ")) throw std::runtime_error(status.toStdString());
                checkGeometryExport(phase==4 ? point_export.path() : mesh_export.path(),*exported_geometry);
                if (phase==4) {
                    auto* build=window->findChild<QPushButton*>("buildMeshGPU");
                    require(build && build->isEnabled(), "GPU mesh generation is unavailable");
                    phase=5; build->click();
                } else {
                    const auto points_depth=cv::imread(point_export.filePath("depth.tiff").toStdString(),cv::IMREAD_UNCHANGED);
                    const auto mesh_depth=cv::imread(mesh_export.filePath("depth.tiff").toStdString(),cv::IMREAD_UNCHANGED);
                    require(cv::norm(points_depth,mesh_depth,cv::NORM_INF)==0, "GPU mesh build changed the saved depth");
                    finish({});
                }
                return;
            }
            auto* camera=window->findChild<QPushButton*>("connectCameras");
            auto* capture=window->findChild<QPushButton*>("capturePair");
            const bool connected=camera->isEnabled() && camera->text()=="Disconnect cameras" && capture->isEnabled();
            if ((phase==0 || phase==2) && connected) {
                if (++ready_ticks<8) return; // Allow fresh camera frames to arrive.
                ready_ticks=0;
                if (phase==0) { camera->click(); phase=1; }
                else {
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
    std::cout<<"PASS: skip calibration, load FS/SAM/MA, D435 capture/reconnect, automatic region updates, GPU mesh/wireframe, PLY/depth export, CUDA/OpenGL display and clean shutdown\n";
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
