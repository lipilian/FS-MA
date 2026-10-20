#include "DesktopController.hpp"
#include "widgets/MeshView.hpp"
#include "widgets/MaskEditor.hpp"
#include "widgets/StereoImageView.hpp"
#include "widgets/CaptureTransition.hpp"
#include <QApplication>
#include <QCheckBox>
#include <QComboBox>
#include <QLabel>
#include <QDialog>
#include <QDoubleSpinBox>
#include <QDataStream>
#include <QFile>
#include <QFileDialog>
#include <QDir>
#include <QEventLoop>
#include <QLineEdit>
#include <QMouseEvent>
#include <QOpenGLWidget>
#include <QPointer>
#include <QPushButton>
#include <QRadioButton>
#include <QTemporaryDir>
#include <QTabWidget>
#include <QTimer>
#include <opencv2/imgcodecs.hpp>
#include <cmath>
#include <iostream>
#include <stdexcept>

namespace {
void require(bool ok, const char* message) { if (!ok) throw std::runtime_error(message); }

void checkCapturePresentation() {
    StereoImageView preview("Preview");
    preview.resize(360,320);
    QImage input(96,80,QImage::Format_RGB32); input.fill(QColor(70,80,90));
    preview.setImage(input); preview.show();
    const auto idle=preview.grab().toImage();
    require(!preview.isProcessing(), "Preview animates before capture");
    preview.setProcessing(true);
    const auto first=preview.grab().toImage();
    QEventLoop wait;
    QTimer::singleShot(250,&wait,&QEventLoop::quit); wait.exec();
    require(preview.isProcessing() && preview.grab().toImage()==first,
            "Preview must remain still while calculating");
    require(preview.image()==input, "Processing modified the captured image");
    preview.setProcessing(false);
    require(!preview.isProcessing() && preview.grab().toImage()==idle,
            "Preview did not return to its original image after processing");
    QWidget scene; scene.resize(900,600); scene.show();
    CaptureTransition transition(&scene);
    const QPolygonF source{{20,40},{320,40},{320,290},{20,290}};
    QPolygonF destination{{650,150},{690,160},{680,190},{640,180}};
    int finished=0;
    QObject::connect(&transition,&CaptureTransition::finished,[&] { ++finished; });
    require(transition.start(input,[&] { return source; },[&] { return destination; }), "Transfer did not start");
    require(transition.imageQuad()==source, "Transfer did not start at the preview image");
    QTimer::singleShot(400,&wait,&QEventLoop::quit); wait.exec();
    const auto middle=transition.imageQuad().boundingRect();
    require(transition.isRunning() && middle.left()>source.boundingRect().left() &&
            middle.width()<source.boundingRect().width() && middle.width()>destination.boundingRect().width(),
            "Transfer did not translate and shrink the image");
    destination.translate(20,10); // Follow the real camera projection if the view or splitter moves.
    QTimer::singleShot(650,&wait,&QEventLoop::quit); wait.exec();
    require(!transition.isRunning() && !transition.isVisible() && finished==1 && transition.imageQuad()==destination,
            "Transfer did not land on the updated camera corners and finish");
    require(transition.start(input,[&] { return source; },[&] { return destination; }), "Transfer cannot restart");
    transition.cancel();
    require(!transition.isRunning() && !transition.isVisible() && finished==2, "Transfer did not cancel cleanly");
    require(!transition.start(input,[&] { return source; },[] { return QPolygonF{}; }),
            "Transfer started without a projected camera plane");
    MeshView unavailable;
    int render_errors=0;
    QObject::connect(&unavailable,&MeshView::renderFailed,[&](const QString&) { ++render_errors; });
    auto geometry=std::make_shared<std::vector<SharedGPUMesh>>(); geometry->push_back(std::make_shared<GPUMeshFrame>());
    unavailable.setMAClouds(geometry); unavailable.setMAClouds(geometry); unavailable.setMAClouds({});
    require(render_errors==1,"Unavailable MA renderer repeated its error on every UI refresh");
    std::cout<<"PASS: static calculation preview, shrinking camera transfer, moving target, completion and cancellation\n";
}

bool hasSceneContent(QOpenGLWidget* canvas) {
    require(canvas && canvas->isValid(), "No valid OpenGL canvas");
    const auto frame=canvas->grabFramebuffer();
    require(!frame.isNull(), "No OpenGL framebuffer");
    const auto background=frame.pixel(0,0);
    for (int y=0;y<frame.height();y+=4)
        for (int x=0;x<frame.width();x+=4)
            if (frame.pixel(x,y)!=background) return true;
    return false;
}
bool hasGreenCamera(QOpenGLWidget* canvas) {
    const auto frame=canvas->grabFramebuffer();
    for (int y=0;y<frame.height();++y) for (int x=0;x<frame.width();++x) {
        const auto p=frame.pixelColor(x,y);
        if (p.green()>230 && p.red()<50 && p.blue()<100) return true;
    }
    return false;
}

void checkGeometryExport(const QString& directory, const GPUMeshFrame& source, bool world=false) {
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
        require(std::isfinite(x) && std::isfinite(y) && std::isfinite(z) && (world || z>0), "Invalid exported vertex");
    }
    for (quint64 i=0;i<faces;++i) {
        quint8 size; qint32 a,b,c; stream>>size>>a>>b>>c;
        require(size==3 && a>=0 && b>=0 && c>=0 && quint64(a)<vertices && quint64(b)<vertices && quint64(c)<vertices,
                "Invalid exported triangle indices");
    }
    require(stream.status()==QDataStream::Ok, "Truncated PLY data");
    const auto depth=cv::imread((directory+"/depth.tiff").toStdString(),cv::IMREAD_UNCHANGED);
    require(depth.type()==CV_32FC1 && depth.size()==cv::Size(960,800) && cv::checkRange(depth), "Invalid depth TIFF");
    // The TIFF holds the current FS view. MA PLY combines all predicted views
    // on a different grid; its point count is checked against GPU stats above.
    if (source.point_cloud && !world)
        require(quint64(cv::countNonZero(depth))==vertices, "Saved depth does not match the captured point cloud");
}

void checkUncalibratedPipeline() {
    PipelineWorker worker({}, {}, {}, std::make_shared<std::atomic_bool>(false));
    PipelineState state;
    QObject::connect(&worker, &PipelineWorker::stateChanged, [&](PipelineState update) { state=std::move(update); });
    worker.execute([](auto&) {});
    require(state.camera_mode==CameraMode::RealSenseD435 && !state.image_camera && !state.has_rectified,
            "Calibration-free startup must select RealSense without invented intrinsics");
    worker.execute([](auto& w) { w.connectCameras(CameraMode::Sentech); });
    require(state.action_failed && state.status.contains("require confirmed calibration"),
            "Sentech must reject missing calibration without dereferencing it");
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
    worker.execute([&](auto& w) { w.setImportDirectory(dir.path()); w.importNextPair(false); });
    require(state.action_failed && state.status.contains("No confirmed calibration") && state.import_index==0,
            "Import must not fall back to absent calibration or skip the pair");
    worker.execute([](auto& w) { w.importNextPair(true); });
    require(!state.action_failed && state.has_rectified && state.image_camera.has_value(),
            "Capture import with its own calibration failed without a calibration session");
    worker.shutdown();
    std::cout<<"PASS: startup without calibration, guarded Sentech/import, missing model, saved capture import\n";
}

void makeImportFixture(const QTemporaryDir& dir) {
    require(dir.isValid(),"Cannot create import fixture");
    for (const auto* name:{"left2.jpg","right2.png","left10.jpg","right10.jpg"})
        require(cv::imwrite(dir.filePath(name).toStdString(),cv::Mat(24,32,CV_8UC3,cv::Scalar(80,120,160))),
                "Cannot write numbered import image");
    QFile file(dir.filePath("stereo_calibration.json"));
    require(file.open(QIODevice::WriteOnly),"Cannot write compact calibration");
    file.write(R"({"K1":[[100,0,16],[0,100,12],[0,0,1]],"D1":[0,0,0,0,0],
        "K2":[[100,0,16],[0,100,12],[0,0,1]],"D2":[0,0,0,0,0],
        "R":[[1,0,0],[0,1,0],[0,0,1]],"T":[-50,0,0]})");
}

void checkImportFolder() {
    QTemporaryDir dir, output, invalid;
    makeImportFixture(dir);
    PipelineWorker worker({}, {}, {}, std::make_shared<std::atomic_bool>(false));
    PipelineState state;
    int images=0;
    QObject::connect(&worker,&PipelineWorker::stateChanged,[&](PipelineState update) { state=std::move(update); });
    QObject::connect(&worker,&PipelineWorker::images,[&](QImage,QImage) { ++images; });
    worker.execute([&](auto& w) { w.setImportDirectory(dir.path()); });
    require(!state.action_failed && state.import_directory==dir.path() && state.import_total==2 &&
            state.import_index==0 && !state.has_rectified && !state.connected && !state.live && images==0,
            "Selecting a folder must select the source without importing a pair");
    worker.execute([](auto& w) { w.importNextPair(true); });
    require(!state.action_failed && state.import_index==1 && state.has_rectified && images==1 &&
            state.input.contains("left2.jpg / right2.png"),"Pairs were not imported in numeric order");
    const auto first_image_id=state.image_id;
    const ReconstructionSaveOptions options{true,true,false,false,false};
    worker.execute([&](auto& w) { w.saveResults(output.path(),options,{}); });
    require(!state.action_failed,"Imported input could not be saved");
    const auto exported=StereoCalibration::from_file(output.filePath("calibration.json").toStdString());
    require(std::abs(cv::norm(exported.right_to_left_translation)-.05)<1e-12,
            "Compact translation was not normalized to metres for export");
    // Following pairs and Restart must use the first successful calibration
    // snapshot, even if the original file is no longer available.
    require(QFile::remove(dir.filePath("stereo_calibration.json")),"Cannot remove cached calibration fixture");

    // A file can disappear after selecting a directory. A failed import must not skip it.
    require(QFile::remove(dir.filePath("right10.jpg")),"Cannot remove import fixture");
    worker.execute([](auto& w) { w.importNextPair(true); });
    require(state.action_failed && state.import_index==1 && state.image_id==first_image_id && images==1 &&
            state.import_directory==dir.path(),"Failed import lost the current pair or advanced the folder cursor");
    worker.execute([&](auto& w) { w.setImportDirectory(invalid.path()); });
    require(state.action_failed && state.import_directory==dir.path() && state.import_index==1,
            "An invalid replacement folder discarded the current source");
    require(cv::imwrite(dir.filePath("right10.jpg").toStdString(),cv::Mat(24,32,CV_8UC3,cv::Scalar(80,120,160))),
            "Cannot restore import fixture");
    worker.execute([](auto& w) { w.importNextPair(true); });
    require(!state.action_failed && state.import_index==2 && images==2 && state.input.contains("left10.jpg"),
            "Import did not retry the same numbered pair");
    worker.execute([](auto& w) { w.importNextPair(true); });
    require(state.action_failed && state.import_index==2 && images==2 && state.import_directory==dir.path(),
            "End of folder should retain import mode and stop without repeating a pair");
    worker.execute([](auto& w) { w.cleanCaptures(); });
    require(!state.action_failed && state.import_directory==dir.path() && state.import_index==0 &&
            !state.has_rectified && !state.live,"Restart did not retain the import source and reset its sequence");
    worker.execute([](auto& w) { w.importNextPair(true); });
    require(!state.action_failed && state.import_index==1 && images==3,"Cannot import after restart");
    worker.execute([&](auto& w) { w.saveResults(output.path(),options,{}, {},true); });
    require(!state.action_failed && std::abs(cv::norm(StereoCalibration::from_file(
                output.filePath("calibration.json").toStdString()).right_to_left_translation)-.05)<1e-12,
            "Reusing calibration converted millimetres to metres more than once");

    makeImportFixture(dir);
    {
        QFile calibration(dir.filePath("stereo_calibration.json"));
        require(calibration.open(QIODevice::ReadWrite),"Cannot update calibration fixture");
        const auto changed=calibration.readAll().replace("[-50,0,0]","[-60,0,0]");
        calibration.resize(0); calibration.seek(0); calibration.write(changed);
    }
    worker.execute([&](auto& w) { w.setImportDirectory(dir.path()); w.importNextPair(true); });
    require(!state.action_failed,"Cannot reselect import folder");
    worker.execute([&](auto& w) { w.saveResults(output.path(),options,{}, {},true); });
    require(!state.action_failed && std::abs(cv::norm(StereoCalibration::from_file(
                output.filePath("calibration.json").toStdString()).right_to_left_translation)-.06)<1e-12,
            "Reselecting a folder did not reload its updated calibration");

    // Sentech is unavailable without calibration; no hardware is contacted.
    worker.execute([](auto& w) { w.connectCameras(CameraMode::Sentech); });
    require(state.action_failed && state.import_directory.isEmpty() && state.import_total==0 && state.has_rectified,
            "Connect cameras must exit import mode even when the connection fails, retaining the frozen pair");
    worker.execute([&](auto& w) { w.setImportDirectory(output.path()); w.importNextPair(true); });
    require(!state.action_failed && state.has_rectified,"Normalized exported calibration did not round-trip");
    // Cached image-size metadata must still reject a mismatched subsequent pair.
    for (const auto* name:{"left01.png","right01.png"})
        require(cv::imwrite(output.filePath(name).toStdString(),cv::Mat(30,40,CV_8UC3,cv::Scalar(80,120,160))),
                "Cannot write mismatched grid fixture");
    worker.execute([&](auto& w) { w.setImportDirectory(output.path()); w.importNextPair(true); });
    require(!state.action_failed && state.import_index==1,"Cannot import metadata fixture");
    const auto before_mismatch=state.image_id;
    require(QFile::remove(output.filePath("calibration.json")),"Cannot remove metadata fixture");
    worker.execute([](auto& w) { w.importNextPair(true); });
    require(state.action_failed && state.status.contains("image dimensions") && state.image_id==before_mismatch && state.import_index==1,
            "Cached calibration failed to validate the next pair's image grid");
    worker.shutdown();
    std::cout<<"PASS: deferred folder import, numeric pairing, fixed millimetres, retry, exhaustion, restart, camera mode and export round-trip\n";
    std::cout<<"PASS: calibration reused without source JSON, stable units, folder reselection reload and cached grid validation\n";
}

void checkRectificationCache() {
    QTemporaryDir dir;
    makeImportFixture(dir);
    auto calibration=StereoCalibration::from_file(dir.filePath("stereo_calibration.json").toStdString());
    calibration.right_to_left_translation*=.001;
    calibration.left_distortion.at<double>(0)=-.15;
    calibration.right_distortion.at<double>(0)=-.12;
    StereoRectificationCache cache;
    const auto check=[&](cv::Size size) {
        cv::Mat left(size,CV_8UC3),right(size,CV_8UC3);
        cv::randu(left,0,255); cv::randu(right,0,255);
        StereoFrame fresh(left,right,calibration),cached(left,right,calibration);
        fresh.rectify(); cached.rectify(&cache);
        require(cv::norm(fresh.rectified_left(),cached.rectified_left(),cv::NORM_INF)==0 &&
                cv::norm(fresh.rectified_right(),cached.rectified_right(),cv::NORM_INF)==0,
                "Cached maps differ from freshly generated rectification");
        const auto a=fresh.rectified_camera_parameters(), b=cached.rectified_camera_parameters();
        require(a.fx==b.fx && a.fy==b.fy && a.cx==b.cx && a.cy==b.cy && a.baseline_meters==b.baseline_meters,
                "Cached maps used stale projection parameters");
        return cached;
    };
    auto retained=check({32,24});
    const auto original=retained.rectified_left().clone();
    const auto camera=retained.rectified_camera_parameters();
    check({32,24}); // Reuse maps for different image pixels.
    check({40,30}); // Rebuild maps for a different grid.
    calibration.left_camera_matrix.at<double>(0,0)=125;
    calibration.right_to_left_translation*=1.2;
    check({40,30}); // Rebuild maps for different intrinsics and baseline.
    require(cv::norm(original,retained.rectified_left(),cv::NORM_INF)==0 &&
            camera.fx==retained.rectified_camera_parameters().fx &&
            camera.baseline_meters==retained.rectified_camera_parameters().baseline_meters,
            "Updating a rectification cache modified an already captured frame");
    std::cout<<"PASS: cached/fresh rectification equivalence, grid/calibration invalidation and retained-frame independence\n";
}

void checkImportControls() {
    QTemporaryDir dir, workspace;
    makeImportFixture(dir);
    require(workspace.isValid() && QDir(workspace.path()).mkpath("data/scan_data"),
            "Cannot create default import directory fixture");
    const QString original_directory=QDir::currentPath();
    require(QDir::setCurrent(workspace.path()),"Cannot change to import workspace fixture");
    const QString default_directory=QDir(workspace.filePath("data/scan_data")).canonicalPath();
    PipelineController controller({}, {}, {});
    bool busy=false;
    QObject::connect(&controller,&PipelineController::busyChanged,[&](bool value) { busy=value; });
    const auto drain=[&] {
        QElapsedTimer timeout; timeout.start();
        do { QCoreApplication::processEvents(QEventLoop::AllEvents,20); }
        while (busy && timeout.elapsed()<5000);
        require(!busy,"Timed out waiting for UI action");
    };
    // Supply a model-ready snapshot through the actual controller signal path.
    // No model execution or camera hardware is needed to exercise the controls.
    controller.submit([](auto& w) {
        QTimer::singleShot(0,&w,[&w] {
            PipelineState ready; ready.engine_ready=ready.sam_ready=ready.ma_ready=true;
            w.stateChanged(ready);
        });
    });
    drain();
    QElapsedTimer timeout; timeout.start();
    while (!controller.state().engine_ready && timeout.elapsed()<5000) QCoreApplication::processEvents(QEventLoop::AllEvents,20);
    require(controller.state().engine_ready,"Cannot seed model-ready UI snapshot");
    ReconstructionWindow window(controller,{},{});
    auto* choose=window.findChild<QPushButton*>("importCapture");
    auto* capture=window.findChild<QPushButton*>("capturePair");
    auto* camera=window.findChild<QPushButton*>("connectCameras");
    require(!window.findChild<QComboBox*>("importTranslationUnit"),
            "Import must not show a translation unit selector");
    require(choose && choose->text()=="Import folder…" && capture && capture->text()=="Capture pair",
            "Initial source controls have incorrect labels");
    const auto select_folder=[&](bool accept) {
        bool handled=false;
        QString opened_directory;
        const QString expected_directory=controller.state().import_directory.isEmpty()
            ? default_directory : controller.state().import_directory;
        QTimer::singleShot(0,[&] {
            if (auto* dialog=qobject_cast<QFileDialog*>(QApplication::activeModalWidget())) {
                handled=true;
                opened_directory=dialog->directory().canonicalPath();
                if (accept) {
                    dialog->setDirectory(dir.path()); dialog->selectFile(dir.path());
                    QMetaObject::invokeMethod(dialog,"accept",Qt::DirectConnection);
                } else dialog->reject();
            }
        });
        choose->click();
        require(handled,"Folder chooser was not shown");
        require(opened_directory==expected_directory,
                "Folder chooser must start in data/scan_data or the selected import folder");
        drain();
    };
    select_folder(false);
    require(capture->text()=="Capture pair","Cancelling a folder chooser changed the input mode");
    select_folder(true);
    require(controller.state().import_directory==dir.path() && controller.state().image_id==0 &&
            capture->text()=="Import pair" && camera->text()=="Connect cameras",
            "Selecting a folder imported immediately or failed to switch the primary action");
    auto ready=controller.state(); ready.engine_ready=ready.sam_ready=ready.ma_ready=true;
    controller.stateChanged(ready);
    require(capture->isEnabled(),"Compact import must be ready without a unit selection");
    select_folder(false);
    require(capture->text()=="Import pair" && capture->isEnabled(),"Cancelling replacement selection lost the import source");
    capture->click(); drain();
    require(controller.state().import_index==1 && controller.state().has_rectified && capture->text()=="Import pair",
            "Import button did not import from the selected folder or retain mode after inference failure");
    auto* mode=window.findChild<QComboBox*>("cameraMode");
    mode->setItemData(mode->currentIndex(),int(CameraMode::Sentech)); // Fail before hardware access.
    camera->click(); drain();
    require(controller.state().import_directory.isEmpty() && capture->text()=="Capture pair",
            "Connect cameras did not restore the capture action");
    window.allowClose(); controller.stop();
    require(QDir::setCurrent(original_directory),"Cannot restore working directory");
    std::cout<<"PASS: default scan_data folder, no unit selector, chooser cancellation, deferred import and explicit source switch\n";
}

// Explicit opt-in: exercises real FS/MA inference from files without cameras or OpenGL.
int checkImportedScan(const QString& directory, const QString& engine) {
    PipelineWorker worker({}, {}, {}, std::make_shared<std::atomic_bool>(false));
    PipelineState state;
    SharedGPUMesh geometry;
    QObject::connect(&worker,&PipelineWorker::stateChanged,[&](PipelineState update) { state=std::move(update); });
    QObject::connect(&worker,&PipelineWorker::gpuMeshReady,[&](SharedGPUMesh update) { geometry=std::move(update); });
    const auto run=[&](const std::function<void(PipelineWorker&)>& action) {
        worker.execute(action);
        if (state.action_failed) throw std::runtime_error(state.status.toStdString());
    };
    run([&](auto& w) { w.initialize(engine); w.setImportDirectory(directory); });
    require(state.import_total>=2,"Import integration test requires at least two stereo pairs");
    std::vector<fs::MeshCamera> cameras;
    std::vector<quint64> ids;
    for (int i=0;i<2;++i) {
        run([](auto& w) { w.importNextPair(true); });
        require(state.image_camera.has_value(),"Imported pair has no rectified camera");
        cameras.push_back(*state.image_camera); ids.push_back(state.image_id);
        run([](auto& w) { w.reconstruct(0,1,{},true,.003F); });
        require(state.capture_count==i+1 && geometry && geometry->point_cloud && geometry->stats.point_count>0,
                "Imported FS reconstruction did not retain a valid point cloud");
        if (i==0) {
            QTemporaryDir output;
            require(output.isValid(),"Cannot create GPU import export fixture");
            run([&](auto& w) { w.saveResults(output.path(),{false,false,false,true,true},{},geometry); });
            checkGeometryExport(output.path(),*geometry);
            run([](auto& w) { w.buildMeshGPU({}); });
            require(geometry && !geometry->point_cloud && geometry->stats.triangle_count>0,
                    "Imported first view did not produce a triangle mesh");
        }
    }
    require(state.predicted_cameras && state.predicted_cameras->size()==2 &&
            state.ma_clouds && state.ma_clouds->size()==2,"Two imported views did not produce MA output");
    for (int i=0;i<2;++i) {
        const auto& predicted=(*state.predicted_cameras)[i];
        const auto& cloud=(*state.ma_clouds)[i];
        require(predicted.image_id==ids[i] && predicted.camera.image_size==cameras[i].image_size &&
                predicted.camera.intrinsics==cameras[i].intrinsics,
                "Historical camera changed after releasing its stereo images");
        require(cloud && cloud->stats.point_count>0 && cloud->image_id==ids[i],
                "MA output lost a retained view");
        auto scaled=cameras[i].intrinsics;
        for (int j=0;j<3;++j) {
            scaled(0,j)*=double(fs::MA_VGGT::kInputWidth)/cameras[i].image_size.width;
            scaled(1,j)*=double(fs::MA_VGGT::kInputHeight)/cameras[i].image_size.height;
        }
        require(cloud->camera.intrinsics==scaled &&
                cloud->camera.image_size==cv::Size(fs::MA_VGGT::kInputWidth,fs::MA_VGGT::kInputHeight),
                "MA camera snapshot does not match its image grid");
    }
    worker.shutdown();
    std::cout<<"PASS: imported FS points/mesh, depth export, two-view MA and historical camera snapshots\n";
    return 0;
}

// Explicit opt-in: needs the actual engines, D435 and an NVIDIA OpenGL display.
int checkLiveScan(QApplication& app) {
    DesktopController desktop(DesktopController::StartupMode::RealSenseScan);
    QPointer<ReconstructionWindow> window;
    bool finished=false, passed=false;
    QString failure;
    int phase=0, ready_ticks=0, presented=0, fs_captures=0, ma_presented=0, landed=0, expected_views=0;
    QTemporaryDir point_export, mesh_export, ma_export;
    require(point_export.isValid() && mesh_export.isValid() && ma_export.isValid(), "Cannot create export test directories");
    SharedGPUMesh exported_geometry;
    SharedGPUClouds latest_ma_clouds;
    QHash<quint64,QImage> captured_left, captured_right;
    std::vector<quint64> preview_ids;
    quint64 replaced_view_id=0;
    const auto check_previews=[&] {
        auto* view=window->findChild<MeshView*>();
        auto* left=window->findChild<StereoImageView*>("capturePreview");
        auto* right=window->findChild<StereoImageView*>("rightCapturePreview");
        auto* mode=window->findChild<QComboBox*>("cameraImageMode");
        // Renderer picking is exercised in MAPointCloudViewTest. Here verify
        // the receiver against the full-resolution images seen at capture time.
        for (const auto id:preview_ids) {
            view->cameraClicked(id);
            require(mode->currentIndex()==0 && left->image()==captured_left.value(id) &&
                    right->image()==captured_right.value(id), "Camera click did not restore its original stereo pair");
            mode->setCurrentIndex(1);
            require(mode->currentIndex()==1 && left->image().size()==captured_left.value(id).size() &&
                    left->image()!=captured_left.value(id), "Selected view did not retain its own depth preview");
        }
        view->cameraClicked(preview_ids.back());
        if (replaced_view_id) {
            const auto before=left->image();
            view->cameraClicked(replaced_view_id);
            require(left->image()==before,"Retaken camera still exposes the replaced image");
        }
    };
    const auto screenshot=[&](const QString& name) {
        const auto directory=qEnvironmentVariable("FS_SCAN_TEST_SCREENSHOTS");
        if (!directory.isEmpty() && window) {
            QDir().mkpath(directory);
            require(window->grab().save(directory+"/"+name+".png"), "Cannot save UI verification screenshot");
        }
    };
    const auto captured_views=[&] {
        int count=0;
        for (int i=1;i<=PipelineState::kMaxCaptures;++i)
            count+=window->findChild<QWidget*>(QString("captureSegment%1").arg(i))->property("captured").toBool();
        return count;
    };
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
                    require(!window->findChild<QPushButton*>("buildMeshGPU"), "GPU mesh toolbar button still exists");
                    require(!window->findChild<QPushButton*>("captureMore"), "Capture more still has a separate button");
                    const auto* mesh_edge=window->findChild<QDoubleSpinBox*>("meshMaxEdge");
                    require(mesh_edge && mesh_edge->value()==.01, "GPU mesh edge limit must default to 10 mm");
                    require(!window->findChild<QDoubleSpinBox*>("meshMaxDepthJump"), "Removed mesh depth-jump control still exists");
                    const auto* mode=window->findChild<QComboBox*>("cameraMode");
                    require(mode && mode->count()==1 && mode->currentData().toInt()==int(CameraMode::RealSenseD435),
                            "Scan workspace must default to RealSense only");
                    const auto* calibration=window->findChild<QCheckBox*>("captureCalibration");
                    require(calibration && calibration->isChecked() && !calibration->isEnabled(),
                            "Capture import must use the capture's calibration");
                    auto* view=window->findChild<MeshView*>();
                    require(view, "No OpenGL mesh view");
                    auto* preview=window->findChild<StereoImageView*>("capturePreview");
                    require(preview && preview->isVisible() &&
                            preview->mapTo(window,QPoint(preview->width(),0)).x()<=view->mapTo(window,QPoint()).x(),
                            "2D preview must be visible to the left of the 3D scene");
                    auto* transfer=window->findChild<CaptureTransition*>();
                    require(transfer, "Capture transfer overlay is missing");
                    QObject::connect(transfer,&CaptureTransition::finished,&app,[&] {
                        if (finished) return;
                        try {
                            auto* transfer=window->findChild<CaptureTransition*>();
                            auto* view=window->findChild<MeshView*>();
                            const auto expected=view->cameraImageQuad().translated(view->mapTo(transfer->parentWidget(),QPoint()));
                            const auto actual=transfer->imageQuad();
                            require(actual.size()==4 && expected.size()==4, "Transfer landed without a camera image plane");
                            for (int i=0;i<4;++i)
                                require(QLineF(actual[i],expected[i]).length()<.5, "Transfer missed the rendered camera corners");
                            ++landed;
                        } catch (const std::exception& error) { finish(QString::fromUtf8(error.what())); }
                    });
                    QObject::connect(view,&MeshView::renderFailed,&app,[&](QString error) { finish(error); });
                    QObject::connect(view,&MeshView::maCloudsPresented,&app,[&](SharedGPUClouds clouds) {
                        if (finished) return;
                        try {
                            require(phase>=7 && clouds && clouds->size()==size_t(captured_views()), "MA renderer did not present the complete current view set");
                            unsigned long long points=0;
                            for (const auto& mesh:*clouds) {
                                require(mesh->point_cloud && !mesh->stats.triangle_count && mesh->camera.image_size==cv::Size(518,434), "MA result is not a point cloud on the predicted grid");
                                points+=mesh->stats.point_count;
                            }
                            require(points>0 && clouds!=latest_ma_clouds, "MA point cloud snapshot was empty or reused after new inference");
                            ++ma_presented; latest_ma_clouds=clouds;
                            std::cout<<"MA point clouds displayed: "<<clouds->size()<<" views, "<<points<<" points\n";
                            require(!window->findChild<CaptureTransition*>()->isRunning(), "Switching to MA restarted the capture transfer");
                            require(window->findChild<QRadioButton*>("maPointCloud")->isChecked() && window->findChild<QComboBox*>("meshMode")->currentIndex()==0,
                                    "Manual MA selection did not display point clouds");
                        } catch (const std::exception& error) { finish(QString::fromUtf8(error.what())); }
                    });
                    QObject::connect(view,&MeshView::gpuMeshPresented,&app,[&](SharedGPUMesh mesh) {
                        if (finished || !mesh) return;
                        captured_left.insert(mesh->image_id,window->findChild<StereoImageView*>("capturePreview")->image());
                        captured_right.insert(mesh->image_id,window->findChild<StereoImageView*>("rightCapturePreview")->image());
                        preview_ids.resize(captured_views());
                        if (!preview_ids.empty()) {
                            if (preview_ids.back() && preview_ids.back()!=mesh->image_id) replaced_view_id=preview_ids.back();
                            preview_ids.back()=mesh->image_id;
                        }
                        if (phase>=7) {
                            try {
                                require((phase==8 || phase==10 || phase==13) && mesh->point_cloud && mesh->stats.point_count,
                                        "Unexpected FS result outside capture completion");
                                require(!window->findChild<StereoImageView*>("capturePreview")->isProcessing(),
                                        "FS points appeared before capture computation finished");
                                require(window->findChild<QRadioButton*>("fsPointCloud")->isChecked() &&
                                        !window->findChild<QWidget*>("maPointCloud")->isEnabled(),
                                        "FS was not selected during the completed capture transfer");
                                ++fs_captures;
                                QTimer::singleShot(350,&app,[&] {
                                    if (finished || !window) return;
                                    try {
                                        require(window->findChild<CaptureTransition*>()->isRunning(), "FS points did not start the camera transfer");
                                        require(window->findChild<QComboBox*>("meshMode")->currentIndex()==0 &&
                                                hasSceneContent(window->findChild<QOpenGLWidget*>()),
                                                "FS point clouds are not visible during the camera transfer");
                                    } catch (const std::exception& error) { finish(QString::fromUtf8(error.what())); }
                                });
                            } catch (const std::exception& error) { finish(QString::fromUtf8(error.what())); }
                            return;
                        }
                        if (!(mesh->point_cloud ? mesh->stats.point_count : mesh->stats.triangle_count)) {
                            finish("RealSense reconstruction produced empty geometry"); return;
                        }
                        ++presented;
                        if (presented==1 && mesh->point_cloud) { finish("First capture displayed intermediate points instead of the automatic GPU mesh"); return; }
                        std::cout<<"RealSense geometry displayed: "<<mesh->stats.point_count<<" points, "<<mesh->stats.triangle_count<<" triangles\n";
                        if (presented==1) QTimer::singleShot(350,&app,[&] {
                            if (finished || !window) return;
                            try {
                                auto* transfer=window->findChild<CaptureTransition*>();
                                require(transfer->isRunning() && transfer->isVisible(), "Camera transfer did not run after GPU mesh presentation");
                                require(!window->findChild<StereoImageView*>("capturePreview")->isProcessing(), "Transfer started before calculation finished");
                                require(window->findChild<QComboBox*>("meshMode")->currentIndex()==1 && hasSceneContent(window->findChild<QOpenGLWidget*>()),
                                        "GPU mesh is not rendered during the image transfer");
                                require(!window->findChild<QPushButton*>("capturePair")->isEnabled(), "A new capture can interrupt the image transfer");
                                screenshot("transfer");
                            } catch (const std::exception& error) { finish(QString::fromUtf8(error.what())); }
                        });
                        // Inspect the framebuffer after the presentation callback returns.
                        QTimer::singleShot(presented==1 ? 1200 : 250,&app,[&,mesh] {
                            auto* canvas=window ? window->findChild<QOpenGLWidget*>() : nullptr;
                            try {
                                require(canvas && canvas->isValid() && !canvas->grabFramebuffer().isNull(),
                                        "No valid OpenGL framebuffer after capture");
                                require(!window->findChild<QWidget*>("workspaceTools")->isVisible(),
                                        "Scan settings sidebar reappeared after capture");
                                require(!window->findChild<QPushButton*>("run"),
                                        "Scan still creates an unused manual reconstruction button");
                                require(!window->findChild<StereoImageView*>("capturePreview")->isProcessing(),
                                        "Preview animation continued after capture completed");
                                auto* restart=window->findChild<QPushButton*>("restartCaptures");
                                auto* retake=window->findChild<QPushButton*>("retake");
                                auto* capture=window->findChild<QPushButton*>("capturePair");
                                require(restart->text()=="Restart" && retake->isVisible() && capture->text()=="Capture more",
                                        "Captured toolbar does not expose Restart / Retake / Capture more");
                                const auto retake_position=retake->mapTo(window,QPoint());
                                const auto capture_position=capture->mapTo(window,QPoint());
                                const auto restart_position=restart->mapTo(window,QPoint());
                                require(restart_position.x()+restart->width()<=retake_position.x() &&
                                        capture_position.x()-retake_position.x()-retake->width()>=48 &&
                                        retake_position.y()+retake->height()/2==capture_position.y()+capture->height()/2 &&
                                        restart_position.y()+restart->height()/2==capture_position.y()+capture->height()/2,
                                        "Restart and Retake must share the capture row, with a wide gap before Capture more");
                                auto* tabs=window->findChild<QTabWidget*>("workspaceTabs");
                                if (presented==1) {
                                    require(!window->findChild<CaptureTransition*>()->isRunning(), "Image transfer did not finish");
                                    require(hasGreenCamera(canvas),"First FS camera did not turn green after landing");
                                    screenshot("landed");
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
                    phase=5; window->buildMeshGPU(); // Function remains available without a button.
                } else {
                    const auto points_depth=cv::imread(point_export.filePath("depth.tiff").toStdString(),cv::IMREAD_UNCHANGED);
                    const auto mesh_depth=cv::imread(mesh_export.filePath("depth.tiff").toStdString(),cv::IMREAD_UNCHANGED);
                    require(cv::norm(points_depth,mesh_depth,cv::NORM_INF)==0, "GPU mesh build changed the saved depth");
                    require(captured_views()==1, "Mesh export changed the captured view count");
                    auto* capture=window->findChild<QPushButton*>("capturePair");
                    require(capture->isEnabled() && capture->text()=="Capture more", "Cannot start the next capture");
                    phase=7; ready_ticks=0; capture->click();
                }
                return;
            }
            auto* camera=window->findChild<QPushButton*>("connectCameras");
            auto* capture=window->findChild<QPushButton*>("capturePair");
            auto* retake=window->findChild<QPushButton*>("retake");
            auto* preview=window->findChild<StereoImageView*>("capturePreview");
            if (phase==3 && preview->isProcessing()) {
                require(!window->findChild<CaptureTransition*>()->isRunning(), "Transfer ran during FS computation");
                require(!hasSceneContent(window->findChild<QOpenGLWidget*>()), "First capture rendered intermediate geometry during computation");
            }
            if ((phase==7 || phase==9) && capture->isEnabled()) {
                require(!retake->isEnabled(), "Next-view or retake action did not resume preview");
                require(captured_views()==(phase==7 ? 1 : 2), "Starting preview changed completed views");
                if (++ready_ticks<8) return;
                require(!preview->image().isNull() && !preview->isProcessing(), "Next-view preview is missing or still animating");
                require(hasSceneContent(window->findChild<QOpenGLWidget*>()), "Completed geometry disappeared during live preview");
                ready_ticks=0; ++phase; capture->click();
                require(preview->isProcessing(), "Capture did not enter the calculation state");
                require(!window->findChild<CaptureTransition*>()->isRunning(), "MA transfer started before computation completed");
                return;
            }
            if ((phase==8 || phase==10) && capture->isEnabled()) {
                require(captured_views()==2, "Capture more did not append, or Retake appended instead of replacing");
                require(retake->isVisible() && retake->isEnabled(), "Retake is unavailable for captured images");
                require(window->findChild<QWidget*>("maPointCloud")->isEnabled(), "Two-view MA reconstruction did not complete");
                require(!preview->isProcessing(), "Two-view processing animation did not stop");
                require(fs_captures==(phase==8 ? 1 : 2) && landed==fs_captures+1, "FS point cloud or its transfer did not complete for capture/retake");
                require(window->findChild<QRadioButton*>("fsPointCloud")->isChecked(), "Capture/retake must default to FS even after viewing MA");
                require(hasGreenCamera(window->findChild<QOpenGLWidget*>()),"Current camera did not turn green after landing");
                screenshot(QString("fs-landed-%1").arg(fs_captures));
                check_previews();
                phase=phase==8 ? 15 : 16;
                window->findChild<QRadioButton*>("maPointCloud")->click();
                return;
            }
            if ((phase==15 || phase==16) && retake->isEnabled() && ma_presented==fs_captures) {
                require(window->findChild<QRadioButton*>("maPointCloud")->isChecked(), "Manual MA selection was not retained");
                require(!window->findChild<QComboBox*>("meshMode")->isEnabled(), "MA point clouds still expose mesh modes");
                check_previews();
                if (phase==15) { phase=9; ready_ticks=0; retake->click(); }
                else { phase=12; ready_ticks=0; capture->click(); }
                return;
            }
            if (phase==12 && capture->isEnabled()) {
                require(!retake->isEnabled(), "Additional MA view did not resume preview");
                if (++ready_ticks<8) return;
                expected_views=captured_views()+1; ready_ticks=0; phase=13; capture->click();
                return;
            }
            if (phase==13 && retake->isEnabled()) {
                require(captured_views()==expected_views && window->findChild<QRadioButton*>("fsPointCloud")->isChecked(),
                        "Additional capture did not default to FS point clouds");
                require(landed==fs_captures+1,"Additional capture skipped its landing animation");
                require(hasGreenCamera(window->findChild<QOpenGLWidget*>()),"Additional capture lost the green current camera");
                check_previews();
                phase=17; window->findChild<QRadioButton*>("maPointCloud")->click();
                return;
            }
            if (phase==17 && retake->isEnabled() && ma_presented==fs_captures) {
                require(latest_ma_clouds && latest_ma_clouds->size()==size_t(expected_views),
                        "Manual MA selection did not display all current views");
                require(!window->findChild<QComboBox*>("meshMode")->isEnabled(), "MA point clouds still expose mesh modes");
                check_previews();
                if (expected_views<PipelineState::kMaxCaptures) { phase=12; ready_ticks=0; capture->click(); }
                else {
                    require(!capture->isEnabled(), "Five-view capture limit was lost");
                    screenshot("ma-five-landed");
                    window->findChild<QPushButton*>("openSaveResults")->click();
                    auto* dialog=window->findChild<QDialog*>("saveResultsDialog");
                    dialog->findChild<QLineEdit*>("saveDirectory")->setText(ma_export.path());
                    require(dialog->findChild<QPushButton*>("saveSelected")->isEnabled(), "Cannot export the displayed MA point clouds");
                    phase=14; dialog->findChild<QPushButton*>("saveSelected")->click(); dialog->close();
                }
                return;
            }
            if (phase==14 && retake->isEnabled()) {
                require(window->findChild<QLabel*>("status")->text().startsWith("Saved "), "MA point cloud export failed");
                GPUMeshFrame total; total.point_cloud=true;
                for (const auto& cloud:*latest_ma_clouds) total.stats.point_count+=cloud->stats.point_count;
                checkGeometryExport(ma_export.path(),total,true);
                QFile ply(ma_export.filePath("mesh.ply")); require(ply.open(QIODevice::ReadOnly),"Cannot read MA PLY");
                require(ply.read(256).contains("MA world frame"),"MA export did not use world coordinates");
                phase=11; window->findChild<QPushButton*>("restartCaptures")->click(); return;
            }
            if (phase==11 && capture->isEnabled()) {
                require(captured_views()==0 && capture->text()=="Capture pair" && !retake->isVisible(),
                        "Restart did not restore the empty capture toolbar");
                require(!window->findChild<QWidget*>("maPointCloud")->isEnabled(), "Restart retained MA geometry");
                require(!preview->isProcessing(), "Restart retained the processing animation");
                require(!hasSceneContent(window->findChild<QOpenGLWidget*>()), "Restart retained geometry or projected the live preview in 3D");
                const auto live_image=preview->image();
                window->findChild<MeshView*>()->cameraClicked(preview_ids.front());
                require(preview->image()==live_image,"Restart retained a selectable captured preview");
                finish({}); return;
            }
            const bool connected=camera->isEnabled() && camera->text()=="Disconnect cameras" && capture->isEnabled();
            if ((phase==0 || phase==2) && connected) {
                require(capture->text()=="Capture pair" && !retake->isVisible(), "Initial toolbar shows Retake or Capture more without an image");
                if (++ready_ticks<8) return; // Allow fresh camera frames to arrive.
                require(!preview->image().isNull() && !preview->isProcessing(), "Live camera frames are missing from the left preview");
                require(!hasSceneContent(window->findChild<QOpenGLWidget*>()), "Live preview still renders projected geometry in 3D");
                ready_ticks=0;
                if (phase==0) { camera->click(); phase=1; }
                else {
                    capture->click(); phase=3;
                    require(preview->isProcessing(), "Capture did not enter the calculation state");
                    require(!window->findChild<CaptureTransition*>()->isRunning(), "Transfer started before calculation");
                    screenshot("calculating");
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
    std::cout<<"PASS: D435 first FS mesh, default FS points for 2–5 views and retake, manual MA selection, camera transfers, restart, region updates, GPU wireframe, FS/MA PLY export and clean shutdown\n";
    std::cout<<"PASS: camera selection restores each original stereo pair and depth in FS/MA modes, replaces retaken images and clears on restart\n";
    return 0;
}
}

int main(int argc, char** argv) {
    const bool live=argc==2 && std::string(argv[1])=="--live";
    const bool imported=argc==4 && std::string(argv[1])=="--import";
    if (!live) qputenv("QT_QPA_PLATFORM","offscreen");
    if (!live) QApplication::setAttribute(Qt::AA_DontUseNativeDialogs);
    QSurfaceFormat::setDefaultFormat(MeshView::surfaceFormat());
    QApplication app(argc,argv);
    cv::setNumThreads(4);
    try {
        if (live) return checkLiveScan(app);
        if (imported) return checkImportedScan(QString::fromLocal8Bit(argv[2]),QString::fromLocal8Bit(argv[3]));
        checkCapturePresentation();
        checkUncalibratedPipeline();
        checkImportFolder();
        checkRectificationCache();
        checkImportControls();
        return 0;
    } catch (const std::exception& error) {
        std::cerr<<error.what()<<'\n'; return 1;
    }
}
