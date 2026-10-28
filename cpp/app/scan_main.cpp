#include "DesktopController.hpp"
#include "widgets/MeshView.hpp"
#include <QApplication>
#include <QCommandLineParser>
#include <QMessageBox>
#include <opencv2/core.hpp>

int main(int argc, char** argv) {
    QSurfaceFormat::setDefaultFormat(MeshView::surfaceFormat());
    QApplication app(argc, argv);
    app.setApplicationName("scan_gui"); app.setApplicationVersion("0.1");
    app.setOrganizationName("FS_Engine"); app.setStyle("Fusion");
    QCommandLineParser parser;
    parser.setApplicationDescription("Load FoundationStereo, SAM and MapAnything, then scan with Stereolabs ZED at 1920x1080, center-cropped to 1296x1080 and resized to 960x800.");
    parser.addHelpOption(); parser.addVersionOption(); parser.process(app);
    try {
        cv::setNumThreads(4);
        DesktopController desktop(DesktopController::StartupMode::ZedScan);
        desktop.show();
        return app.exec();
    } catch (const std::exception& e) {
        QMessageBox::critical(nullptr, "Unable to start ZED scanning", QString::fromUtf8(e.what()));
        return 1;
    }
}
