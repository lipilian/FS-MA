#include "DesktopController.hpp"
#include <QApplication>
#include <QCommandLineParser>
#include <QMessageBox>
#include <opencv2/core.hpp>
int main(int argc, char** argv) {
    QApplication app(argc, argv);
    app.setApplicationName("fs_gui"); app.setApplicationVersion("0.1");
    app.setOrganizationName("FS_Engine"); app.setStyle("Fusion");
    QCommandLineParser parser;
    parser.setApplicationDescription("FoundationStereo Qt 6 stereo calibration window");
    parser.addHelpOption(); parser.addVersionOption(); parser.process(app);
    try {
        // Avoid oversubscribing the desktop during full-resolution detection.
        cv::setNumThreads(4);
        DesktopController desktop; desktop.show();
        return app.exec();
    } catch (const std::exception& e) {
        QMessageBox::critical(nullptr, "Unable to start calibration", QString::fromUtf8(e.what()));
        return 1;
    }
}
