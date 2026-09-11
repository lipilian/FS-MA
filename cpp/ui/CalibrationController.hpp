#pragma once
#include "CalibrationWorker.hpp"
#include <QThread>

class CalibrationController : public QObject {
    Q_OBJECT
public:
    explicit CalibrationController(QObject* parent = nullptr, CalibrationWorker::SourceFactory factory = {});
    ~CalibrationController() override;
    void submit(std::function<void(CalibrationWorker&)> action);
    void stop();
signals:
    void stateChanged(CalibrationState state);
    void preview(QImage left, QImage right, QString caption);
    void failed(QString message);
    void actionFinished();
    void completed(ConfirmedCalibration result, QString path, SharedStereoSource source);
    void stopped();
private:
    QThread thread_;
    CalibrationWorker* worker_;
    bool stopping_{false};
};
