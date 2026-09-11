#pragma once
#include <QWidget>
#include <QElapsedTimer>
class QLabel;
class QLineEdit;
class QPushButton;
class QProgressBar;

// GUI-only loading screen; FS and SAM construction and allocation stay on the pipeline thread.
class InferenceSplashWindow : public QWidget {
    Q_OBJECT
public:
    explicit InferenceSplashWindow();
    QString enginePath() const;
    QString samEncoderPath() const;
    QString samDecoderPath() const;
    void startLoading();
    void setStatus(const QString& message);
    void showFailure(const QString& message);
    void allowClose();
signals:
    void retryRequested(QString engine_path, QString sam_encoder, QString sam_decoder);
    void closeRequested();
protected:
    void closeEvent(QCloseEvent* event) override;
private:
    QLineEdit *engine_, *sam_encoder_, *sam_decoder_;
    QPushButton *browse_sam_encoder_, *browse_sam_decoder_;
    QLabel *status_, *elapsed_label_;
    QPushButton *browse_, *retry_, *cancel_;
    QProgressBar* progress_;
    QElapsedTimer elapsed_;
    bool loading_{false}, closing_{false}, allow_close_{false};
};
