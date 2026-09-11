#include "InferenceSplashWindow.hpp"
#include <QCloseEvent>
#include <QCoreApplication>
#include <QDir>
#include <QFileDialog>
#include <QFileInfo>
#include <QHBoxLayout>
#include <QLabel>
#include <QLineEdit>
#include <QProgressBar>
#include <QPushButton>
#include <QTimer>
#include <QVBoxLayout>

namespace {
QString defaultEngine(const QString& relative = "onnx/foundationstereo_800x960_gwc_plugin.engine") {
    for (const auto& base : {QDir::currentPath(), QCoreApplication::applicationDirPath() + "/../.."}) {
        const QFileInfo candidate(QDir(base).filePath(relative));
        if (candidate.isFile()) return candidate.absoluteFilePath();
    }
    return QDir::current().absoluteFilePath(relative);
}
QLabel* label(const QString& text) {
    auto* result = new QLabel(text); result->setWordWrap(true); result->setTextFormat(Qt::PlainText); return result;
}
}
InferenceSplashWindow::InferenceSplashWindow() : QWidget(nullptr, Qt::Dialog) {
    setObjectName("inferenceSplashWindow"); setWindowTitle("FoundationStereo · Preparing reconstruction");
    resize(760, 520); setMinimumSize(640, 460);
    setStyleSheet(R"(
        QWidget#inferenceSplashWindow { background: #f3f6fa; }
        QWidget { color: #20314a; font-family: 'Noto Sans', sans-serif; font-size: 13px; }
        QLabel#step { color: #2464d9; font-size: 11px; font-weight: 700; }
        QLabel#title { font-size: 25px; font-weight: 700; }
        QLabel#loadingStatus { background: #e5edfa; color: #234772; border-radius: 6px; padding: 14px; }
        QLineEdit { background: white; border: 1px solid #ced8e5; border-radius: 4px; padding: 7px; }
        QPushButton { background: white; border: 1px solid #cbd6e5; border-radius: 5px; padding: 8px 14px; }
        QPushButton:hover { background: #eaf1fc; }
        QPushButton:disabled { color: #9aa8b9; background: #edf1f6; }
        QPushButton#retryInitialization { background: #2464d9; color: white; }
        QProgressBar { border: 0; background: #e0e8f5; min-height: 7px; max-height: 7px; }
        QProgressBar::chunk { background: #2464d9; }
    )");
    auto* layout = new QVBoxLayout(this); layout->setContentsMargins(28,24,28,24); layout->setSpacing(14);
    auto* step = label("CALIBRATION  ✓     →     PREPARING INFERENCE     →     RECONSTRUCTION"); step->setObjectName("step"); layout->addWidget(step);
    auto* title = label("Preparing FoundationStereo + SAM 2.1"); title->setObjectName("title"); layout->addWidget(title);
    layout->addWidget(label("The reconstruction workspace will open when both models and their inference buffers are ready."));
    auto* path = new QHBoxLayout;
    engine_ = new QLineEdit(defaultEngine()); engine_->setObjectName("splashEnginePath"); path->addWidget(engine_,1);
    browse_ = new QPushButton("Choose engine…"); browse_->setObjectName("splashBrowseEngine"); path->addWidget(browse_); layout->addLayout(path);
    const auto samRow = [&](const QString& title, const QString& relative, const char* object_name, QLineEdit*& field, QPushButton*& browse) {
        layout->addWidget(label(title)); auto* row = new QHBoxLayout;
        field = new QLineEdit(defaultEngine(relative)); field->setObjectName(object_name); row->addWidget(field,1);
        browse = new QPushButton("Choose engine…"); row->addWidget(browse); layout->addLayout(row);
        connect(browse,&QPushButton::clicked,this,[this,field] {
            const auto path=QFileDialog::getOpenFileName(this,"Choose SAM TensorRT engine",field->text(),"TensorRT engine (*.engine *.plan)");
            if (!path.isEmpty()) field->setText(path);
        });
    };
    samRow("SAM 2.1 encoder","onnx/sam2.1_hiera_large.encoder.engine","splashSamEncoder",sam_encoder_,browse_sam_encoder_);
    samRow("SAM 2.1 decoder","onnx/sam2.1_hiera_large.decoder.engine","splashSamDecoder",sam_decoder_,browse_sam_decoder_);
    status_ = label("Waiting to initialize…"); status_->setObjectName("loadingStatus"); status_->setMinimumHeight(72); layout->addWidget(status_,1);
    progress_ = new QProgressBar; progress_->setTextVisible(false); layout->addWidget(progress_);
    auto* footer = new QHBoxLayout;
    elapsed_label_ = label("Starting…"); footer->addWidget(elapsed_label_,1);
    retry_ = new QPushButton("Retry initialization"); retry_->setObjectName("retryInitialization"); retry_->hide(); footer->addWidget(retry_);
    cancel_ = new QPushButton("Cancel and exit"); cancel_->setObjectName("cancelInitialization"); footer->addWidget(cancel_); layout->addLayout(footer);
    connect(cancel_, &QPushButton::clicked, this, &QWidget::close);
    connect(retry_, &QPushButton::clicked, this, [this] { emit retryRequested(enginePath(),samEncoderPath(),samDecoderPath()); });
    connect(browse_, &QPushButton::clicked, this, [this] {
        const auto path = QFileDialog::getOpenFileName(this,"Choose TensorRT engine",enginePath(),"TensorRT engine (*.engine *.plan);;All files (*)");
        if (!path.isEmpty()) engine_->setText(path);
    });
    auto* timer = new QTimer(this); timer->setInterval(200);
    connect(timer, &QTimer::timeout, this, [this] {
        if (loading_) elapsed_label_->setText(QString("Elapsed: %1 s").arg(elapsed_.elapsed()/1000.0,0,'f',1));
    }); timer->start();
}
QString InferenceSplashWindow::enginePath() const { return engine_->text().trimmed(); }
QString InferenceSplashWindow::samEncoderPath() const { return sam_encoder_->text().trimmed(); }
QString InferenceSplashWindow::samDecoderPath() const { return sam_decoder_->text().trimmed(); }
void InferenceSplashWindow::startLoading() {
    for (QWidget* w : std::initializer_list<QWidget*>{sam_encoder_,sam_decoder_,browse_sam_encoder_,browse_sam_decoder_}) w->setEnabled(false);
    loading_ = true; elapsed_.start(); engine_->setEnabled(false); browse_->setEnabled(false); retry_->hide();
    progress_->setRange(0,0); status_->setText("Creating FS and preparing the inference engine…"); elapsed_label_->setText("Starting…");
}
void InferenceSplashWindow::setStatus(const QString& message) { if (!closing_) status_->setText(message); }
void InferenceSplashWindow::showFailure(const QString& message) {
    if (closing_) return;
    loading_ = false; progress_->setRange(0,1); progress_->setValue(0);
    status_->setText("Initialization failed\n" + message);
    elapsed_label_->setText("Choose a compatible engine and retry, or exit.");
    engine_->setEnabled(true); browse_->setEnabled(true); retry_->show();
    for (QWidget* w : std::initializer_list<QWidget*>{sam_encoder_,sam_decoder_,browse_sam_encoder_,browse_sam_decoder_}) w->setEnabled(true);
}
void InferenceSplashWindow::closeEvent(QCloseEvent* event) {
    if (allow_close_) { event->accept(); return; }
    event->ignore(); if (closing_) return;
    for (QWidget* w : std::initializer_list<QWidget*>{sam_encoder_,sam_decoder_,browse_sam_encoder_,browse_sam_decoder_}) w->setEnabled(false);
    closing_ = true; engine_->setEnabled(false); browse_->setEnabled(false); retry_->setEnabled(false); cancel_->setEnabled(false);
    status_->setText("Closing · waiting for initialization to finish safely and releasing GPU / cameras…"); emit closeRequested();
}
void InferenceSplashWindow::allowClose() { allow_close_ = true; close(); }
