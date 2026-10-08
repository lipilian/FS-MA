#include "CaptureTransition.hpp"
#include <QPainter>

CaptureTransition::CaptureTransition(QWidget* parent) : QWidget(parent) {
    setObjectName("captureTransition");
    setAttribute(Qt::WA_TransparentForMouseEvents);
    setAttribute(Qt::WA_NoSystemBackground);
    hide();
    animation_.setDuration(900);
    animation_.setStartValue(0.0); animation_.setEndValue(1.0);
    animation_.setEasingCurve(QEasingCurve::InOutCubic);
    connect(&animation_, &QVariantAnimation::valueChanged, this, [this](const QVariant& value) {
        if (!updateQuad(value.toReal())) cancel();
        else update();
    });
    connect(&animation_, &QVariantAnimation::finished, this, &CaptureTransition::finish);
}
bool CaptureTransition::start(QImage image, std::function<QPolygonF()> source,
                              std::function<QPolygonF()> destination) {
    cancel();
    if (image.isNull()) return false;
    image_ = std::move(image); source_ = std::move(source); destination_ = std::move(destination);
    if (!updateQuad(0)) { image_ = {}; source_ = {}; destination_ = {}; return false; }
    show(); raise(); animation_.start();
    return isRunning();
}
bool CaptureTransition::updateQuad(qreal progress) {
    if (!parentWidget() || !source_ || !destination_) return false;
    const auto from = source_(), to = destination_();
    if (from.size() != 4 || to.size() != 4) return false;
    setGeometry(parentWidget()->rect());
    quad_.clear();
    for (int i = 0; i < 4; ++i) quad_ << from[i] * (1 - progress) + to[i] * progress;
    return true;
}
void CaptureTransition::cancel() {
    if (!isRunning()) return;
    animation_.stop(); finish();
}
void CaptureTransition::finish() {
    hide(); image_ = {}; source_ = {}; destination_ = {};
    emit finished();
}
void CaptureTransition::paintEvent(QPaintEvent*) {
    if (image_.isNull()) return;
    const QPolygonF source{{0,0}, {qreal(image_.width()),0},
                          {qreal(image_.width()),qreal(image_.height())}, {0,qreal(image_.height())}};
    QTransform transform;
    if (!QTransform::quadToQuad(source, quad_, transform)) return;
    QPainter painter(this);
    painter.setRenderHint(QPainter::SmoothPixmapTransform);
    painter.setRenderHint(QPainter::Antialiasing);
    painter.setTransform(transform);
    painter.drawImage(QPointF(0,0), image_);
}
