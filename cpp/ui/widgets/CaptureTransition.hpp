#pragma once
#include <QImage>
#include <QPolygonF>
#include <QVariantAnimation>
#include <QWidget>
#include <functional>

// A screen-space image transfer over the preview and OpenGL siblings. Only the
// four destination corners come from the renderer; mesh data stays on the GPU.
class CaptureTransition : public QWidget {
    Q_OBJECT
public:
    explicit CaptureTransition(QWidget* parent);
    bool start(QImage image, std::function<QPolygonF()> source,
               std::function<QPolygonF()> destination);
    void cancel();
    bool isRunning() const { return animation_.state() == QAbstractAnimation::Running; }
    const QPolygonF& imageQuad() const { return quad_; }
signals:
    void finished();
protected:
    void paintEvent(QPaintEvent*) override;
private:
    bool updateQuad(qreal progress);
    void finish();
    QVariantAnimation animation_;
    QImage image_;
    QPolygonF quad_;
    std::function<QPolygonF()> source_, destination_;
};
