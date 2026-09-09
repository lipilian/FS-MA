#include "StereoImageView.hpp"
#include <QPainter>
StereoImageView::StereoImageView(QString title, QWidget* parent) : QWidget(parent), title_(std::move(title)) {
    setMinimumSize(220, 240); setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Expanding);
}
void StereoImageView::setImage(QImage image) { image_ = std::move(image); update(); }
void StereoImageView::paintEvent(QPaintEvent*) {
    QPainter p(this); p.setRenderHint(QPainter::Antialiasing);
    p.fillRect(rect(), QColor("#111c2c"));
    p.setPen(QColor("#d8e4f3"));
    p.drawText(QRect(16, 8, width() - 32, 28), Qt::AlignVCenter, title_);
    const QRect viewport(8, 44, width() - 16, height() - 52);
    if (image_.isNull()) {
        p.setPen(QColor("#74869e"));
        p.drawText(viewport, Qt::AlignCenter, "Waiting for camera\nConnect to start the stereo preview");
    } else {
        const QSize size = image_.size().scaled(viewport.size(), Qt::KeepAspectRatio);
        const QRect target(viewport.center() - QPoint(size.width()/2, size.height()/2), size);
        p.setRenderHint(QPainter::SmoothPixmapTransform); p.drawImage(target, image_);
    }
}
