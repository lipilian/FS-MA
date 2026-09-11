#include "MaskEditor.hpp"
#include <QKeyEvent>
#include <QMouseEvent>
#include <QPainter>
#include <algorithm>
#include <cmath>

MaskEditor::MaskEditor(QWidget* parent) : QWidget(parent) {
    setFocusPolicy(Qt::StrongFocus); setMouseTracking(true); setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Expanding);
}
void MaskEditor::setImage(QImage image) { image_ = std::move(image); clearPolygon(); }
void MaskEditor::setEditingEnabled(bool enabled) {
    if (editing_ == enabled) return;
    if (dragging_ >= 0) { polygon_ = before_drag_; before_drag_.clear(); dragging_ = -1; }
    editing_ = enabled; update();
}
void MaskEditor::setOverlayVisible(bool enabled) { overlay_ = enabled; update(); }
void MaskEditor::clearPolygon() {
    polygon_.clear(); before_drag_.clear(); drawing_ = closed_ = false; dragging_ = -1; update(); emit selectionChanged();
}
void MaskEditor::startPolygon() {
    if (image_.isNull() || !editing_) return;
    clearPolygon(); drawing_ = true; setFocus(); emit hint("Click vertices. Enter or click the first vertex to close. Backspace undoes; Escape cancels.");
}
void MaskEditor::undoVertex() {
    if (!editing_ || !drawing_ || polygon_.empty()) return;
    polygon_.removeLast(); update(); emit selectionChanged();
}
namespace {
double cross(QPointF a, QPointF b, QPointF c) {
    return (b.x()-a.x())*(c.y()-a.y()) - (b.y()-a.y())*(c.x()-a.x());
}
bool onSegment(QPointF a, QPointF b, QPointF p) {
    return std::abs(cross(a,b,p)) < 1e-6 && p.x() >= std::min(a.x(),b.x())-1e-6 &&
        p.x() <= std::max(a.x(),b.x())+1e-6 && p.y() >= std::min(a.y(),b.y())-1e-6 && p.y() <= std::max(a.y(),b.y())+1e-6;
}
bool intersects(QPointF a, QPointF b, QPointF c, QPointF d) {
    return (cross(a,b,c)*cross(a,b,d) < 0 && cross(c,d,a)*cross(c,d,b) < 0) ||
        onSegment(a,b,c) || onSegment(a,b,d) || onSegment(c,d,a) || onSegment(c,d,b);
}
}
bool MaskEditor::validPolygon() const {
    const int n = polygon_.size();
    if (n < 3) return false;
    double area = 0;
    for (int i = 0; i < n; ++i) {
        const auto a = polygon_[i], b = polygon_[(i+1)%n];
        if (QLineF(a,b).length() < 1) return false;
        area += a.x()*b.y() - a.y()*b.x();
        for (int j = i+1; j < n; ++j) {
            if (j == i+1 || (i == 0 && j == n-1)) continue;
            if (intersects(a,b,polygon_[j],polygon_[(j+1)%n])) return false;
        }
    }
    return std::abs(area) >= 2;
}
void MaskEditor::finishPolygon() {
    if (!editing_ || !drawing_) return;
    if (!validPolygon()) { emit hint("Use at least 3 distinct vertices and a non-zero region without crossing edges."); return; }
    closed_ = true; drawing_ = false; update(); emit selectionChanged();
    emit hint("Selection ready. Drag vertices to edit. Geometry / area integration is pending.");
}
QRectF MaskEditor::imageRect() const {
    if (image_.isNull()) return {};
    const QRectF viewport = QRectF(rect()).adjusted(12, 44, -12, -12);
    const QSizeF size = QSizeF(image_.size()).scaled(viewport.size(), Qt::KeepAspectRatio);
    return {viewport.center()-QPointF(size.width()/2, size.height()/2), size};
}
QPointF MaskEditor::toImage(QPointF p) const {
    const auto r = imageRect();
    return {std::clamp((p.x()-r.x())*image_.width()/r.width(), 0.0, double(image_.width()-1)),
            std::clamp((p.y()-r.y())*image_.height()/r.height(), 0.0, double(image_.height()-1))};
}
QPointF MaskEditor::toWidget(QPointF p) const {
    const auto r = imageRect();
    return {r.x()+p.x()*r.width()/image_.width(), r.y()+p.y()*r.height()/image_.height()};
}
void MaskEditor::paintEvent(QPaintEvent*) {
    QPainter p(this); p.setRenderHint(QPainter::Antialiasing); p.fillRect(rect(), QColor("#111c2c"));
    p.setPen(QColor("#d8e4f3")); p.drawText(QRect(16,8,width()-32,28), Qt::AlignVCenter, "RECTIFIED LEFT  /  SELECTION");
    if (image_.isNull()) { p.drawText(rect(), Qt::AlignCenter, "Import or capture a stereo pair\nto edit a region on the rectified left image"); return; }
    p.setRenderHint(QPainter::SmoothPixmapTransform); p.drawImage(imageRect(), image_);
    if (!overlay_) return;
    QPolygonF screen; for (const auto& point : polygon_) screen << toWidget(point);
    p.setPen(QPen(QColor("#54e5bd"), 2));
    if (closed_) { p.setBrush(QColor(52,211,153,65)); p.drawPolygon(screen); }
    else p.drawPolyline(screen);
    p.setBrush(QColor("#e6fff6"));
    for (const auto& point : screen) p.drawEllipse(point, 4, 4);
}
void MaskEditor::mousePressEvent(QMouseEvent* event) {
    if (!editing_ || image_.isNull() || !overlay_) return;
    setFocus();
    if (event->button() == Qt::RightButton) { finishPolygon(); return; }
    if (event->button() != Qt::LeftButton || !imageRect().contains(event->position())) return;
    if (drawing_) {
        if (polygon_.size() >= 3 && QLineF(event->position(),toWidget(polygon_.first())).length() < 10) { finishPolygon(); return; }
        polygon_ << toImage(event->position()); update(); emit selectionChanged();
    } else if (closed_) {
        for (int i = 0; i < polygon_.size(); ++i)
            if (QLineF(event->position(),toWidget(polygon_[i])).length() <= 10) { dragging_ = i; before_drag_ = polygon_; break; }
    }
}
void MaskEditor::mouseMoveEvent(QMouseEvent* event) {
    if (dragging_ < 0 || !editing_) return;
    polygon_[dragging_] = toImage(event->position()); update();
}
void MaskEditor::mouseReleaseEvent(QMouseEvent*) {
    if (dragging_ < 0) return;
    const bool invalid = !validPolygon();
    if (invalid) polygon_ = before_drag_;
    dragging_ = -1; before_drag_.clear(); update(); emit selectionChanged();
    if (invalid) emit hint("Edit reverted: the region must not cross itself or collapse.");
}
void MaskEditor::keyPressEvent(QKeyEvent* event) {
    if (!editing_) return;
    if (event->key() == Qt::Key_Return || event->key() == Qt::Key_Enter) finishPolygon();
    else if (event->key() == Qt::Key_Backspace) undoVertex();
    else if (event->key() == Qt::Key_Escape && drawing_) clearPolygon();
    else QWidget::keyPressEvent(event);
}
QImage MaskEditor::mask() const {
    if (!closed_ || image_.isNull()) return {};
    QImage result(image_.size(), QImage::Format_RGB32); result.fill(Qt::black);
    QPainter p(&result); p.setPen(Qt::NoPen); p.setBrush(Qt::white); p.drawPolygon(polygon_); p.end();
    return result.convertToFormat(QImage::Format_Grayscale8);
}
