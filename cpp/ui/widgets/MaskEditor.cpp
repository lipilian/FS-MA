#include "MaskEditor.hpp"
#include <QKeyEvent>
#include <QMouseEvent>
#include <QWheelEvent>
#include <QPainter>
#include <algorithm>
#include <cmath>
#include <opencv2/imgproc.hpp>

MaskEditor::MaskEditor(QWidget* parent) : QWidget(parent) {
    setFocusPolicy(Qt::StrongFocus); setMouseTracking(true); setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Expanding);
}
void MaskEditor::setImage(QImage image) {
    image_ = std::move(image); zoom_=1.0; view_center_={0.5,0.5}; wheel_delta_=0; clearMask();
}
void MaskEditor::setEditingEnabled(bool enabled) {
    if (editing_ == enabled) return;
    if (brushing_) { brushing_=false; before_stroke_={}; }
    if (point_drag_ >= 0) { points_[point_drag_].position=point_before_drag_; point_drag_=-1; }
    if (boxing_) { box_=previous_box_; boxing_=false; }
    editing_ = enabled; if (!enabled) cursor_.reset(); update();
}
void MaskEditor::setOverlayVisible(bool enabled) { overlay_ = enabled; update(); }
void MaskEditor::clearMask() {
    base_mask_={}; corrections_={}; before_stroke_={}; brushing_=false; cursor_.reset();
    points_.clear(); box_.reset(); previous_box_.reset(); boxing_=false; point_drag_=-1;
    sam_mask_={}; sam_overlay_={}; accepted_=false;
    update(); emit selectionChanged(); emit promptsChanged();
}
QRectF MaskEditor::imageViewport() const {
    return {12,44,qreal(std::max(1,width()-24)),qreal(std::max(1,height()-56))};
}
QRectF MaskEditor::imageRect() const {
    if (image_.isNull()) return {};
    const QRectF viewport = imageViewport();
    const QSizeF size = QSizeF(image_.size()).scaled(viewport.size(), Qt::KeepAspectRatio)*zoom_;
    return {viewport.center()-QPointF(view_center_.x()*size.width(),view_center_.y()*size.height()),size};
}
QPointF MaskEditor::toImage(QPointF p) const {
    const auto viewport=imageViewport();
    p.setX(std::clamp(p.x(),viewport.left(),viewport.right()));
    p.setY(std::clamp(p.y(),viewport.top(),viewport.bottom()));
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
    p.drawText(QRect(16,8,width()-32,28),Qt::AlignRight|Qt::AlignVCenter,QString::number(qRound(zoom_*100))+"%");
    p.setClipRect(imageViewport());
    p.setRenderHint(QPainter::SmoothPixmapTransform); p.drawImage(imageRect(), image_);
    if (!overlay_) return;
    if (!sam_overlay_.isNull()) p.drawImage(imageRect(),sam_overlay_);
    if (box_) {
        p.setPen(QPen(QColor("#facc15"),2,boxing_ ? Qt::DashLine : Qt::SolidLine)); p.setBrush(Qt::NoBrush);
        p.drawRect(QRectF(toWidget(box_->topLeft()),toWidget(box_->bottomRight())));
    }
    for (const auto& point : points_) {
        const QPointF pos=toWidget(point.position);
        p.setPen(QPen(Qt::white,1)); p.setBrush(point.label ? QColor("#22c55e") : QColor("#ef4444")); p.drawEllipse(pos,6,6);
        p.setPen(QPen(Qt::white,2)); p.drawLine(pos+QPointF(-3,0),pos+QPointF(3,0));
        if (point.label) p.drawLine(pos+QPointF(0,-3),pos+QPointF(0,3));
    }
    if (brushTool() && cursor_ && editing_) {
        const qreal radius=brush_size_*imageRect().width()/image_.width()/2;
        p.setBrush(Qt::NoBrush); p.setPen(QPen(Qt::black,3)); p.drawEllipse(*cursor_,radius,radius);
        p.setPen(QPen(tool_==Tool::Brush ? QColor("#86efac") : QColor("#fca5a5"),1));
        p.drawEllipse(*cursor_,radius,radius);
    }
}
void MaskEditor::mousePressEvent(QMouseEvent* event) {
    if (!editing_ || image_.isNull() || !overlay_) return;
    setFocus();
    if (!imageViewport().contains(event->position()) || !imageRect().contains(event->position())) return;
    if (brushTool()) {
        if (event->button()!=Qt::LeftButton) return;
        before_stroke_=corrections_; accepted_before_stroke_=accepted_;
        brushing_=true; stroke_last_=toImage(event->position()); cursor_=event->position();
        paintStroke(stroke_last_,stroke_last_); return;
    }
    if (event->button()==Qt::RightButton || (event->button()==Qt::LeftButton && tool_==Tool::Remove)) { removePromptAt(event->position()); return; }
    if (event->button()!=Qt::LeftButton) return;
    if (tool_==Tool::Box) {
        if (points_.size()+2>fs::SamSegmenter::kMaxPrompts) { emit hint("The prompt limit is 64; a box uses two points."); return; }
        previous_box_=box_; box_start_=toImage(event->position()); boxing_=true; box_=QRectF(box_start_,box_start_); update(); return;
    }
    for (size_t i=0;i<points_.size();++i) if (QLineF(event->position(),toWidget(points_[i].position)).length()<=10) {
        point_drag_=int(i); point_before_drag_=points_[i].position; return;
    }
    if (prompts().size()>=fs::SamSegmenter::kMaxPrompts) { emit hint("The prompt limit is 64; remove a point first."); return; }
    points_.push_back({toImage(event->position()),tool_==Tool::Foreground ? 1 : 0}); promptsEdited();
}
void MaskEditor::mouseMoveEvent(QMouseEvent* event) {
    if (!editing_ || image_.isNull()) return;
    cursor_=imageViewport().contains(event->position()) && imageRect().contains(event->position())
        ? std::optional<QPointF>(event->position()) : std::nullopt;
    if (brushing_) {
        const auto next=toImage(event->position()); paintStroke(stroke_last_,next); stroke_last_=next; return;
    }
    if (boxing_) { box_=QRectF(box_start_,toImage(event->position())).normalized(); update(); return; }
    if (point_drag_>=0) { points_[point_drag_].position=toImage(event->position()); update(); return; }
    if (brushTool()) update();
}
void MaskEditor::mouseReleaseEvent(QMouseEvent* event) {
    if (!editing_ || event->button()!=Qt::LeftButton) return;
    if (brushing_) {
        paintStroke(stroke_last_,toImage(event->position())); brushing_=false; before_stroke_={};
        emit selectionChanged(); emit hint("Mask edited. Finish draw to confirm; brush edits are retained when SAM prompts change."); return;
    }
    if (boxing_) {
        box_=QRectF(box_start_,toImage(event->position())).normalized(); boxing_=false;
        if (box_->width()<1 || box_->height()<1) { box_=previous_box_; update(); emit hint("Drag a rectangle with non-zero width and height."); }
        else promptsEdited();
        previous_box_.reset(); return;
    }
    if (point_drag_>=0) { points_[point_drag_].position=toImage(event->position()); point_drag_=-1; promptsEdited(); return; }
}
void MaskEditor::keyPressEvent(QKeyEvent* event) {
    if (!editing_) return;
    if (event->key()==Qt::Key_Escape) {
        if (brushing_) {
            corrections_=before_stroke_; before_stroke_={}; brushing_=false; composeMask();
            accepted_=accepted_before_stroke_ && hasPrediction(); emit selectionChanged();
        }
        else if (boxing_) { box_=previous_box_; previous_box_.reset(); boxing_=false; }
        else if (point_drag_>=0) { points_[point_drag_].position=point_before_drag_; point_drag_=-1; }
        update(); return;
    }
    if (event->key()==Qt::Key_Backspace || event->key()==Qt::Key_Delete) { undoPrompt(); return; }
    if (event->key()==Qt::Key_Return || event->key()==Qt::Key_Enter) { acceptPrediction(); return; }
    QWidget::keyPressEvent(event);
}
void MaskEditor::leaveEvent(QEvent* event) { wheel_delta_=0; cursor_.reset(); update(); QWidget::leaveEvent(event); }

QImage MaskEditor::mask() const {
    return accepted_ ? sam_mask_.copy() : QImage();
}

void MaskEditor::setTool(Tool tool) {
    if (!editing_ || image_.isNull()) return;
    wheel_delta_=0; tool_=tool; overlay_=true; setFocus(); update();
    emit hint(brushTool() ? "Drag to paint or erase. Wheel: zoom at cursor (up to 5x). Shift+wheel: brush size in original-image pixels. SAM updates retain your edits."
        : tool==Tool::Box ? "Drag a box around the target. Then add foreground / background points to refine it."
        : tool==Tool::Remove ? "Click a prompt point or box edge to remove it."
        : "Click to add a prompt; drag existing points to move them. Right-click removes a prompt; Backspace undoes.");
}
std::vector<fs::SamPrompt> MaskEditor::prompts() const {
    std::vector<fs::SamPrompt> result;
    if (box_) {
        result.push_back({{float(box_->left()),float(box_->top())},2});
        result.push_back({{float(box_->right()),float(box_->bottom())},3});
    }
    for(const auto& p : points_) result.push_back({{float(p.position.x()),float(p.position.y())},p.label});
    return result;
}
void MaskEditor::promptsEdited() {
    base_mask_={}; accepted_=false; composeMask();
    update(); emit selectionChanged(); emit promptsChanged();
}
void MaskEditor::setPrediction(QImage mask) {
    if (!mask.isNull() && mask.size()!=image_.size()) return;
    accepted_=false; if (brushing_) accepted_before_stroke_=false;
    base_mask_=hasPrompts() ? mask.convertToFormat(QImage::Format_Grayscale8) : QImage();
    composeMask(); update(); emit selectionChanged();
}
bool MaskEditor::brushTool() const { return tool_==Tool::Brush || tool_==Tool::Eraser; }
void MaskEditor::setBrushSize(int diameter) {
    diameter=std::clamp(diameter,1,100);
    if (brush_size_==diameter) return;
    brush_size_=diameter; update(); emit brushSizeChanged(brush_size_);
}
void MaskEditor::wheelEvent(QWheelEvent* event) {
    if (image_.isNull() || !imageViewport().contains(event->position()) || !imageRect().contains(event->position())) {
        wheel_delta_=0; event->ignore(); return;
    }
    if (event->modifiers().testFlag(Qt::ShiftModifier)) {
        if (!editing_ || !overlay_ || !brushTool()) { wheel_delta_=0; event->ignore(); return; }
        // Accumulate partial wheel steps. One notch changes one original-image pixel.
        wheel_delta_+=event->angleDelta().y();
        const int steps=wheel_delta_/120; wheel_delta_%=120;
        cursor_=event->position(); setBrushSize(brush_size_+steps); update(); event->accept(); return;
    }
    wheel_delta_=0;
    event->accept();
    // Do not move the image underneath an unfinished drawing gesture.
    if (brushing_ || boxing_ || point_drag_>=0) return;
    const qreal steps=event->pixelDelta().isNull() ? event->angleDelta().y()/120.0 : event->pixelDelta().y()/120.0;
    const qreal next_zoom=std::clamp(zoom_*std::pow(1.2,steps),1.0,5.0);
    if (next_zoom==zoom_) return;
    const QRectF before=imageRect();
    const QPointF anchor=event->position();
    const QSizeF size=before.size()*(next_zoom/zoom_);
    // Preserve the image point beneath the cursor while changing the scale.
    view_center_={(anchor.x()-before.left())/before.width()+(imageViewport().center().x()-anchor.x())/size.width(),
                  (anchor.y()-before.top())/before.height()+(imageViewport().center().y()-anchor.y())/size.height()};
    zoom_=next_zoom;
    if (zoom_==1.0) view_center_={0.5,0.5};
    cursor_=anchor; update();
}
void MaskEditor::composeMask() {
    sam_mask_={}; sam_overlay_={};
    if (image_.isNull() || (base_mask_.isNull() && corrections_.isNull())) return;
    sam_mask_=base_mask_.isNull() ? QImage(image_.size(),QImage::Format_Grayscale8) : base_mask_.copy();
    if (base_mask_.isNull()) sam_mask_.fill(0);
    // 128 means untouched; 0 and 255 are persistent manual overrides of SAM pixels.
    if (!corrections_.isNull()) for (int y=0;y<sam_mask_.height();++y) {
        auto* output=sam_mask_.scanLine(y); const auto* edits=corrections_.constScanLine(y);
        for (int x=0;x<sam_mask_.width();++x) if (edits[x]!=128) output[x]=edits[x];
    }
    sam_overlay_=QImage(sam_mask_.constBits(),sam_mask_.width(),sam_mask_.height(),sam_mask_.bytesPerLine(),QImage::Format_Indexed8).copy();
    QList<QRgb> colors(256,qRgba(52,211,153,100)); colors[0]=qRgba(0,0,0,0); sam_overlay_.setColorTable(colors);
}
void MaskEditor::paintStroke(QPointF from, QPointF to) {
    if (corrections_.isNull()) { corrections_=QImage(image_.size(),QImage::Format_Grayscale8); corrections_.fill(128); }
    cv::Mat edits(corrections_.height(),corrections_.width(),CV_8UC1,corrections_.bits(),corrections_.bytesPerLine());
    const auto pixel=[](QPointF p) { return cv::Point(qRound(p.x()),qRound(p.y())); };
    // LINE_8 leaves strictly binary pixels, with continuous strokes between mouse events.
    cv::line(edits,pixel(from),pixel(to),cv::Scalar(tool_==Tool::Brush ? 255 : 0),brush_size_,cv::LINE_8);
    accepted_=false; composeMask(); update(); emit selectionChanged();
}
void MaskEditor::acceptPrediction() {
    if (!editing_ || sam_mask_.isNull() || boxing_ || point_drag_>=0 || brushing_) return;
    accepted_=true; update(); emit selectionChanged(); emit hint("Mask selected. Prompts and brush edits remain editable.");
}
void MaskEditor::undoPrompt() {
    if (!editing_ || boxing_ || point_drag_>=0 || brushing_) return;
    if (!points_.empty()) points_.pop_back();
    else if (box_) box_.reset();
    else return;
    promptsEdited();
}
void MaskEditor::removePromptAt(QPointF position) {
    if (boxing_ || point_drag_>=0) return;
    for(size_t i=points_.size();i>0;--i) if (QLineF(position,toWidget(points_[i-1].position)).length()<=10) {
        points_.erase(points_.begin()+i-1); promptsEdited(); return;
    }
    if (box_) {
        const QRectF r(toWidget(box_->topLeft()),toWidget(box_->bottomRight()));
        if (r.adjusted(-8,-8,8,8).contains(position) && !r.adjusted(8,8,-8,-8).contains(position)) { box_.reset(); promptsEdited(); }
    }
}
