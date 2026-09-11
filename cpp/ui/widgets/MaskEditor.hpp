#pragma once
#include <QImage>
#include <QWidget>
#include <optional>
#include "fs/inference/SamSegmenter.hpp"

// Prompts and brush strokes use full-resolution rectified-left pixel coordinates.
class MaskEditor : public QWidget {
    Q_OBJECT
public:
    explicit MaskEditor(QWidget* parent = nullptr);
    enum class Tool { Box, Foreground, Background, Remove, Brush, Eraser };
    void setTool(Tool tool);
    Tool tool() const { return tool_; }
    std::vector<fs::SamPrompt> prompts() const;
    void setPrediction(QImage mask);
    void acceptPrediction();
    void undoPrompt();
    bool hasPrediction() const { return !sam_mask_.isNull(); }
    bool hasPrompts() const { return !points_.empty() || box_.has_value(); }
    bool samSelection() const { return accepted_; }
    void setImage(QImage image);
    void clearMask();
    void setBrushSize(int diameter);
    int brushSize() const { return brush_size_; }
    QImage mask() const;
    bool hasSelection() const { return accepted_; }
    QSize imageSize() const { return image_.size(); }
    void setEditingEnabled(bool enabled);
    QSize minimumSizeHint() const override { return {260, 240}; }
public slots:
    void setOverlayVisible(bool enabled);
signals:
    void selectionChanged();
    void promptsChanged();
    void brushSizeChanged(int diameter);
    void hint(QString message);
protected:
    void paintEvent(QPaintEvent*) override;
    void mousePressEvent(QMouseEvent*) override;
    void mouseMoveEvent(QMouseEvent*) override;
    void mouseReleaseEvent(QMouseEvent*) override;
    void keyPressEvent(QKeyEvent*) override;
    void leaveEvent(QEvent*) override;
    void wheelEvent(QWheelEvent*) override;
private:
    QRectF imageRect() const;
    QPointF toImage(QPointF position) const;
    QPointF toWidget(QPointF position) const;
    bool brushTool() const;
    void paintStroke(QPointF from, QPointF to);
    void composeMask();
    void promptsEdited();
    void removePromptAt(QPointF position);
    struct Point { QPointF position; int label; };
    std::vector<Point> points_;
    std::optional<QRectF> box_, previous_box_;
    QPointF box_start_, point_before_drag_;
    bool boxing_{false}, accepted_{false};
    int point_drag_{-1};
    Tool tool_{Tool::Box};
    QImage sam_mask_, sam_overlay_;
    QImage image_, base_mask_, corrections_, before_stroke_;
    QPointF stroke_last_;
    std::optional<QPointF> cursor_;
    bool overlay_{true}, editing_{true}, brushing_{false}, accepted_before_stroke_{false};
    int brush_size_{12}, wheel_delta_{0};
};
