#pragma once
#include <QImage>
#include <QPolygonF>
#include <QWidget>

// Polygon vertices are always in full-resolution rectified-left pixel coordinates.
class MaskEditor : public QWidget {
    Q_OBJECT
public:
    explicit MaskEditor(QWidget* parent = nullptr);
    void setImage(QImage image);
    void startPolygon();
    void clearPolygon();
    void finishPolygon();
    void undoVertex();
    QImage mask() const;
    int vertexCount() const { return polygon_.size(); }
    bool hasSelection() const { return closed_; }
    QSize imageSize() const { return image_.size(); }
    const QPolygonF& polygon() const { return polygon_; }
    void setEditingEnabled(bool enabled);
    QSize minimumSizeHint() const override { return {260, 240}; }
public slots:
    void setOverlayVisible(bool enabled);
signals:
    void selectionChanged();
    void hint(QString message);
protected:
    void paintEvent(QPaintEvent*) override;
    void mousePressEvent(QMouseEvent*) override;
    void mouseMoveEvent(QMouseEvent*) override;
    void mouseReleaseEvent(QMouseEvent*) override;
    void keyPressEvent(QKeyEvent*) override;
private:
    QRectF imageRect() const;
    QPointF toImage(QPointF position) const;
    QPointF toWidget(QPointF position) const;
    bool validPolygon() const;
    QImage image_;
    QPolygonF polygon_, before_drag_;
    bool closed_{false}, drawing_{false}, overlay_{true}, editing_{true};
    int dragging_{-1};
};
