#pragma once
#include <QWidget>
#include <QImage>

class StereoImageView : public QWidget {
    Q_OBJECT
public:
    explicit StereoImageView(QString title, QWidget* parent = nullptr);
    void setImage(QImage image);
    void setProcessing(bool processing);
    bool isProcessing() const { return processing_; }
    const QImage& image() const { return image_; }
    QRect imageRect() const;
    void setImageVisible(bool visible);
protected:
    void paintEvent(QPaintEvent*) override;
private:
    QString title_;
    QImage image_;
    bool processing_{false}, image_visible_{true};
};
