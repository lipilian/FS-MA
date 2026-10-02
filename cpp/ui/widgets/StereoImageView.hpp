#pragma once
#include <QWidget>
#include <QImage>

class StereoImageView : public QWidget {
public:
    explicit StereoImageView(QString title, QWidget* parent = nullptr);
    void setImage(QImage image);
protected:
    void paintEvent(QPaintEvent*) override;
private:
    QString title_;
    QImage image_;
};
