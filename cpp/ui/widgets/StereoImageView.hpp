#pragma once
#include <QWidget>
#include <QImage>

class StereoImageView : public QWidget {
public:
    explicit StereoImageView(QString title, QWidget* parent = nullptr);
    void setImage(QImage image);
    void setEmptyText(QString text) { empty_text_ = std::move(text); update(); }
    void setEpilines(bool enabled) { epilines_ = enabled; update(); }
protected:
    void paintEvent(QPaintEvent*) override;
private:
    QString title_;
    QImage image_;
    QString empty_text_{"Waiting for camera\nConnect to start the stereo preview"};
    bool epilines_{false};
};
