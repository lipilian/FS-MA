#pragma once
#include "CalibrationController.hpp"
#include <QMainWindow>
class QSpinBox;
class QDoubleSpinBox;
class QComboBox;
class QPushButton;
class QCheckBox;
class QLabel;
class QListWidget;
class QProgressBar;
class QGroupBox;
class StereoImageView;

class CalibrationWindow : public QMainWindow {
    Q_OBJECT
public:
    explicit CalibrationWindow(CalibrationController& controller);
    void allowClose();
signals:
    void closeRequested();
protected:
    void closeEvent(QCloseEvent* event) override;
private:
    void updateState(CalibrationState state);
    void refreshActions();
    void dispatch(std::function<void(CalibrationWorker&)> action);
    fs::calibration::BoardConfig boardFromForm() const;
    CalibrationController& controller_;
    CalibrationState state_;
    bool awaiting_{false}, closing_{false}, allow_close_{false};
    QSpinBox *squares_x_, *squares_y_;
    QDoubleSpinBox *square_mm_, *marker_mm_, *exposure_, *threshold_;
    QComboBox* dictionary_;
    QGroupBox* board_group_;
    QPushButton *connect_, *apply_, *capture_, *compute_, *check_, *save_, *load_, *finish_, *cancel_, *remove_, *live_, *reuse_;
    QCheckBox *overlay_, *rectified_;
    QLabel *status_, *quality_, *caption_, *sample_count_, *saved_, *reuse_hint_;
    QListWidget* samples_;
    QProgressBar* progress_;
    StereoImageView *left_, *right_;
};
