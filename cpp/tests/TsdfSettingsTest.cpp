#include "TsdfSettingsDialog.hpp"
#include <QApplication>
#include <QDialogButtonBox>
#include <QDoubleSpinBox>
#include <QKeyEvent>
#include <QLineEdit>
#include <QPushButton>
#include <QTimer>
#include <cmath>
#include <iostream>
#include <stdexcept>

int main(int argc,char** argv) {
    qputenv("QT_QPA_PLATFORM","offscreen");
    QApplication app(argc,argv);
    try {
        const auto require=[](bool ok,const char* message) { if (!ok) throw std::runtime_error(message); };
        TsdfSettingsDialog dialog({});
        auto* voxel=dialog.findChild<QDoubleSpinBox*>("tsdfVoxelSize");
        auto* trunc=dialog.findChild<QDoubleSpinBox*>("tsdfTruncationVoxels");
        auto* distance=dialog.findChild<QDoubleSpinBox*>("tsdfTruncationDistance");
        require(voxel && trunc && distance && voxel->minimum()==.1 && voxel->maximum()==10 && voxel->value()==1 && distance->value()==8,
                "TSDF dialog defaults/range are wrong");
        voxel->setValue(.1);
        require(trunc->value()==8 && distance->value()==.8,"Changing voxel size did not preserve the truncation multiplier");
        distance->setValue(1.2);
        require(trunc->value()==12 && std::abs(dialog.options().truncation_mm()-1.2)<1e-8,"Editing actual distance did not update the multiplier");
        trunc->setValue(4); voxel->setValue(10);
        require(distance->value()==40,"Truncation controls lost synchronization");
        dialog.findChild<QDialogButtonBox*>()->button(QDialogButtonBox::RestoreDefaults)->click();
        require(voxel->value()==1 && trunc->value()==8 && distance->value()==8 && dialog.options().block_count==30000,
                "Restore defaults failed");
        auto* buttons=dialog.findChild<QDialogButtonBox*>();
        require(buttons->button(QDialogButtonBox::Close) && !buttons->button(QDialogButtonBox::Ok) &&
                !buttons->button(QDialogButtonBox::Cancel),"TSDF must apply on close without OK/Cancel");
        for (int action=0;action<3;++action) {
            const double mm=2.5+action;
            QTimer::singleShot(0,[&] {
                distance->setFocus();
                distance->findChild<QLineEdit*>()->setText(QString::number(mm)+" mm");
                if (action==0) dialog.close();
                else if (action==1) buttons->button(QDialogButtonBox::Close)->click();
                else {
                    QKeyEvent escape(QEvent::KeyPress,Qt::Key_Escape,Qt::NoModifier);
                    QApplication::sendEvent(&dialog,&escape);
                }
            });
            require(dialog.exec()==QDialog::Accepted && std::abs(dialog.options().truncation_mm()-mm)<1e-8,
                    "Closing TSDF settings discarded the active text edit");
        }
        std::cout<<"PASS: TSDF defaults/ranges, linked truncation, restore defaults and apply on window/Close/Escape\n";
        return 0;
    } catch (const std::exception& error) { std::cerr<<error.what()<<'\n'; return 1; }
}
