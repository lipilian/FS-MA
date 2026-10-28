#include "TsdfSettingsDialog.hpp"
#include <QDialogButtonBox>
#include <QDoubleSpinBox>
#include <QFormLayout>
#include <QLabel>
#include <QPushButton>
#include <QSignalBlocker>
#include <QSpinBox>
#include <QVBoxLayout>

TsdfSettingsDialog::TsdfSettingsDialog(const fs::TsdfOptions& options, QWidget* parent) : QDialog(parent) {
    setObjectName("tsdfSettingsDialog"); setWindowTitle("TSDF settings");
    auto* layout=new QVBoxLayout(this);
    auto* form=new QFormLayout; layout->addLayout(form);
    const auto decimal=[&](const char* name,double low,double high,int digits,const QString& suffix) {
        auto* box=new QDoubleSpinBox(this); box->setObjectName(name);
        box->setDecimals(digits); box->setRange(low,high); box->setSuffix(suffix);
        box->setKeyboardTracking(false); return box;
    };
    voxel_=decimal("tsdfVoxelSize",.1,10,3," mm"); voxel_->setSingleStep(.1);
    truncation_=decimal("tsdfTruncationVoxels",.0001,1000,6," × voxel"); truncation_->setSingleStep(1);
    distance_=decimal("tsdfTruncationDistance",.00001,10000,6," mm"); distance_->setSingleStep(1);
    resolution_=new QSpinBox(this); resolution_->setObjectName("tsdfBlockResolution"); resolution_->setRange(1,64);
    count_=new QSpinBox(this); count_->setObjectName("tsdfBlockCount"); count_->setRange(1,1000000); count_->setSingleStep(1000);
    count_->setToolTip("Initial voxel block capacity; Open3D can grow the grid during fusion.");
    mesh_weight_=decimal("tsdfMeshWeight",0,1000000,3,{});
    point_weight_=decimal("tsdfPointWeight",0,1000000,3,{});
    form->addRow("Voxel size",voxel_);
    form->addRow("Truncation distance (voxels)",truncation_);
    form->addRow("Actual truncation distance",distance_);
    form->addRow("Block resolution",resolution_);
    form->addRow("Block count",count_);
    form->addRow("Mesh extraction weight threshold",mesh_weight_);
    form->addRow("Point cloud extraction weight threshold",point_weight_);
    connect(voxel_,&QDoubleSpinBox::valueChanged,this,[this] { updateDistance(); });
    connect(truncation_,&QDoubleSpinBox::valueChanged,this,[this] { updateDistance(); });
    connect(distance_,&QDoubleSpinBox::valueChanged,this,[this](double mm) {
        { const QSignalBlocker blocker(truncation_); truncation_->setValue(mm/voxel_->value()); }
        updateDistance();
    });
    layout->addWidget(new QLabel("Close this window to apply settings.",this));
    auto* buttons=new QDialogButtonBox(QDialogButtonBox::Close|QDialogButtonBox::RestoreDefaults,this);
    layout->addWidget(buttons);
    connect(buttons,&QDialogButtonBox::rejected,this,&QDialog::reject);
    connect(buttons->button(QDialogButtonBox::RestoreDefaults),&QPushButton::clicked,this,[this] { setOptions({}); });
    setOptions(options);
}
void TsdfSettingsDialog::done(int) {
    // Commit the active editor first: editing the metric distance updates the
    // multiplier, and editing voxel size or multiplier updates metric distance.
    if (auto* editor=qobject_cast<QAbstractSpinBox*>(focusWidget())) editor->interpretText();
    for (auto* editor:{voxel_,truncation_,distance_,mesh_weight_,point_weight_}) editor->interpretText();
    resolution_->interpretText(); count_->interpretText();
    QDialog::done(QDialog::Accepted);
}
void TsdfSettingsDialog::updateDistance() {
    const QSignalBlocker blocker(distance_);
    distance_->setValue(voxel_->value()*truncation_->value());
}
void TsdfSettingsDialog::setOptions(const fs::TsdfOptions& o) {
    voxel_->setValue(o.voxel_size_mm); truncation_->setValue(o.truncation_voxels);
    resolution_->setValue(o.block_resolution); count_->setValue(o.block_count);
    mesh_weight_->setValue(o.mesh_weight_threshold); point_weight_->setValue(o.point_weight_threshold);
    updateDistance();
}
fs::TsdfOptions TsdfSettingsDialog::options() const {
    return {voxel_->value(),truncation_->value(),resolution_->value(),count_->value(),mesh_weight_->value(),point_weight_->value()};
}
