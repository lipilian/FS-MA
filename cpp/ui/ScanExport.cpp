#include "ScanExport.hpp"
#include <QDir>
#include <QFileInfo>
#include <QJsonArray>
#include <QJsonDocument>
#include <QJsonObject>
#include <QSaveFile>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <limits>
#include <memory>
#include <stdexcept>

namespace {
QString suffix(int index) { return QString::number(index+1).rightJustified(2,'0'); }
template<int N> QJsonArray matrix(const cv::Matx<double,N,N>& value) {
    QJsonArray rows;
    for (int r=0;r<N;++r) {
        QJsonArray row;
        for (int c=0;c<N;++c) row.append(value(r,c));
        rows.append(row);
    }
    return rows;
}
bool validPose(const cv::Matx44d& pose) {
    return cv::checkRange(cv::Mat(pose)) && pose(3,0)==0 && pose(3,1)==0 && pose(3,2)==0 && pose(3,3)==1;
}
QImage exportMask(const QImage& source) {
    constexpr int width=2448, height=2048;
    QImage output(width,height,QImage::Format_Grayscale8);
    if (output.isNull()) throw std::runtime_error("Cannot allocate export mask");
    if (source.isNull()) { output.fill(0); return output; }
    const auto gray=source.convertToFormat(QImage::Format_Grayscale8);
    const cv::Mat input(gray.height(),gray.width(),CV_8UC1,
                        const_cast<uchar*>(gray.constBits()),gray.bytesPerLine());
    cv::Mat target(height,width,CV_8UC1,output.bits(),output.bytesPerLine());
    cv::resize(input,target,target.size(),0,0,cv::INTER_NEAREST);
    cv::threshold(target,target,0,255,cv::THRESH_BINARY);
    return output;
}
}

QStringList scanResultFileNames(int views) {
    QStringList names{"cameras.json"};
    for (int i=0;i<views;++i) {
        const auto number=suffix(i);
        names << "rgb"+number+".png" << "mask"+number+".png" << "depth"+number+".tiff";
    }
    return names;
}

QStringList writeScanResults(const QString& directory, const std::vector<ScanExportView>& views,
                            bool overwrite, const std::function<void()>& checkpoint) {
    const auto check=[&] { if (checkpoint) checkpoint(); };
    check();
    if (views.empty()) throw std::runtime_error("Complete at least one view before saving.");
    const QDir dir(directory);
    if (directory.trimmed().isEmpty() || !dir.exists()) throw std::runtime_error("Choose an existing output directory.");
    const auto names=scanResultFileNames(int(views.size()));
    for (const auto& name:names) {
        const QFileInfo destination(dir.filePath(name));
        if (destination.isDir() || (!overwrite && (destination.exists() || destination.isSymLink())))
            throw std::runtime_error(("Output already exists: "+destination.filePath()).toStdString());
    }
    QJsonArray cameras;
    for (int i=0;i<int(views.size());++i) {
        check();
        const auto& view=views[i];
        if (!cv::checkRange(cv::Mat(view.intrinsic)) || view.intrinsic(0,0)<=0 || view.intrinsic(1,1)<=0 ||
            view.intrinsic(2,0)!=0 || view.intrinsic(2,1)!=0 || view.intrinsic(2,2)!=1 ||
            cv::norm(cv::Mat(view.intrinsic),cv::Mat(views.front().intrinsic),cv::NORM_INF)>1e-9)
            throw std::runtime_error("All views must have the same valid FS projection intrinsics.");
        if (!validPose(view.camera_to_world))
            throw std::runtime_error("Invalid camera-to-world pose in scan results.");
        if (view.rgb.size()!=QSize(960,800) || view.rgb.format()!=QImage::Format_RGB888 ||
            view.depth.type()!=CV_32FC1 || view.depth.size()!=cv::Size(960,800) ||
            !cv::checkRange(view.depth,true,nullptr,0,std::numeric_limits<float>::max()))
            throw std::runtime_error("Each view requires its completed 960 x 800 RGB and float32 Z depth.");
        cameras.append(QJsonObject{{"view",i+1},{"extrinsic",matrix(view.camera_to_world)}});
    }
    const auto json=QJsonDocument(QJsonObject{
        {"intrinsic",matrix(views.front().intrinsic)},
        {"extrinsic_convention","camera_to_world"}, {"translation_unit","meter"}, {"depth_unit","meter"},
        {"views",cameras}}).toJson(QJsonDocument::Indented);

    // Stage every file before committing any destination. Each replacement is
    // atomic; cameras.json is committed last, after all of its view files.
    std::vector<std::unique_ptr<QSaveFile>> pending;
    const auto queue=[&](const QString& name,const auto& write) {
        check();
        auto file=std::make_unique<QSaveFile>(dir.filePath(name));
        if (!file->open(QIODevice::WriteOnly))
            throw std::runtime_error(("Cannot save "+name+": "+file->errorString()).toStdString());
        write(*file);
        pending.push_back(std::move(file));
    };
    for (int i=0;i<int(views.size());++i) {
        const auto& view=views[i];
        const auto number=suffix(i);
        queue("rgb"+number+".png",[&](auto& file) {
            if (!view.rgb.save(&file,"PNG")) throw std::runtime_error("Cannot encode scan RGB PNG");
        });
        queue("mask"+number+".png",[&](auto& file) {
            if (!exportMask(view.mask).save(&file,"PNG")) throw std::runtime_error("Cannot encode scan mask PNG");
        });
        queue("depth"+number+".tiff",[&](auto& file) {
            std::vector<uchar> encoded;
            if (!cv::imencode(".tiff",view.depth,encoded,{cv::IMWRITE_TIFF_COMPRESSION,1}))
                throw std::runtime_error("Cannot encode scan depth TIFF");
            const auto bytes=static_cast<qint64>(encoded.size());
            if (file.write(reinterpret_cast<const char*>(encoded.data()),bytes)!=bytes)
                throw std::runtime_error("Cannot write scan depth TIFF");
        });
    }
    queue("cameras.json",[&](auto& file) {
        if (file.write(json)!=json.size()) throw std::runtime_error("Cannot write cameras.json");
    });
    check();
    for (auto& file:pending) {
        if (!file->commit())
            throw std::runtime_error(("Could not finish saving "+file->fileName()+": "+file->errorString()+
                                     ". Some earlier files may have been saved.").toStdString());
    }
    return names;
}
