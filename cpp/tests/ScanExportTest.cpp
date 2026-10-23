#include "ScanExport.hpp"
#include <QCoreApplication>
#include <QDir>
#include <QFile>
#include <QJsonArray>
#include <QJsonDocument>
#include <QJsonObject>
#include <QTemporaryDir>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <cstring>
#include <iostream>
#include <limits>
#include <stdexcept>

namespace {
void require(bool ok,const char* message) { if (!ok) throw std::runtime_error(message); }
QByteArray read(const QString& path) {
    QFile file(path); require(file.open(QIODevice::ReadOnly),"Cannot read saved scan file"); return file.readAll();
}
void verify(const QString& directory,const std::vector<ScanExportView>& expected) {
    const auto json=QJsonDocument::fromJson(read(directory+"/cameras.json")).object();
    require(json.value("extrinsic_convention")=="camera_to_world" &&
            json.value("translation_unit")=="meter" && json.value("depth_unit")=="meter",
            "Export omitted pose direction or metric units");
    const auto k=json.value("intrinsic").toArray(), views=json.value("views").toArray();
    require(k.size()==3 && views.size()==int(expected.size()),"Wrong intrinsic/view count in JSON");
    for (int r=0;r<3;++r) for (int c=0;c<3;++c)
        require(k[r].toArray()[c].toDouble()==expected.front().intrinsic(r,c),"Export changed FS intrinsics");
    require(QDir(directory).entryList(QDir::Files|QDir::NoDotAndDotDot).size()==1+3*int(expected.size()) &&
            QDir(directory).entryList(QDir::Dirs|QDir::NoDotAndDotDot).isEmpty(),
            "Export did not save exactly one JSON and RGB/mask/depth per view in a flat folder");
    for (int i=0;i<int(expected.size());++i) {
        const auto number=QString::number(i+1).rightJustified(2,'0');
        const auto& view=expected[i];
        const auto camera=views[i].toObject();
        const auto pose=camera.value("extrinsic").toArray();
        const auto before=camera.value("extrinsic_before_icp").toArray();
        const auto after=camera.value("extrinsic_after_icp");
        require(camera.value("view").toInt()==i+1 && pose.size()==4,"View numbering or pose dimensions are wrong");
        require(before.size()==4 && camera.value("icp_applied").isBool() &&
                camera.value("icp_applied").toBool()==view.icp_applied,"Missing original pose or incorrect active ICP state");
        require(view.camera_to_world_after_icp ? after.isArray() && after.toArray().size()==4 : after.isNull(),
                "Export lost the ICP result or invented one before ICP ran");
        for (int r=0;r<4;++r) for (int c=0;c<4;++c) {
            require(pose[r].toArray()[c].toDouble()==view.camera_to_world(r,c),
                    "Export transposed/inverted/rebased an MA pose");
            require(before[r].toArray()[c].toDouble()==view.camera_to_world_before_icp(r,c),
                    "Export changed the original pose before ICP");
            if (view.camera_to_world_after_icp)
                require(after.toArray()[r].toArray()[c].toDouble()==(*view.camera_to_world_after_icp)(r,c),
                        "Export changed the saved pose after ICP");
        }
        const auto rgb=cv::imread((directory+"/rgb"+number+".png").toStdString(),cv::IMREAD_UNCHANGED);
        require(rgb.size()==cv::Size(960,800) && rgb.type()==CV_8UC3,"RGB PNG has incorrect dimensions/type");
        cv::Mat decoded; cv::cvtColor(rgb,decoded,cv::COLOR_BGR2RGB);
        for (int y=0;y<800;++y)
            require(std::memcmp(decoded.ptr(y),view.rgb.constScanLine(y),960*3)==0,"Export changed RGB values or channel order");
        const auto depth=cv::imread((directory+"/depth"+number+".tiff").toStdString(),cv::IMREAD_UNCHANGED);
        require(depth.size()==cv::Size(960,800) && depth.type()==CV_32FC1,"Depth TIFF is not 800 x 960 float32");
        for (int y=0;y<800;++y)
            require(std::memcmp(depth.ptr(y),view.depth.ptr(y),960*sizeof(float))==0,"Export changed metric Z-depth bits");
        const auto mask=cv::imread((directory+"/mask"+number+".png").toStdString(),cv::IMREAD_UNCHANGED);
        require(mask.size()==cv::Size(2448,2048) && mask.type()==CV_8UC1,"Mask is not 2048 x 2448 uint8");
        for (int y=0;y<mask.rows;++y) for (int x=0;x<mask.cols;++x) {
            const int want=view.mask.isNull() ? 0 :
                view.mask.constScanLine(y*view.mask.height()/mask.rows)[x*view.mask.width()/mask.cols]!=0 ? 255 : 0;
            require(mask.at<uchar>(y,x)==want,"Mask binary values, view assignment or nearest resize changed");
        }
    }
}
}

int main(int argc,char** argv) {
    QCoreApplication app(argc,argv);
    try {
        std::vector<ScanExportView> views;
        for (int i=0;i<5;++i) {
            ScanExportView view;
            view.intrinsic={602.25,0,481.125,0,603.5,398.625,0,0,1};
            // First camera deliberately has a nonidentity rotation/translation.
            view.camera_to_world={0,-1,0,i*.03125,1,0,0,-.0625,0,0,1,.125,0,0,0,1};
            view.camera_to_world_before_icp=view.camera_to_world;
            view.rgb=QImage(960,800,QImage::Format_RGB888);
            view.depth=cv::Mat(800,960,CV_32FC1);
            for (int y=0;y<800;++y) for (int x=0;x<960;++x) {
                auto* pixel=view.rgb.scanLine(y)+x*3;
                pixel[0]=(x+i*20)%256; pixel[1]=(y+i*40)%256; pixel[2]=(x+y+i*60)%256;
                view.depth.at<float>(y,x)=(x+y)%7 ? .125f+(x+960*y+i)/1000000.f : 0.f;
            }
            if (i<4) {
                view.mask=QImage(i==3 ? 320 : 2448,i==3 ? 240 : 2048,QImage::Format_Grayscale8);
                view.mask.fill(0);
                for (int y=0;y<view.mask.height()/2;++y)
                    std::memset(view.mask.scanLine(y),255,view.mask.width()/(i+2));
            }
            views.push_back(std::move(view));
        }
        QTemporaryDir output;
        require(output.isValid(),"Cannot create scan export fixture");
        writeScanResults(output.path(),views);
        verify(output.path(),views);
        const auto before=read(output.filePath("cameras.json"));
        bool refused=false;
        try { writeScanResults(output.path(),views); } catch (const std::exception&) { refused=true; }
        require(refused && read(output.filePath("cameras.json"))==before,"Export overwrote files without confirmation");
        views.back().camera_to_world(0,3)+=.125;
        views.back().camera_to_world_before_icp=views.back().camera_to_world;
        writeScanResults(output.path(),views,true);
        verify(output.path(),views);
        for (std::size_t i=0;i<views.size();++i) {
            auto& view=views[i];
            view.camera_to_world_after_icp=view.camera_to_world_before_icp;
            if (i>0) (*view.camera_to_world_after_icp)(1,3)+=.002;
            view.camera_to_world=*view.camera_to_world_after_icp;
            view.icp_applied=true;
        }
        writeScanResults(output.path(),views,true); verify(output.path(),views);
        for (auto& view:views) {
            view.camera_to_world=view.camera_to_world_before_icp;
            view.icp_applied=false; // Undo keeps the most recent ICP result for comparison.
        }
        writeScanResults(output.path(),views,true); verify(output.path(),views);
        QTemporaryDir aborted;
        int checks=0; bool cancelled=false;
        try {
            writeScanResults(aborted.path(),views,false,[&] {
                if (++checks==10) throw std::runtime_error("cancelled");
            });
        } catch (const std::exception&) { cancelled=true; }
        require(cancelled && QDir(aborted.path()).entryList(QDir::Files|QDir::NoDotAndDotDot).isEmpty(),
                "Cancelling during encoding committed a partial dataset");
        auto mismatched=views;
        mismatched.back().intrinsic(0,0)+=1;
        refused=false;
        try { writeScanResults(aborted.path(),mismatched); } catch (const std::exception&) { refused=true; }
        require(refused && QDir(aborted.path()).entryList(QDir::Files|QDir::NoDotAndDotDot).isEmpty(),
                "Export silently shared incompatible intrinsics between views");
        for (bool after : {false,true}) {
            auto invalid=views;
            auto& pose=after ? *invalid.back().camera_to_world_after_icp : invalid.back().camera_to_world_before_icp;
            pose(0,3)=std::numeric_limits<double>::quiet_NaN();
            refused=false;
            try { writeScanResults(aborted.path(),invalid); } catch (const std::exception&) { refused=true; }
            require(refused && QDir(aborted.path()).entryList(QDir::Files|QDir::NoDotAndDotDot).isEmpty(),
                    "Export accepted a nonfinite before/after ICP pose or wrote partial results");
        }
        QTemporaryDir single;
        views.resize(1); views.front().camera_to_world=cv::Matx44d::eye(); views.front().mask={};
        views.front().camera_to_world_before_icp=cv::Matx44d::eye(); views.front().camera_to_world_after_icp.reset();
        writeScanResults(single.path(),views); verify(single.path(),views);
        std::cout<<"PASS: exact before/after ICP poses, active pose compatibility, null before ICP, retained result after undo and invalid-pose rejection\n";
        std::cout<<"PASS: one/five-view flat export, exact RGB and float32 depth, binary full-resolution masks, shared K and camera-to-world poses\n";
        std::cout<<"PASS: zero undrawn masks, identity single view, overwrite guard, staged cancellation and incompatible-intrinsic rejection\n";
    } catch (const std::exception& error) { std::cerr<<error.what()<<'\n'; return 1; }
}
