#include "MeshView.hpp"
#include <QGuiApplication>
#include <QLabel>
#include <QMatrix4x4>
#include <QMouseEvent>
#include <QOpenGLBuffer>
#include <QOpenGLContext>
#include <QOpenGLFunctions_4_5_Core>
#include <QOpenGLVersionFunctionsFactory>
#include <QOpenGLVertexArrayObject>
#include <QElapsedTimer>
#include <cuda_gl_interop.h>
#include <limits>
#include <stdexcept>
#include <QOpenGLShaderProgram>
#include <QOpenGLWidget>
#include <QVBoxLayout>
#include <QWheelEvent>
#include <algorithm>
#include <array>
#include <cmath>

class MeshCanvas : public QOpenGLWidget {
  public:
    explicit MeshCanvas(MeshView *parent)
        : QOpenGLWidget(parent), owner_(parent), buffer_(QOpenGLBuffer::VertexBuffer) {
        setMinimumSize(260, 240);
        setFormat(MeshView::surfaceFormat());
    }
    ~MeshCanvas() override { cleanup(); }
    void setMesh(std::shared_ptr<const fs::MeshResult> mesh) {
        const bool had_mesh=hasMesh();
        gpu_mesh_.reset();
        mesh_ = std::move(mesh);
        failed_=false; announce_gpu_=false;
        dirty_ = true;
        if (mesh_ && !had_mesh) resetView(); else update();
    }
    void setGPUMesh(SharedGPUMesh mesh) {
        const bool had_mesh=hasMesh();
        mesh_.reset(); gpu_mesh_=std::move(mesh);
        if (gpu_mesh_ && gpu_mesh_->point_cloud) mode_=0;
        dirty_=true; failed_=false; announce_gpu_=true;
        if (!had_mesh) resetView(); else update();
    }
    void setCapturedClouds(std::vector<SharedGPUMesh> clouds, SharedGPUMesh hidden) {
        if (clouds==captured_clouds_ && hidden==hidden_capture_) return;
        captured_clouds_=std::move(clouds);
        hidden_capture_=std::move(hidden);
        captures_dirty_=true;
        dirty_=true; failed_=false;
        update();
    }
    void setMode(int mode) {
        mode_ = mode;
        if (!gpu_mesh_) dirty_ = true;
        update();
    }
    void setCamera(const std::optional<fs::MeshCamera>& camera) {
        if (camera_.has_value() == camera.has_value() &&
            (!camera || (camera_->image_size == camera->image_size && camera_->intrinsics == camera->intrinsics))) return;
        camera_ = camera;
        camera_dirty_ = true;
        update();
    }
    void setCameraImage(QImage image, int camera = 0) {
        if (camera_images_[camera].cacheKey() == image.cacheKey()) return;
        camera_images_[camera] = std::move(image);
        image_dirty_[camera] = true;
        update();
    }
    void setRightCameraVisible(bool visible) {
        if (right_camera_visible_ == visible) return;
        right_camera_visible_ = visible;
        camera_dirty_ = true;
        update();
    }
    void resetView() {
        yaw_ = hasMesh() ? 0 : -18;
        pitch_ = hasMesh() ? 0 : -10;
        zoom_ = 1;
        pan_ = {};
        update();
    }

  protected:
    void initializeGL() override {
        ready_ = false;
        gl_ = nullptr;
        if (error_label_) error_label_->hide();
        connect(
            context(), &QOpenGLContext::aboutToBeDestroyed, this,
            [this] { cleanup(); }, Qt::DirectConnection);
        const auto actual = context()->format();
        if (context()->isOpenGLES() || actual.majorVersion() < 4 ||
            (actual.majorVersion() == 4 && actual.minorVersion() < 6) ||
            actual.profile() != QSurfaceFormat::CoreProfile) {
            failInitialization(QString("The 3D renderer requires OpenGL 4.6 Core; received %1.%2 (profile %3).")
                                   .arg(actual.majorVersion()).arg(actual.minorVersion())
                                   .arg(int(actual.profile())));
            return;
        }
        // Qt's versioned wrappers end at 4.5. These core functions are also
        // available in 4.6; the context and shaders explicitly require 4.6.
        // Reacquire the context-owned wrapper after every context recreation.
        gl_ = QOpenGLVersionFunctionsFactory::get<QOpenGLFunctions_4_5_Core>(context());
        if (!gl_) {
            failInitialization("Unable to initialize OpenGL core functions.");
            return;
        }
        gl_->glEnable(GL_DEPTH_TEST);
        program_ = std::make_unique<QOpenGLShaderProgram>();
        const char *vertex = R"(
#version 460 core
layout(location = 0) in vec3 position;
layout(location = 1) in vec3 color;
uniform mat4 matrix;
uniform bool deviceMesh;
uniform vec3 center;
uniform float scale;
out vec3 tint;
out vec3 point;
out vec2 uv;
void main() {
    point=position;
    vec3 p=deviceMesh ? (position-center)/scale*vec3(1.0,-1.0,-1.0) : position;
    gl_Position=deviceMesh && position.z<=0.0 ? vec4(2.0,2.0,2.0,1.0) : matrix*vec4(p,1.0);
    gl_PointSize=1.0;
    tint=color;
    uv=color.xy;
})";
        const char *fragment = R"(
#version 460 core
in vec3 tint;
in vec3 point;
in vec2 uv;
layout(location = 0) out vec4 fragmentColor;
uniform bool shaded;
uniform bool wireframe;
uniform bool textured;
uniform sampler2D cameraImage;
void main() {
    if (textured) {
        fragmentColor=texture(cameraImage,uv);
        return;
    }
    float light=1.0;
    if (shaded) {
        vec3 n=cross(dFdx(point),dFdy(point));
        light=0.45+0.55*abs(n.z)/max(length(n),1e-20);
    }
    fragmentColor=vec4(wireframe ? vec3(0.0) : tint*light,1.0);
})";
        if (!program_->addShaderFromSourceCode(QOpenGLShader::Vertex, vertex) ||
            !program_->addShaderFromSourceCode(QOpenGLShader::Fragment,
                                               fragment) ||
            !program_->link()) {
            failInitialization(QString("Unable to compile/link the OpenGL 4.6 shaders: %1").arg(program_->log()));
            return;
        }
        if (!buffer_.create() || !vao_.create() || !camera_buffer_.create() || !camera_vao_.create()) {
            failInitialization("Unable to create the OpenGL vertex buffer/array.");
            return;
        }
        const auto configure = [this](QOpenGLBuffer& buffer, QOpenGLVertexArrayObject& vao) {
            vao.bind(); buffer.bind();
            gl_->glEnableVertexAttribArray(0);
            gl_->glEnableVertexAttribArray(1);
            gl_->glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 6 * sizeof(float), nullptr);
            gl_->glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 6 * sizeof(float),
                                      reinterpret_cast<const void *>(3 * sizeof(float)));
            vao.release(); buffer.release();
        };
        configure(buffer_, vao_); configure(camera_buffer_, camera_vao_);
        gl_->glGenTextures(int(textures_.size()), textures_.data());
        for (const auto texture : textures_) {
            gl_->glBindTexture(GL_TEXTURE_2D, texture);
            gl_->glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
            gl_->glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
            gl_->glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
            gl_->glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
        }
        gl_->glBindTexture(GL_TEXTURE_2D, 0);
        ready_ = true;
        camera_dirty_ = true; image_dirty_.fill(true);
        captures_dirty_=true;
        dirty_ = true; failed_=false; announce_gpu_=bool(gpu_mesh_);
        emit owner_->log(QString("3D renderer: OpenGL %1 (Core), %2")
                             .arg(QString::fromLatin1(reinterpret_cast<const char *>(gl_->glGetString(GL_VERSION))))
                             .arg(QString::fromLatin1(reinterpret_cast<const char *>(gl_->glGetString(GL_RENDERER)))));
    }
    void resizeGL(int width, int height) override {
        if (gl_) gl_->glViewport(0, 0, width, height);
    }
    void paintGL() override {
        if (!gl_) return;
        gl_->glClearColor(.067f, .11f, .17f, 1);
        gl_->glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
        if (!ready_ || failed_) return;
        if (dirty_) {
            try {
                if (captures_dirty_) uploadCaptures();
                updateBounds();
                if (gpu_mesh_ && !isCaptured(gpu_mesh_)) uploadGPU();
                else if (mesh_ && !mesh_->vertices.empty()) upload();
                else { unregisterInterop(); count_ = 0; dirty_ = false; }
                camera_dirty_ = true;
            }
            catch (const std::exception& e) {
                failed_=true;
                emit owner_->renderFailed(QString("Mesh rendering failed: %1").arg(e.what()));
                return;
            }
        }
        if (camera_dirty_) uploadCamera();
        for (int camera = 0; camera < (right_camera_visible_ ? 2 : 1); ++camera)
            if (image_dirty_[camera]) uploadImage(camera);
        QMatrix4x4 projection;
        const float aspect = float(width()) / std::max(1, height());
        // Fit the camera as well as the mesh while preserving the mesh-centred
        // rotation pivot. Visibility changes preserve user rotation/pan/zoom.
        const float half_angle = std::atan(
            std::tan(20.f * float(CV_PI) / 180.f) * std::min(1.f, aspect));
        const float distance =
            camera_count_ > 0
                ? std::max(hasMesh() ? 3.2f : 0.f, camera_radius_ / std::sin(half_angle) * 1.1f)
                : 3.2f;
        projection.perspective(
            40, aspect, .01f,
            std::max(100.f, distance / zoom_ + camera_radius_ * 2));
        QMatrix4x4 view;
        view.translate(pan_.x(), pan_.y(), -distance / zoom_);
        view.rotate(pitch_, 1, 0, 0);
        view.rotate(yaw_, 0, 1, 0);
        const auto matrix = projection * view;
        program_->bind();
        program_->setUniformValue("matrix", matrix);
        program_->setUniformValue("center",center_);
        program_->setUniformValue("scale",scale_);
        program_->setUniformValue("textured",false);
        // Captures have independent VBOs. Live images never upload these again.
        program_->setUniformValue("deviceMesh",true);
        program_->setUniformValue("shaded",false);
        program_->setUniformValue("wireframe",false);
        for (const auto& capture:cloud_resources_) {
            if (capture->frame==hidden_capture_) continue;
            capture->vao.bind();
            gl_->glDrawArrays(GL_POINTS,0,capture->count);
            capture->vao.release();
        }
        program_->setUniformValue("deviceMesh",bool(gpu_mesh_));
        program_->setUniformValue("shaded",bool(gpu_mesh_) && mode_==1);
        program_->setUniformValue("wireframe",bool(gpu_mesh_) && mode_==2);
        vao_.bind();
        if (gpu_mesh_ && mode_==2) gl_->glPolygonMode(GL_FRONT_AND_BACK,GL_LINE);
        gl_->glDrawArrays(mode_==0 ? GL_POINTS : (mode_==2 && !gpu_mesh_ ? GL_LINES : GL_TRIANGLES),0,count_);
        if (gpu_mesh_ && mode_==2) gl_->glPolygonMode(GL_FRONT_AND_BACK,GL_FILL);
        vao_.release();
        if (camera_count_>0) {
            camera_vao_.bind();
            program_->setUniformValue("deviceMesh",false);
            program_->setUniformValue("shaded",false);
            program_->setUniformValue("wireframe",false);
            for (int camera = 0; camera < (right_camera_visible_ ? 2 : 1); ++camera) {
                if (!camera_images_[camera].isNull()) {
                    gl_->glActiveTexture(GL_TEXTURE0);
                    gl_->glBindTexture(GL_TEXTURE_2D,textures_[camera]);
                    program_->setUniformValue("cameraImage",0);
                    program_->setUniformValue("textured",true);
                    gl_->glEnable(GL_POLYGON_OFFSET_FILL);
                    gl_->glPolygonOffset(1.f,1.f);
                    gl_->glDrawArrays(GL_TRIANGLES,camera_count_+camera*6,6);
                    gl_->glDisable(GL_POLYGON_OFFSET_FILL);
                    gl_->glBindTexture(GL_TEXTURE_2D,0);
                    program_->setUniformValue("textured",false);
                }
            }
            gl_->glDrawArrays(GL_LINES,0,camera_count_);
            camera_vao_.release();
        }
        program_->release();
        if (announce_gpu_ && gpu_mesh_) {
            announce_gpu_=false;
            const auto error=gl_->glGetError();
            if (error!=GL_NO_ERROR) {
                failed_=true;
                emit owner_->renderFailed(QString("OpenGL mesh draw failed (error %1).").arg(error));
            } else emit owner_->gpuMeshPresented(gpu_mesh_);
        }
    }
    void mousePressEvent(QMouseEvent *e) override { last_ = e->position(); }
    void mouseMoveEvent(QMouseEvent *e) override {
        auto d = e->position() - last_;
        last_ = e->position();
        if (e->buttons() & Qt::LeftButton) {
            yaw_ += d.x() * .4;
            pitch_ += d.y() * .4;
        }
        if (e->buttons() & Qt::RightButton)
            pan_ += QPointF(d.x() / std::max(1, width()) * 2,
                            -d.y() / std::max(1, height()) * 2);
        update();
    }
    void wheelEvent(QWheelEvent *e) override {
        zoom_ = std::clamp(zoom_ * std::pow(1.15f, e->angleDelta().y() / 120.f),
                           .2f, 8.f);
        update();
        e->accept();
    }

  private:
    struct CloudResource {
        SharedGPUMesh frame;
        QOpenGLBuffer buffer{QOpenGLBuffer::VertexBuffer};
        QOpenGLVertexArrayObject vao;
        cudaGraphicsResource* interop{nullptr};
        int device{0}, count{0};
    };
    bool hasMesh() const { return !captured_clouds_.empty() || gpu_mesh_ || (mesh_ && !mesh_->vertices.empty()); }
    bool isCaptured(const SharedGPUMesh& mesh) const {
        return std::find(captured_clouds_.begin(),captured_clouds_.end(),mesh)!=captured_clouds_.end();
    }
    void destroyCapture(CloudResource& capture) {
        if (capture.interop) {
            cudaSetDevice(capture.device);
            cudaGraphicsUnregisterResource(capture.interop);
            capture.interop=nullptr;
        }
        capture.vao.destroy(); capture.buffer.destroy();
    }
    void failInitialization(const QString &message) {
        program_.reset();
        if (!error_label_) {
            error_label_ = new QLabel("Unable to initialize the 3D renderer (OpenGL 4.6 Core required)", this);
            error_label_->setStyleSheet("color: white;");
            error_label_->adjustSize();
        }
        error_label_->show();
        emit owner_->renderFailed(message);
    }
    void cleanup() {
        ready_ = false;
        if (context()) {
            makeCurrent();
            if (interop_) {
                cudaSetDevice(interop_device_);
                cudaGraphicsUnregisterResource(interop_); interop_=nullptr;
            }
            for (auto& capture:cloud_resources_) destroyCapture(*capture);
            cloud_resources_.clear();
            vao_.destroy();
            buffer_.destroy();
            camera_vao_.destroy();
            camera_buffer_.destroy();
            if (gl_) gl_->glDeleteTextures(int(textures_.size()), textures_.data());
            textures_ = {}; texture_sizes_ = {};
            program_.reset();
            doneCurrent();
        }
        if (context())
            disconnect(context(), nullptr, this, nullptr);
        gl_ = nullptr;
    }
    static void cudaCheck(cudaError_t e, const char* operation) {
        if (e!=cudaSuccess) throw std::runtime_error(std::string(operation)+": "+cudaGetErrorString(e));
    }
    static void unregisterInterop(cudaGraphicsResource*& interop, int device) {
        if (interop) {
            cudaCheck(cudaSetDevice(device),"select GL buffer CUDA device");
            cudaCheck(cudaGraphicsUnregisterResource(interop),"unregister GL buffer");
            interop=nullptr;
        }
    }
    void unregisterInterop() { unregisterInterop(interop_,interop_device_); }
    void updateBounds() {
        cv::Vec3f low(std::numeric_limits<float>::max(),std::numeric_limits<float>::max(),std::numeric_limits<float>::max());
        cv::Vec3f high(-low);
        bool any=false;
        const auto add=[&](const cv::Vec3f& p) {
            any=true;
            for (int k=0;k<3;++k) { low[k]=std::min(low[k],p[k]); high[k]=std::max(high[k],p[k]); }
        };
        const auto add_gpu=[&](const SharedGPUMesh& mesh) {
            if (!mesh) return;
            const auto& s=mesh->stats;
            add({s.low[0],s.low[1],s.low[2]}); add({s.high[0],s.high[1],s.high[2]});
        };
        for (const auto& cloud:captured_clouds_) if (cloud!=hidden_capture_) add_gpu(cloud);
        add_gpu(gpu_mesh_);
        if (mesh_) for (const auto& p:mesh_->vertices) add(p);
        if (!any) { center_={}; scale_=1; return; }
        const auto center=(low+high)*.5f;
        center_=QVector3D(center[0],center[1],center[2]);
        scale_=std::max({high[0]-low[0],high[1]-low[1],high[2]-low[2],1e-6f});
    }
    void uploadCaptures() {
        for (auto it=cloud_resources_.begin();it!=cloud_resources_.end();) {
            if (isCaptured((*it)->frame)) { ++it; continue; }
            destroyCapture(**it); it=cloud_resources_.erase(it);
        }
        for (const auto& frame:captured_clouds_) {
            if (!frame) continue;
            const auto existing=std::find_if(cloud_resources_.begin(),cloud_resources_.end(),
                [&](const auto& resource) { return resource->frame==frame; });
            if (existing!=cloud_resources_.end()) continue;
            auto resource=std::make_unique<CloudResource>(); resource->frame=frame;
            try {
                if (!resource->buffer.create() || !resource->vao.create())
                    throw std::runtime_error("Unable to create captured point-cloud buffer/array");
                resource->vao.bind(); resource->buffer.bind();
                gl_->glEnableVertexAttribArray(0); gl_->glEnableVertexAttribArray(1);
                gl_->glVertexAttribPointer(0,3,GL_FLOAT,GL_FALSE,sizeof(fs::MeshGPUVertex),nullptr);
                gl_->glVertexAttribPointer(1,3,GL_FLOAT,GL_FALSE,sizeof(fs::MeshGPUVertex),reinterpret_cast<const void*>(3*sizeof(float)));
                resource->vao.release(); resource->buffer.release();
                uploadGPU(*frame,resource->buffer,resource->interop,resource->device);
                resource->count=int(frame->buffer->vertex_slots());
                cloud_resources_.push_back(std::move(resource));
            } catch (...) {
                if (resource) destroyCapture(*resource);
                throw;
            }
        }
        captures_dirty_=false;
    }
    void uploadGPU() {
        uploadGPU(*gpu_mesh_,buffer_,interop_,interop_device_);
        count_=int(gpu_mesh_->buffer->vertex_slots()); dirty_=false;
    }
    void uploadGPU(const GPUMeshFrame& frame, QOpenGLBuffer& buffer,
                   cudaGraphicsResource*& interop, int& interop_device) {
        QElapsedTimer elapsed; elapsed.start();
        if (!frame.buffer || !(frame.point_cloud ? frame.stats.point_count : frame.stats.triangle_count) || !frame.stream || !frame.stream_owner)
            throw std::runtime_error("Incomplete GPU mesh result");
        unsigned int count=0; int devices[8];
        cudaCheck(cudaGLGetDevices(&count,devices,8,cudaGLDeviceListAll),"find CUDA device for OpenGL");
        if (std::find(devices,devices+count,frame.device)==devices+count)
            throw std::runtime_error("OpenGL and mesh computation must use the same NVIDIA GPU");
        cudaCheck(cudaSetDevice(frame.device),"select mesh CUDA device");
        const std::size_t bytes=frame.buffer->vertex_slots()*sizeof(fs::MeshGPUVertex);
        const std::size_t total=bytes;
        if (total>std::size_t(std::numeric_limits<int>::max())) throw std::runtime_error("GPU mesh exceeds OpenGL buffer capacity");
        buffer.bind();
        if (!interop || buffer.size()!=int(total) || interop_device!=frame.device) {
            unregisterInterop(interop,interop_device);
            cudaCheck(cudaSetDevice(frame.device),"select mesh CUDA device");
            buffer.allocate(nullptr,int(total));
            if (gl_->glGetError()!=GL_NO_ERROR || buffer.size()!=int(total)) throw std::runtime_error("Unable to allocate OpenGL mesh buffer");
            cudaCheck(cudaGraphicsGLRegisterBuffer(&interop,buffer.bufferId(),cudaGraphicsRegisterFlagsWriteDiscard),"register OpenGL mesh buffer");
            interop_device=frame.device;
        }
        cudaCheck(cudaGraphicsMapResources(1,&interop,frame.stream),"map OpenGL mesh buffer");
        bool mapped=true;
        try {
            void* destination=nullptr; std::size_t capacity=0;
            cudaCheck(cudaGraphicsResourceGetMappedPointer(&destination,&capacity,interop),"get mapped vertex pointer");
            if (capacity<total) throw std::runtime_error("Mapped OpenGL buffer is too small");
            cudaCheck(cudaMemcpyAsync(destination,frame.buffer->vertices(),bytes,cudaMemcpyDeviceToDevice,frame.stream),"copy GPU mesh into OpenGL buffer");
            cudaCheck(cudaGraphicsUnmapResources(1,&interop,frame.stream),"unmap OpenGL mesh buffer");
            mapped=false;
            cudaCheck(cudaStreamSynchronize(frame.stream),"complete GPU mesh handoff");
        } catch (...) {
            if (mapped) cudaGraphicsUnmapResources(1,&interop,frame.stream);
            cudaStreamSynchronize(frame.stream);
            throw;
        }
        buffer.release();
        emit owner_->log(QString("GPU %1 → OpenGL: %2 ms (VBO setup + device copy + handoff; excludes drawing)")
            .arg(frame.point_cloud ? "point cloud" : "mesh").arg(elapsed.nsecsElapsed()/1e6,0,'f',3));
    }
    void upload() {
        unregisterInterop();
        const cv::Vec3f center(center_.x(),center_.y(),center_.z());
        const float scale=scale_;
        std::vector<float> data;
        data.reserve(mode_ == 0
                         ? mesh_->vertices.size() * 6
                         : mesh_->triangles.size() * (mode_ == 2 ? 36 : 18));
        const auto add = [&](int id, float light) {
            auto p = (mesh_->vertices[id] - center) / scale;
            const auto c = mode_ == 2 ? cv::Vec3b(0, 0, 0) : mesh_->colors[id];
            data.insert(data.end(),
                        {p[0], -p[1], -p[2], c[0] / 255.f * light,
                         c[1] / 255.f * light, c[2] / 255.f * light});
        };
        if (mode_ == 0) {
            for (size_t i = 0; i < mesh_->vertices.size(); ++i)
                add(i, 1);
        } else
            for (auto f : mesh_->triangles) {
                const auto n =
                    (mesh_->vertices[f[1]] - mesh_->vertices[f[0]])
                        .cross(mesh_->vertices[f[2]] - mesh_->vertices[f[0]]);
                const float light =
                    .45f + .55f * std::abs(n[2]) /
                               std::max(float(cv::norm(n)), 1e-10f);
                if (mode_ == 2)
                    for (int i = 0; i < 3; ++i) {
                        add(f[i], 1);
                        add(f[(i + 1) % 3], 1);
                    }
                else
                    for (int i = 0; i < 3; ++i)
                        add(f[i], light);
            }
        count_ = int(data.size() / 6);
        buffer_.bind();
        buffer_.allocate(data.data(), int(data.size() * sizeof(float)));
        buffer_.release();
        dirty_ = false;
    }
    void uploadImage(int camera) {
        image_dirty_[camera] = false;
        if (camera_images_[camera].isNull()) return;
        const auto image = camera_images_[camera].convertToFormat(QImage::Format_RGBA8888);
        gl_->glActiveTexture(GL_TEXTURE0);
        gl_->glBindTexture(GL_TEXTURE_2D, textures_[camera]);
        gl_->glPixelStorei(GL_UNPACK_ALIGNMENT, 4);
        gl_->glPixelStorei(GL_UNPACK_ROW_LENGTH, image.bytesPerLine() / 4);
        if (texture_sizes_[camera] != image.size()) {
            gl_->glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, image.width(), image.height(),
                              0, GL_RGBA, GL_UNSIGNED_BYTE, image.constBits());
            texture_sizes_[camera] = image.size();
        } else gl_->glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, image.width(), image.height(),
                                    GL_RGBA, GL_UNSIGNED_BYTE, image.constBits());
        gl_->glBindTexture(GL_TEXTURE_2D, 0);
        gl_->glPixelStorei(GL_UNPACK_ROW_LENGTH, 0);
    }
    void uploadCamera() {
        // The live frustum has its own buffer: new images never re-upload mesh
        // geometry or touch CUDA/OpenGL interop resources.
        const auto metadata = camera_ ? camera_ : gpu_mesh_ ? std::optional<fs::MeshCamera>(gpu_mesh_->camera)
                                                   : mesh_ ? mesh_->camera : std::nullopt;
        const cv::Vec3f center(center_.x(), center_.y(), center_.z());
        auto data = cameraData(metadata, center, scale_,camera_count_,camera_radius_);
        // Keep each completed capture's frustum even when live calibration or
        // the active input changes. Poses are still local camera coordinates;
        // multi-view registration will supply their transforms separately.
        std::vector<float> lines;
        for (const auto& frame:captured_clouds_) {
            if (!frame) continue;
            int count=0; float radius=0;
            const auto captured=cameraData(frame->camera,center,scale_,count,radius);
            lines.insert(lines.end(),captured.begin(),captured.begin()+count*6);
            camera_radius_=std::max(camera_radius_,radius);
        }
        data.insert(data.begin(),lines.begin(),lines.end());
        camera_count_+=int(lines.size()/6);
        camera_buffer_.bind();
        camera_buffer_.allocate(data.data(), int(data.size() * sizeof(float)));
        camera_buffer_.release();
        camera_dirty_ = false;
    }
    std::vector<float> cameraData(const std::optional<fs::MeshCamera>& metadata, cv::Vec3f center, float scale,
                                  int& line_count, float& radius) {
        std::vector<float> data;
        line_count = 0;
        radius = .87f; // Radius of the mesh's normalized bounding box.
        if (metadata) {
            const auto &camera = *metadata;
            const double fx = camera.intrinsics(0, 0),
                         fy = camera.intrinsics(1, 1);
            const double cx = camera.intrinsics(0, 2),
                         cy = camera.intrinsics(1, 2);
            if (camera.image_size.width > 0 && camera.image_size.height > 0 &&
                std::isfinite(fx) && fx > 0 && std::isfinite(fy) && fy > 0 &&
                std::isfinite(cx) && std::isfinite(cy)) {
                // A short frustum depicts field of view, not the depth clip
                // plane or physical camera housing. Its apex is the true XYZ
                // origin.
                const float z = hasMesh() ? std::clamp(center[2] * .12f, .015f, .06f) : .06f;
                const auto ray = [&](double u, double v) {
                    return cv::Vec3f(float((u - cx) * z / fx),
                                     float((v - cy) * z / fy), z);
                };
                const cv::Vec3f origin(0, 0, 0);
                const cv::Vec3f corners[] = {
                    ray(0, 0), ray(camera.image_size.width, 0),
                    ray(camera.image_size.width, camera.image_size.height),
                    ray(0, camera.image_size.height)};
                // The comparison image has equal world dimensions and shares
                // the left image's plane. Translate before the common view /
                // projection transform so their perspective stays consistent.
                // This preview spacing does not represent the rig's baseline.
                const float right_offset = (corners[1][0] - corners[0][0]) * 1.2f;
                if (!hasMesh()) {
                    center = (corners[0] + corners[2]) * .5f;
                    if (right_camera_visible_) center[0] += right_offset * .5f;
                    center[2] = z * .8f;
                    scale = std::max({float(cv::norm(corners[1]-corners[0])),
                                      float(cv::norm(corners[3]-corners[0])), z * 1.6f});
                    radius = 0;
                }
                const auto add_camera = [&](cv::Vec3f world) {
                    const auto p = (world - center) / scale;
                    radius = std::max(radius, float(cv::norm(p)));
                    data.insert(data.end(),
                                {p[0], -p[1], -p[2], 1.f, .65f, .12f});
                };
                const auto line = [&](cv::Vec3f a, cv::Vec3f b) {
                    add_camera(a);
                    add_camera(b);
                };
                for (int i = 0; i < 4; ++i) {
                    line(origin, corners[i]);
                    line(corners[i], corners[(i + 1) % 4]);
                }
                // Optical axis and arrowhead point along the camera's +Z axis.
                const cv::Vec3f tip(0, 0, z * 1.6f);
                line(origin, tip);
                line(tip, cv::Vec3f(-z * .12f, 0, z * 1.35f));
                line(tip, cv::Vec3f(z * .12f, 0, z * 1.35f));
                // Up marker distinguishes the image top from the bottom.
                const auto top = (corners[0] + corners[1]) * .5f;
                const auto up = top + cv::Vec3f(0, -z * .2f, 0);
                line(corners[0], up);
                line(up, corners[1]);
                line_count = int(data.size() / 6);
                // QImage row zero is the image top. Upload unchanged and give
                // the upper frustum corners v=0 to avoid a vertical flip.
                const float uv[][2] = {{0,0}, {1,0}, {1,1}, {0,1}};
                for (int camera=0; camera<(right_camera_visible_ ? 2 : 1); ++camera)
                    for (int i : {0,1,2,0,2,3}) {
                        const auto world = corners[i] + cv::Vec3f(camera * right_offset,0,0);
                        const auto p = (world - center) / scale;
                        radius = std::max(radius,float(cv::norm(p)));
                        data.insert(data.end(), {p[0], -p[1], -p[2], uv[i][0], uv[i][1], 0.f});
                    }
            }
        }
        return data;
    }
    MeshView* owner_;
    SharedGPUMesh gpu_mesh_;
    std::vector<SharedGPUMesh> captured_clouds_;
    SharedGPUMesh hidden_capture_;
    std::vector<std::unique_ptr<CloudResource>> cloud_resources_;
    bool captures_dirty_{true};
    cudaGraphicsResource* interop_{nullptr};
    int interop_device_{0};
    QVector3D center_;
    float scale_{1};
    bool failed_{false}, announce_gpu_{false};
    bool ready_{false};
    QOpenGLFunctions_4_5_Core *gl_{nullptr}; // Owned by the current QOpenGLContext.
    QLabel *error_label_{nullptr};
    QOpenGLBuffer buffer_;
    QOpenGLVertexArrayObject vao_;
    QOpenGLBuffer camera_buffer_{QOpenGLBuffer::VertexBuffer};
    QOpenGLVertexArrayObject camera_vao_;
    std::optional<fs::MeshCamera> camera_;
    std::array<QImage, 2> camera_images_;
    std::array<GLuint, 2> textures_{};
    std::array<QSize, 2> texture_sizes_;
    std::array<bool, 2> image_dirty_{{true, true}};
    bool camera_dirty_{true};
    std::unique_ptr<QOpenGLShaderProgram> program_;
    std::shared_ptr<const fs::MeshResult> mesh_;
    bool dirty_{true}, right_camera_visible_{false};
    int mode_{1}, count_{}, camera_count_{};
    float camera_radius_{};
    float yaw_{-18}, pitch_{-10}, zoom_{1};
    QPointF last_, pan_;
};

QSurfaceFormat MeshView::surfaceFormat() {
    QSurfaceFormat format;
    format.setRenderableType(QSurfaceFormat::OpenGL);
    format.setVersion(4, 6);
    format.setProfile(QSurfaceFormat::CoreProfile);
    format.setDepthBufferSize(24);
    return format;
}

MeshView::MeshView(QWidget *parent) : QWidget(parent) {
    auto *layout = new QVBoxLayout(this);
    layout->setContentsMargins(0, 0, 0, 0);
    if (QGuiApplication::platformName() == "offscreen") {
        auto *text = new QLabel("3D view unavailable in this display session");
        text->setAlignment(Qt::AlignCenter);
        layout->addWidget(text);
    } else {
        canvas_ = new MeshCanvas(this);
        layout->addWidget(canvas_);
    }
}
void MeshView::setMesh(std::shared_ptr<const fs::MeshResult> mesh) {
    if (canvas_)
        canvas_->setMesh(std::move(mesh));
}
void MeshView::setGPUMesh(SharedGPUMesh mesh) {
    if (canvas_) canvas_->setGPUMesh(std::move(mesh));
    else emit renderFailed("GPU mesh rendering requires a hardware OpenGL display session.");
}
void MeshView::setCapturedClouds(std::vector<SharedGPUMesh> clouds, SharedGPUMesh hidden) {
    if (canvas_) canvas_->setCapturedClouds(std::move(clouds),std::move(hidden));
}
void MeshView::setMode(int mode) {
    if (canvas_)
        canvas_->setMode(mode);
}
void MeshView::resetView() {
    if (canvas_)
        canvas_->resetView();
}

void MeshView::setRightCameraVisible(bool visible) {
    if (canvas_)
        canvas_->setRightCameraVisible(visible);
}

void MeshView::setCamera(const std::optional<fs::MeshCamera>& camera) {
    if (canvas_) canvas_->setCamera(camera);
}

void MeshView::setCameraImage(QImage image) {
    if (canvas_) canvas_->setCameraImage(std::move(image));
}

void MeshView::setRightCameraImage(QImage image) {
    if (canvas_) canvas_->setCameraImage(std::move(image), 1);
}
