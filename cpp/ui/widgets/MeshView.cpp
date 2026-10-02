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
        gpu_mesh_.reset();
        mesh_ = std::move(mesh);
        failed_=false; announce_gpu_=false;
        dirty_ = true;
        resetView();
    }
    void setGPUMesh(SharedGPUMesh mesh) {
        mesh_.reset(); gpu_mesh_=std::move(mesh);
        dirty_=true; failed_=false; announce_gpu_=true;
        resetView();
    }
    void setMode(int mode) {
        mode_ = mode;
        if (!gpu_mesh_) dirty_ = true;
        update();
    }
    void setCameraVisible(bool visible) {
        camera_visible_ = visible;
        update();
    }
    void resetView() {
        yaw_ = pitch_ = 0;
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
void main() {
    point=position;
    vec3 p=deviceMesh ? (position-center)/scale*vec3(1.0,-1.0,-1.0) : position;
    gl_Position=deviceMesh && position.z<=0.0 ? vec4(2.0,2.0,2.0,1.0) : matrix*vec4(p,1.0);
    gl_PointSize=1.0;
    tint=color;
})";
        const char *fragment = R"(
#version 460 core
in vec3 tint;
in vec3 point;
layout(location = 0) out vec4 fragmentColor;
uniform bool shaded;
uniform bool wireframe;
void main() {
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
        if (!buffer_.create() || !vao_.create() || !buffer_.bind()) {
            failInitialization("Unable to create the OpenGL vertex buffer/array.");
            return;
        }
        vao_.bind();
        gl_->glEnableVertexAttribArray(0);
        gl_->glEnableVertexAttribArray(1);
        gl_->glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 6 * sizeof(float), nullptr);
        gl_->glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 6 * sizeof(float),
                                  reinterpret_cast<const void *>(3 * sizeof(float)));
        vao_.release();
        buffer_.release();
        ready_ = true;
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
        if (!ready_ || failed_ || (!gpu_mesh_ && (!mesh_ || mesh_->vertices.empty()))) return;
        if (dirty_) {
            try { if (gpu_mesh_) uploadGPU(); else upload(); }
            catch (const std::exception& e) {
                failed_=true;
                emit owner_->renderFailed(QString("Mesh rendering failed: %1").arg(e.what()));
                return;
            }
        }
        QMatrix4x4 projection;
        const float aspect = float(width()) / std::max(1, height());
        // Fit the camera as well as the mesh while preserving the mesh-centred
        // rotation pivot. Visibility changes preserve user rotation/pan/zoom.
        const float half_angle = std::atan(
            std::tan(20.f * float(CV_PI) / 180.f) * std::min(1.f, aspect));
        const float distance =
            camera_visible_ && camera_count_ > 0
                ? std::max(3.2f, camera_radius_ / std::sin(half_angle) * 1.1f)
                : 3.2f;
        projection.perspective(
            40, aspect, .01f,
            std::max(100.f, distance / zoom_ + camera_radius_ * 2));
        QMatrix4x4 view;
        view.translate(pan_.x(), pan_.y(), -distance / zoom_);
        view.rotate(pitch_, 1, 0, 0);
        view.rotate(yaw_, 0, 1, 0);
        program_->bind();
        program_->setUniformValue("matrix", projection * view);
        program_->setUniformValue("deviceMesh",bool(gpu_mesh_));
        program_->setUniformValue("center",center_);
        program_->setUniformValue("scale",scale_);
        program_->setUniformValue("shaded",bool(gpu_mesh_) && mode_==1);
        program_->setUniformValue("wireframe",bool(gpu_mesh_) && mode_==2);
        vao_.bind();
        if (gpu_mesh_ && mode_==2) gl_->glPolygonMode(GL_FRONT_AND_BACK,GL_LINE);
        gl_->glDrawArrays(mode_==0 ? GL_POINTS : (mode_==2 && !gpu_mesh_ ? GL_LINES : GL_TRIANGLES),0,count_);
        if (gpu_mesh_ && mode_==2) gl_->glPolygonMode(GL_FRONT_AND_BACK,GL_FILL);
        if (camera_visible_ && camera_count_>0) {
            program_->setUniformValue("deviceMesh",false);
            program_->setUniformValue("shaded",false);
            program_->setUniformValue("wireframe",false);
            gl_->glDrawArrays(GL_LINES,count_,camera_count_);
        }
        vao_.release();
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
            vao_.destroy();
            buffer_.destroy();
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
    void unregisterInterop() {
        if (interop_) {
            cudaCheck(cudaSetDevice(interop_device_),"select GL buffer CUDA device");
            cudaCheck(cudaGraphicsUnregisterResource(interop_),"unregister GL buffer");
            interop_=nullptr;
        }
    }
    void uploadGPU() {
        QElapsedTimer elapsed; elapsed.start();
        const auto& frame=*gpu_mesh_;
        if (!frame.buffer || !frame.stats.triangle_count || !frame.stream || !frame.stream_owner)
            throw std::runtime_error("Incomplete GPU mesh result");
        unsigned int count=0; int devices[8];
        cudaCheck(cudaGLGetDevices(&count,devices,8,cudaGLDeviceListAll),"find CUDA device for OpenGL");
        if (std::find(devices,devices+count,frame.device)==devices+count)
            throw std::runtime_error("OpenGL and mesh computation must use the same NVIDIA GPU");
        cudaCheck(cudaSetDevice(frame.device),"select mesh CUDA device");
        const auto& stats=frame.stats;
        const cv::Vec3f low(stats.low[0],stats.low[1],stats.low[2]), high(stats.high[0],stats.high[1],stats.high[2]);
        const auto center=(low+high)*.5f;
        scale_=std::max({high[0]-low[0],high[1]-low[1],high[2]-low[2],1e-6f});
        center_=QVector3D(center[0],center[1],center[2]);
        auto camera=cameraData(frame.camera,center,scale_);
        const std::size_t bytes=frame.buffer->vertex_slots()*sizeof(fs::MeshGPUVertex);
        const std::size_t total=bytes+camera.size()*sizeof(float);
        if (total>std::size_t(std::numeric_limits<int>::max())) throw std::runtime_error("GPU mesh exceeds OpenGL buffer capacity");
        buffer_.bind();
        if (!interop_ || buffer_.size()!=int(total) || interop_device_!=frame.device) {
            unregisterInterop();
            cudaCheck(cudaSetDevice(frame.device),"select mesh CUDA device");
            buffer_.allocate(nullptr,int(total));
            if (gl_->glGetError()!=GL_NO_ERROR || buffer_.size()!=int(total)) throw std::runtime_error("Unable to allocate OpenGL mesh buffer");
            cudaCheck(cudaGraphicsGLRegisterBuffer(&interop_,buffer_.bufferId(),cudaGraphicsRegisterFlagsWriteDiscard),"register OpenGL mesh buffer");
            interop_device_=frame.device;
        }
        cudaCheck(cudaGraphicsMapResources(1,&interop_,frame.stream),"map OpenGL mesh buffer");
        bool mapped=true;
        try {
            void* destination=nullptr; std::size_t capacity=0;
            cudaCheck(cudaGraphicsResourceGetMappedPointer(&destination,&capacity,interop_),"get mapped vertex pointer");
            if (capacity<total) throw std::runtime_error("Mapped OpenGL buffer is too small");
            cudaCheck(cudaMemcpyAsync(destination,frame.buffer->vertices(),bytes,cudaMemcpyDeviceToDevice,frame.stream),"copy GPU mesh into OpenGL buffer");
            cudaCheck(cudaGraphicsUnmapResources(1,&interop_,frame.stream),"unmap OpenGL mesh buffer");
            mapped=false;
            cudaCheck(cudaStreamSynchronize(frame.stream),"complete GPU mesh handoff");
        } catch (...) {
            if (mapped) cudaGraphicsUnmapResources(1,&interop_,frame.stream);
            cudaStreamSynchronize(frame.stream);
            throw;
        }
        if (!camera.empty()) buffer_.write(int(bytes),camera.data(),int(camera.size()*sizeof(float)));
        count_=int(frame.buffer->vertex_slots());
        buffer_.release(); dirty_=false;
        emit owner_->log(QString("GPU mesh → OpenGL: %1 ms (VBO setup + device copy + handoff; excludes drawing)").arg(elapsed.nsecsElapsed()/1e6,0,'f',3));
    }
    void upload() {
        unregisterInterop();
        cv::Vec3f low = mesh_->vertices[0], high = low;
        for (auto p : mesh_->vertices)
            for (int k = 0; k < 3; ++k) {
                low[k] = std::min(low[k], p[k]);
                high[k] = std::max(high[k], p[k]);
            }
        const auto center = (low + high) * .5f;
        const float scale = std::max(
            {high[0] - low[0], high[1] - low[1], high[2] - low[2], 1e-6f});
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
        const auto camera=cameraData(mesh_->camera,center,scale);
        data.insert(data.end(),camera.begin(),camera.end());
        buffer_.bind();
        buffer_.allocate(data.data(), int(data.size() * sizeof(float)));
        buffer_.release();
        dirty_ = false;
    }
    std::vector<float> cameraData(const std::optional<fs::MeshCamera>& metadata, cv::Vec3f center, float scale) {
        std::vector<float> data;
        camera_count_ = 0;
        camera_radius_ = .87f; // Radius of the mesh's normalized bounding box.
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
                const float z = std::clamp(center[2] * .12f, .015f, .06f);
                const auto ray = [&](double u, double v) {
                    return cv::Vec3f(float((u - cx) * z / fx),
                                     float((v - cy) * z / fy), z);
                };
                const cv::Vec3f origin(0, 0, 0);
                const cv::Vec3f corners[] = {
                    ray(0, 0), ray(camera.image_size.width, 0),
                    ray(camera.image_size.width, camera.image_size.height),
                    ray(0, camera.image_size.height)};
                const auto add_camera = [&](cv::Vec3f world) {
                    const auto p = (world - center) / scale;
                    camera_radius_ =
                        std::max(camera_radius_, float(cv::norm(p)));
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
                camera_count_ = int(data.size() / 6);
            }
        }
        return data;
    }
    MeshView* owner_;
    SharedGPUMesh gpu_mesh_;
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
    std::unique_ptr<QOpenGLShaderProgram> program_;
    std::shared_ptr<const fs::MeshResult> mesh_;
    bool dirty_{true}, camera_visible_{true};
    int mode_{1}, count_{}, camera_count_{};
    float camera_radius_{};
    float yaw_{}, pitch_{}, zoom_{1};
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
void MeshView::setMode(int mode) {
    if (canvas_)
        canvas_->setMode(mode);
}
void MeshView::resetView() {
    if (canvas_)
        canvas_->resetView();
}

void MeshView::setCameraVisible(bool visible) {
    if (canvas_)
        canvas_->setCameraVisible(visible);
}
