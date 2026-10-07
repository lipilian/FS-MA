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
#include <QTimer>
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
        animation_timer_.setInterval(33);
        animation_timer_.setTimerType(Qt::PreciseTimer);
        connect(&animation_timer_, &QTimer::timeout, this, [this] { if (isVisible()) update(); });
    }
    ~MeshCanvas() override { cleanup(); }
    void setMesh(std::shared_ptr<const fs::MeshResult> mesh) {
        const bool had_mesh=hasMesh();
        gpu_mesh_.reset();
        mesh_ = std::move(mesh);
        failed_=false; announce_gpu_=false;
        dirty_ = true;
        if (mesh_ && !had_mesh && !preserve_view_) resetView(); else update();
    }
    void setGPUMesh(SharedGPUMesh mesh) {
        const bool had_mesh=hasMesh();
        mesh_.reset(); gpu_mesh_=std::move(mesh);
        if (gpu_mesh_ && gpu_mesh_->point_cloud) mode_=0;
        dirty_=true; failed_=false; announce_gpu_=true;
        if (!had_mesh && !preserve_view_) resetView(); else update();
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
    void setMAClouds(SharedGPUClouds clouds) {
        if (ma_clouds_==clouds) return;
        ma_clouds_=std::move(clouds);
        captures_dirty_=true; dirty_=true; failed_=false;
        update();
    }
    void setShowMAClouds(bool enabled) {
        if (show_ma_clouds_==enabled) return;
        show_ma_clouds_=enabled;
        // Keep identical framing when comparing both reconstructions.
        preserveView();
        dirty_=true;
        update();
    }
    void setPredictedCameras(SharedPredictedCameras cameras) {
        if (predicted_cameras_ == cameras) return;
        predicted_cameras_ = std::move(cameras);
        camera_dirty_ = true;
        if (predicted_cameras_) preserveView();
        update();
    }
    void setCamera(const std::optional<fs::MeshCamera>& camera) {
        if (camera_.has_value() == camera.has_value() &&
            (!camera || (camera_->image_size == camera->image_size && camera_->intrinsics == camera->intrinsics))) return;
        camera_ = camera;
        camera_dirty_ = true; preview_geometry_dirty_ = true;
        if (!hasMesh()) dirty_ = true;
        update();
    }
    void setLivePreview(bool enabled) {
        if (live_preview_ == enabled) return;
        live_preview_ = enabled;
        camera_dirty_ = true;
        if (!hasMesh()) dirty_ = true;
        update();
    }
    void setCameraImage(QImage image, int camera = 0) {
        if (camera_images_[camera].cacheKey() == image.cacheKey()) return;
        if (camera == 0 && !image.isNull() && preview_size_ != image.size())
            preview_geometry_dirty_ = true;
        camera_images_[camera] = std::move(image);
        image_dirty_[camera] = true;
        update();
    }
    void setRightCameraVisible(bool visible) {
        if (right_camera_visible_ == visible) return;
        right_camera_visible_ = visible;
        camera_dirty_ = true;
        if (!hasMesh()) dirty_ = true;
        update();
    }
    void resetView() {
        if (live_preview_ || !hasMesh()) { resetCaptureView(); return; }
        preserve_view_ = false;
        dirty_ = true;
        yaw_ = 0;
        pitch_ = 0;
        zoom_ = 1;
        pan_ = {};
        update();
    }
    void resetCaptureView() {
        preserve_view_ = false;
        dirty_ = true;
        yaw_ = kCaptureYaw;
        pitch_ = kCapturePitch;
        zoom_ = 1;
        pan_ = {};
        update();
    }
    void preserveView() {
        // Retain the last displayed center, world scale and camera distance.
        // Mouse rotation/pan/zoom remain available throughout reconstruction.
        preserve_view_ = true;
    }
    void setProcessing(bool processing) {
        if (processing_ == processing) return;
        if (processing) {
            processing_camera_ = activeCamera();
            processing_images_ = camera_images_; // Shared, immutable last preview frame.
            animation_clock_.start();
            animation_timer_.start();
        } else {
            animation_timer_.stop();
            processing_images_ = {};
            processing_camera_.reset();
        }
        processing_ = processing;
        image_dirty_.fill(true); camera_dirty_ = true;
        if (!hasMesh()) dirty_ = true;
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
uniform mat4 cameraToWorld;
uniform bool cameraSpace;
uniform bool clipInvalid;
uniform bool previewPoints;
uniform bool processingPreview;
uniform float processingTime;
uniform sampler2D cameraImage;
uniform vec3 center;
uniform float scale;
out vec3 tint;
out vec3 point;
out vec2 uv;
out float scanGlow;
void main() {
    point=position;
    vec3 displayed=position;
    scanGlow=0.0;
    gl_PointSize=1.0;
    tint=color;
    uv=color.xy;
    if (previewPoints) {
        ivec2 size=textureSize(cameraImage,0);
        uv=(vec2(gl_VertexID%size.x,gl_VertexID/size.x)+0.5)/vec2(size);
        if (processingPreview) {
            // A 600 ms scan, with a small display-only lift in camera-space Z.
            float scan=fract(processingTime/0.6)*1.24-0.12;
            scanGlow=1.0-smoothstep(0.0,0.10,abs(uv.x-scan));
            displayed.z+=0.008*scanGlow;
        }
    }
    vec3 world=(cameraToWorld*vec4(displayed,1.0)).xyz;
    vec3 p=cameraSpace ? (world-center)/scale*vec3(1.0,-1.0,-1.0) : displayed;
    // Invalid slots have zero camera-space Z. Valid world-space Z may be negative.
    gl_Position=clipInvalid && position.z<=0.0 ? vec4(2.0,2.0,2.0,1.0) : matrix*vec4(p,1.0);
})";
        const char *fragment = R"(
#version 460 core
in vec3 tint;
in vec3 point;
in vec2 uv;
in float scanGlow;
layout(location = 0) out vec4 fragmentColor;
uniform bool shaded;
uniform bool wireframe;
uniform bool textured;
uniform sampler2D cameraImage;
void main() {
    if (textured) {
        fragmentColor=texture(cameraImage,uv);
        fragmentColor.rgb=mix(fragmentColor.rgb,vec3(0.65,0.88,1.0),0.35*scanGlow);
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
        // Generate camera-space XYZ once on the GPU. Drawing and color updates
        // reuse this buffer until the image grid or calibration changes.
        const char *preview_compute = R"(
#version 460 core
layout(local_size_x=16,local_size_y=16) in;
layout(std430,binding=0) writeonly buffer PreviewPositions { vec4 positions[]; };
uniform ivec2 imageSize;
uniform vec2 calibrationSize;
uniform vec4 intrinsics; // fx, fy, cx, cy
uniform float depth;
void main() {
    ivec2 pixel=ivec2(gl_GlobalInvocationID.xy);
    if (any(greaterThanEqual(pixel,imageSize))) return;
    vec2 uv=(vec2(pixel)+0.5)*calibrationSize/vec2(imageSize)-0.5;
    vec2 xy=(uv-intrinsics.zw)*depth/intrinsics.xy;
    positions[pixel.y*imageSize.x+pixel.x]=vec4(xy,depth,1.0);
})";
        preview_program_ = std::make_unique<QOpenGLShaderProgram>();
        if (!preview_program_->addShaderFromSourceCode(QOpenGLShader::Compute, preview_compute) ||
            !preview_program_->link()) {
            failInitialization(QString("Unable to compile/link the preview projection shader: %1").arg(preview_program_->log()));
            return;
        }
        if (!buffer_.create() || !vao_.create() || !camera_buffer_.create() || !camera_vao_.create() ||
            !preview_buffer_.create() || !preview_vao_.create()) {
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
        preview_vao_.bind(); preview_buffer_.bind();
        gl_->glEnableVertexAttribArray(0);
        gl_->glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 4 * sizeof(float), nullptr);
        preview_vao_.release(); preview_buffer_.release();
        for (auto& texture : textures_) texture=createTexture();
        ready_ = true;
        camera_dirty_ = true; image_dirty_.fill(true);
        preview_geometry_dirty_ = true;
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
        // Switch directly to the completed cloud once its VBO handoff succeeds.
        // Do not wait for another animation cycle or worker completion signal.
        if (processing_ && announce_gpu_ && gpu_mesh_) setProcessing(false);
        if (camera_dirty_) uploadCamera();
        for (int camera = 0; camera < (right_camera_visible_ ? 2 : 1); ++camera)
            if (image_dirty_[camera]) uploadImage(camera);
        if ((live_preview_ || processing_) && preview_geometry_dirty_) {
            try { uploadPreviewPoints(); }
            catch (const std::exception& e) {
                failed_ = true;
                emit owner_->renderFailed(QString("Preview projection failed: %1").arg(e.what()));
                return;
            }
        }
        QMatrix4x4 projection;
        const float aspect = float(width()) / std::max(1, height());
        // Capture must keep the physical viewpoint as well as the orbit angles:
        // changing the fitted distance would otherwise move the camera.
        const float half_angle = std::atan(
            std::tan(20.f * float(CV_PI) / 180.f) * std::min(1.f, aspect));
        if (!preserve_view_) view_distance_ =
            camera_count_ > 0
                ? std::max(hasMesh() ? 3.2f : 0.f, camera_radius_ / std::sin(half_angle) * 1.1f)
                : 3.2f;
        const float distance = view_distance_;
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
        program_->setUniformValue("previewPoints",false);
        program_->setUniformValue("processingPreview",false);
        // Captures have independent VBOs. Live images never upload these again.
        program_->setUniformValue("cameraSpace",true);
        program_->setUniformValue("clipInvalid",true);
        program_->setUniformValue("shaded",false);
        program_->setUniformValue("wireframe",false);
        for (const auto& capture:cloud_resources_) {
            if (isMACloud(capture->frame)!=show_ma_clouds_) continue;
            if (!show_ma_clouds_ && capture->frame==hidden_capture_) continue;
            program_->setUniformValue("cameraToWorld",cameraPose(capture->frame));
            capture->vao.bind();
            gl_->glDrawArrays(GL_POINTS,0,capture->count);
            capture->vao.release();
        }
        program_->setUniformValue("cameraToWorld",cameraPose(gpu_mesh_ ? gpu_mesh_ : hidden_capture_));
        program_->setUniformValue("clipInvalid",bool(gpu_mesh_));
        program_->setUniformValue("shaded",bool(gpu_mesh_) && mode_==1);
        program_->setUniformValue("wireframe",bool(gpu_mesh_) && mode_==2);
        vao_.bind();
        if (gpu_mesh_ && mode_==2) gl_->glPolygonMode(GL_FRONT_AND_BACK,GL_LINE);
        if (!show_ma_clouds_)
            gl_->glDrawArrays(mode_==0 ? GL_POINTS : (mode_==2 && !gpu_mesh_ ? GL_LINES : GL_TRIANGLES),0,count_);
        if (gpu_mesh_ && mode_==2) gl_->glPolygonMode(GL_FRONT_AND_BACK,GL_FILL);
        vao_.release();
        if ((live_preview_ || processing_) && preview_count_ > 0 && !displayCameraImage(0).isNull() && texture_sizes_[0] == preview_size_) {
            program_->setUniformValue("cameraToWorld",QMatrix4x4());
            program_->setUniformValue("clipInvalid",true);
            program_->setUniformValue("shaded",false);
            program_->setUniformValue("wireframe",false);
            program_->setUniformValue("textured",true);
            program_->setUniformValue("previewPoints",true);
            program_->setUniformValue("processingPreview",processing_);
            program_->setUniformValue("processingTime",processing_ ? float(animation_clock_.elapsed()%600)/1000.f : 0.f);
            program_->setUniformValue("cameraImage",0);
            preview_vao_.bind();
            // Only the left image supplies the fixed-depth preview points.
            gl_->glActiveTexture(GL_TEXTURE0);
            gl_->glBindTexture(GL_TEXTURE_2D,textures_[0]);
            gl_->glDrawArrays(GL_POINTS,0,preview_count_);
            preview_vao_.release();
            gl_->glBindTexture(GL_TEXTURE_2D,0);
            program_->setUniformValue("previewPoints",false);
            program_->setUniformValue("processingPreview",false);
            program_->setUniformValue("textured",false);
        }
        if (camera_count_>0) {
            camera_vao_.bind();
            program_->setUniformValue("cameraSpace",false);
            program_->setUniformValue("clipInvalid",false);
            program_->setUniformValue("shaded",false);
            program_->setUniformValue("wireframe",false);
            const auto draw_image=[&](GLuint texture,int first) {
                gl_->glActiveTexture(GL_TEXTURE0);
                gl_->glBindTexture(GL_TEXTURE_2D,texture);
                program_->setUniformValue("cameraImage",0);
                program_->setUniformValue("textured",true);
                gl_->glEnable(GL_POLYGON_OFFSET_FILL);
                gl_->glPolygonOffset(1.f,1.f);
                gl_->glDrawArrays(GL_TRIANGLES,first,6);
                gl_->glDisable(GL_POLYGON_OFFSET_FILL);
                gl_->glBindTexture(GL_TEXTURE_2D,0);
                program_->setUniformValue("textured",false);
            };
            for (int camera = 0; camera < camera_image_count_; ++camera)
                if (!displayCameraImage(camera).isNull()) draw_image(textures_[camera],camera_count_+camera*6);
            for (const auto& texture:predicted_textures_)
                if (texture.first_vertex>=0) draw_image(texture.id,texture.first_vertex);
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
    struct CameraTexture {
        GLuint id{0};
        QSize size;
        qint64 image_key{0};
        int first_vertex{-1};
    };
    bool hasMesh() const {
        return show_ma_clouds_ ? ma_clouds_ && !ma_clouds_->empty()
            : !captured_clouds_.empty() || gpu_mesh_ || (mesh_ && !mesh_->vertices.empty());
    }
    bool isMACloud(const SharedGPUMesh& mesh) const {
        return ma_clouds_ && std::find(ma_clouds_->begin(),ma_clouds_->end(),mesh)!=ma_clouds_->end();
    }
    bool isCaptured(const SharedGPUMesh& mesh) const {
        return std::find(captured_clouds_.begin(),captured_clouds_.end(),mesh)!=captured_clouds_.end();
    }
    QMatrix4x4 cameraPose(const SharedGPUMesh& mesh) const {
        QMatrix4x4 pose;
        if (!mesh || !mesh->image_id || !predicted_cameras_) return pose;
        const auto found=std::find_if(predicted_cameras_->begin(),predicted_cameras_->end(),
            [&](const auto& frame) { return frame.image_id==mesh->image_id; });
        if (found==predicted_cameras_->end()) return pose;
        // Use MA's original world frame for every view, including the first.
        // FS vertices are already in metres; only rotation and translation apply.
        for (int row=0;row<4;++row) for (int col=0;col<4;++col)
            pose(row,col)=float(found->camera_to_world(row,col));
        return pose;
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
            preview_vao_.destroy(); preview_buffer_.destroy();
            preview_count_ = 0; preview_geometry_dirty_ = true;
            if (gl_) {
                gl_->glDeleteTextures(int(textures_.size()), textures_.data());
                for (const auto& texture:predicted_textures_) gl_->glDeleteTextures(1,&texture.id);
            }
            predicted_textures_.clear();
            textures_ = {}; texture_sizes_ = {};
            uploaded_image_keys_ = {};
            program_.reset();
            preview_program_.reset();
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
        // A new depth cloud must not recenter or rescale the captured scene.
        if (preserve_view_) return;
        cv::Vec3f low(std::numeric_limits<float>::max(),std::numeric_limits<float>::max(),std::numeric_limits<float>::max());
        cv::Vec3f high(-low);
        bool any=false;
        const auto add=[&](const cv::Vec3f& p) {
            any=true;
            for (int k=0;k<3;++k) { low[k]=std::min(low[k],p[k]); high[k]=std::max(high[k],p[k]); }
        };
        const auto add_gpu=[&](const SharedGPUMesh& mesh) {
            if (!mesh || !(mesh->stats.point_count || mesh->stats.triangle_count)) return;
            const auto& s=mesh->stats;
            const auto pose=cameraPose(mesh);
            // A rotated local bounding box needs all eight corners.
            for (int corner=0;corner<8;++corner) {
                const auto p=pose.map(QVector3D(corner&1 ? s.high[0] : s.low[0],
                    corner&2 ? s.high[1] : s.low[1],corner&4 ? s.high[2] : s.low[2]));
                add({p.x(),p.y(),p.z()});
            }
        };
        if (show_ma_clouds_) {
            if (ma_clouds_) for (const auto& cloud:*ma_clouds_) add_gpu(cloud);
        } else {
            for (const auto& cloud:captured_clouds_) if (cloud!=hidden_capture_) add_gpu(cloud);
            add_gpu(gpu_mesh_);
        }
        if (!show_ma_clouds_ && mesh_) {
            const auto pose=cameraPose(hidden_capture_);
            for (const auto& local:mesh_->vertices) {
                const auto p=pose.map(QVector3D(local[0],local[1],local[2]));
                add({p.x(),p.y(),p.z()});
            }
        }
        if (!any) {
            center_={}; scale_=1;
            const auto camera=activeCamera();
            if (validCamera(camera)) {
                const auto& k=camera->intrinsics;
                const float z=(live_preview_ || processing_) ? kPreviewDepth : .06f;
                const float width=float(camera->image_size.width*z/k(0,0));
                const float height=float(camera->image_size.height*z/k(1,1));
                center_=QVector3D(float((camera->image_size.width*.5-k(0,2))*z/k(0,0)),
                                  float((camera->image_size.height*.5-k(1,2))*z/k(1,1)),z*.8f);
                if (right_camera_visible_ && !live_preview_ && !processing_) center_.setX(center_.x()+width*.6f);
                scale_=std::max({width,height,z*1.6f});
            }
            return;
        }
        const auto center=(low+high)*.5f;
        center_=QVector3D(center[0],center[1],center[2]);
        scale_=std::max({high[0]-low[0],high[1]-low[1],high[2]-low[2],1e-6f});
    }
    void uploadCaptures() {
        for (auto it=cloud_resources_.begin();it!=cloud_resources_.end();) {
            if (isCaptured((*it)->frame) || isMACloud((*it)->frame)) { ++it; continue; }
            destroyCapture(**it); it=cloud_resources_.erase(it);
        }
        auto clouds=captured_clouds_;
        if (ma_clouds_) clouds.insert(clouds.end(),ma_clouds_->begin(),ma_clouds_->end());
        for (const auto& frame:clouds) {
            if (!frame || !frame->stats.point_count) continue;
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
        std::vector<float> data;
        data.reserve(mode_ == 0
                         ? mesh_->vertices.size() * 6
                         : mesh_->triangles.size() * (mode_ == 2 ? 36 : 18));
        const auto add = [&](int id, float light) {
            const auto& p = mesh_->vertices[id];
            const auto c = mode_ == 2 ? cv::Vec3b(0, 0, 0) : mesh_->colors[id];
            data.insert(data.end(),
                        {p[0], p[1], p[2], c[0] / 255.f * light,
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
    GLuint createTexture() {
        GLuint texture=0;
        gl_->glGenTextures(1,&texture);
        gl_->glBindTexture(GL_TEXTURE_2D,texture);
        gl_->glTexParameteri(GL_TEXTURE_2D,GL_TEXTURE_MIN_FILTER,GL_LINEAR);
        gl_->glTexParameteri(GL_TEXTURE_2D,GL_TEXTURE_MAG_FILTER,GL_LINEAR);
        gl_->glTexParameteri(GL_TEXTURE_2D,GL_TEXTURE_WRAP_S,GL_CLAMP_TO_EDGE);
        gl_->glTexParameteri(GL_TEXTURE_2D,GL_TEXTURE_WRAP_T,GL_CLAMP_TO_EDGE);
        gl_->glBindTexture(GL_TEXTURE_2D,0);
        return texture;
    }
    void uploadTexture(GLuint texture,const QImage& source,QSize& size,qint64& image_key) {
        if (source.isNull() || image_key == source.cacheKey()) return;
        const auto image = source.convertToFormat(QImage::Format_RGBA8888);
        gl_->glActiveTexture(GL_TEXTURE0);
        gl_->glBindTexture(GL_TEXTURE_2D, texture);
        gl_->glPixelStorei(GL_UNPACK_ALIGNMENT, 4);
        gl_->glPixelStorei(GL_UNPACK_ROW_LENGTH, image.bytesPerLine() / 4);
        if (size != image.size()) {
            gl_->glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, image.width(), image.height(),
                              0, GL_RGBA, GL_UNSIGNED_BYTE, image.constBits());
            size = image.size();
        } else gl_->glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, image.width(), image.height(),
                                    GL_RGBA, GL_UNSIGNED_BYTE, image.constBits());
        gl_->glBindTexture(GL_TEXTURE_2D, 0);
        gl_->glPixelStorei(GL_UNPACK_ROW_LENGTH, 0);
        image_key = source.cacheKey();
    }
    void uploadImage(int camera) {
        image_dirty_[camera] = false;
        uploadTexture(textures_[camera],displayCameraImage(camera),texture_sizes_[camera],uploaded_image_keys_[camera]);
    }
    const QImage& displayCameraImage(int camera) const {
        return processing_ ? processing_images_[camera] : camera_images_[camera];
    }
    std::optional<fs::MeshCamera> activeCamera() const {
        if (processing_) return processing_camera_;
        return camera_ ? camera_ : gpu_mesh_ ? std::optional<fs::MeshCamera>(gpu_mesh_->camera)
                                            : mesh_ ? mesh_->camera : std::nullopt;
    }
    static bool validCamera(const std::optional<fs::MeshCamera>& camera) {
        if (!camera || camera->image_size.width<=0 || camera->image_size.height<=0) return false;
        const auto& k=camera->intrinsics;
        return std::isfinite(k(0,0)) && k(0,0)>0 && std::isfinite(k(1,1)) && k(1,1)>0 &&
               std::isfinite(k(0,2)) && std::isfinite(k(1,2));
    }
    void uploadPreviewPoints() {
        const auto camera=activeCamera();
        if (!validCamera(camera)) { preview_count_=0; return; }
        if (displayCameraImage(0).isNull()) return;
        const auto size=displayCameraImage(0).size();
        // Retake / capture-more and frozen image-mode changes can revisit the
        // same grid. Keep its XYZ rather than dispatching the projection again.
        if (preview_count_>0 && preview_size_==size && preview_camera_ &&
            preview_camera_->image_size==camera->image_size && preview_camera_->intrinsics==camera->intrinsics) {
            preview_geometry_dirty_=false;
            return;
        }
        preview_count_=0;
        const auto count=static_cast<size_t>(size.width())*size.height();
        if (count>static_cast<size_t>(std::numeric_limits<int>::max())/(4*sizeof(float)))
            throw std::runtime_error("Preview image exceeds the OpenGL vertex buffer limit");
        const auto& k=camera->intrinsics;
        preview_buffer_.bind();
        preview_buffer_.allocate(nullptr,int(count*4*sizeof(float)));
        preview_buffer_.release();
        gl_->glBindBufferBase(GL_SHADER_STORAGE_BUFFER,0,preview_buffer_.bufferId());
        preview_program_->bind();
        gl_->glUniform2i(preview_program_->uniformLocation("imageSize"),size.width(),size.height());
        preview_program_->setUniformValue("calibrationSize",QVector2D(camera->image_size.width,camera->image_size.height));
        preview_program_->setUniformValue("intrinsics",QVector4D(k(0,0),k(1,1),k(0,2),k(1,2)));
        preview_program_->setUniformValue("depth",kPreviewDepth);
        gl_->glDispatchCompute((size.width()+15)/16,(size.height()+15)/16,1);
        gl_->glMemoryBarrier(GL_VERTEX_ATTRIB_ARRAY_BARRIER_BIT);
        preview_program_->release();
        gl_->glBindBufferBase(GL_SHADER_STORAGE_BUFFER,0,0);
        if (gl_->glGetError()!=GL_NO_ERROR)
            throw std::runtime_error("Unable to allocate/project the preview point grid");
        preview_size_=size;
        preview_camera_=camera;
        preview_count_=int(count);
        preview_geometry_dirty_=false;
        emit owner_->log(QString("GPU preview: cached %1 × %2 points at 0.40 m; subsequent frames update color only.")
                             .arg(size.width()).arg(size.height()));
    }
    void uploadCamera() {
        // The live frustum has its own buffer: new images never re-upload mesh
        // geometry or touch CUDA/OpenGL interop resources.
        const auto metadata = activeCamera();
        const cv::Vec3f center(center_.x(), center_.y(), center_.z());
        auto data = cameraData(metadata, center, scale_,camera_count_,camera_radius_,
                               cv::Matx44d::eye(),cv::Vec3f(1.f,.65f,.12f),right_camera_visible_ ? 2 : 1);
        camera_image_count_ = (int(data.size()/6) - camera_count_) / 6;
        // The original image remains on the short camera frustum. Also fit the
        // separate 40 cm point plane, without extending/moving that image.
        if ((live_preview_ || processing_) && validCamera(metadata)) {
            const auto& k=metadata->intrinsics;
            const auto size=metadata->image_size;
            for (int u : {0,size.width}) for (int v : {0,size.height}) {
                const cv::Vec3f world(float((u-k(0,2))*kPreviewDepth/k(0,0)),
                                     float((v-k(1,2))*kPreviewDepth/k(1,1)),kPreviewDepth);
                camera_radius_=std::max(camera_radius_,float(cv::norm((world-center)/scale_)));
            }
        }
        // Retain immutable MA-sized textures when only poses change. Clean
        // releases the whole set; retakes replace only the corresponding image.
        const auto texture_count=predicted_cameras_ ? predicted_cameras_->size() : 0;
        while (predicted_textures_.size()>texture_count) {
            gl_->glDeleteTextures(1,&predicted_textures_.back().id);
            predicted_textures_.pop_back();
        }
        while (predicted_textures_.size()<texture_count) {
            CameraTexture texture; texture.id=createTexture();
            predicted_textures_.push_back(texture);
        }
        // Keep all lines before the active image planes, then append one image
        // plane per MA camera. Both use the same original camera-to-world pose.
        std::vector<float> lines, images;
        if (predicted_cameras_) {
            for (size_t i=0;i<predicted_cameras_->size();++i) {
                const auto& frame=(*predicted_cameras_)[i];
                auto& texture=predicted_textures_[i];
                int count=0; float radius=0;
                const auto predicted=cameraData(frame.camera,center,scale_,count,radius,
                    frame.camera_to_world,cv::Vec3f(.10f,.45f,1.f),frame.image.isNull() ? 0 : 1);
                lines.insert(lines.end(),predicted.begin(),predicted.begin()+count*6);
                texture.first_vertex=-1;
                if (int(predicted.size())>count*6) {
                    texture.first_vertex=int(images.size()/6);
                    images.insert(images.end(),predicted.begin()+count*6,predicted.end());
                    uploadTexture(texture.id,frame.image,texture.size,texture.image_key);
                }
                camera_radius_=std::max(camera_radius_,radius);
            }
        } else {
            for (const auto& frame:captured_clouds_) {
                if (!frame) continue;
                int count=0; float radius=0;
                const auto captured=cameraData(frame->camera,center,scale_,count,radius,
                    cv::Matx44d::eye(),cv::Vec3f(1.f,.65f,.12f),0);
                lines.insert(lines.end(),captured.begin(),captured.begin()+count*6);
                camera_radius_=std::max(camera_radius_,radius);
            }
        }
        data.insert(data.begin(),lines.begin(),lines.end());
        camera_count_+=int(lines.size()/6);
        for (auto& texture:predicted_textures_)
            if (texture.first_vertex>=0) texture.first_vertex+=int(data.size()/6);
        data.insert(data.end(),images.begin(),images.end());
        camera_buffer_.bind();
        camera_buffer_.allocate(data.data(), int(data.size() * sizeof(float)));
        camera_buffer_.release();
        camera_dirty_ = false;
    }
    std::vector<float> cameraData(const std::optional<fs::MeshCamera>& metadata, cv::Vec3f center, float scale,
                                  int& line_count, float& radius,
                                  const cv::Matx44d& pose = cv::Matx44d::eye(),
                                  const cv::Vec3f& color = cv::Vec3f(1.f,.65f,.12f), int image_count = 1) {
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
                // Keep the original short frustum and textured image near the
                // camera, independently of the live point plane at 40 cm.
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
                if (!hasMesh()) radius = 0;
                const auto to_world = [&](cv::Vec3f local) {
                    const auto world = pose * cv::Vec4d(local[0],local[1],local[2],1);
                    return cv::Vec3f(float(world[0]),float(world[1]),float(world[2]));
                };
                const auto add_camera = [&](cv::Vec3f local) {
                    const auto world = to_world(local);
                    const auto p = (world - center) / scale;
                    radius = std::max(radius, float(cv::norm(p)));
                    data.insert(data.end(),
                                {p[0], -p[1], -p[2], color[0], color[1], color[2]});
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
                if (!image_count) return data;
                // QImage row zero is the image top. Upload unchanged and give
                // the upper frustum corners v=0 to avoid a vertical flip.
                const float uv[][2] = {{0,0}, {1,0}, {1,1}, {0,1}};
                for (int camera=0; camera<image_count; ++camera)
                    for (int i : {0,1,2,0,2,3}) {
                        const auto world = to_world(corners[i] + cv::Vec3f(camera * right_offset,0,0));
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
    SharedGPUClouds ma_clouds_;
    bool show_ma_clouds_{false};
    SharedPredictedCameras predicted_cameras_;
    std::vector<CameraTexture> predicted_textures_;
    SharedGPUMesh hidden_capture_;
    std::vector<std::unique_ptr<CloudResource>> cloud_resources_;
    bool captures_dirty_{true};
    cudaGraphicsResource* interop_{nullptr};
    int interop_device_{0};
    QVector3D center_;
    float scale_{1};
    float view_distance_{3.2f};
    bool preserve_view_{false};
    bool failed_{false}, announce_gpu_{false};
    bool ready_{false};
    QOpenGLFunctions_4_5_Core *gl_{nullptr}; // Owned by the current QOpenGLContext.
    QLabel *error_label_{nullptr};
    QOpenGLBuffer buffer_;
    QOpenGLVertexArrayObject vao_;
    QOpenGLBuffer camera_buffer_{QOpenGLBuffer::VertexBuffer};
    QOpenGLVertexArrayObject camera_vao_;
    static constexpr float kPreviewDepth = .40f; // Optical-axis depth, in metres.
    QOpenGLBuffer preview_buffer_{QOpenGLBuffer::VertexBuffer};
    QOpenGLVertexArrayObject preview_vao_;
    std::unique_ptr<QOpenGLShaderProgram> preview_program_;
    std::optional<fs::MeshCamera> preview_camera_;
    QSize preview_size_;
    int preview_count_{};
    bool live_preview_{false}, preview_geometry_dirty_{true};
    bool processing_{false};
    QTimer animation_timer_;
    QElapsedTimer animation_clock_;
    std::optional<fs::MeshCamera> processing_camera_;
    std::array<QImage, 2> processing_images_;
    std::optional<fs::MeshCamera> camera_;
    std::array<QImage, 2> camera_images_;
    std::array<GLuint, 2> textures_{};
    std::array<QSize, 2> texture_sizes_;
    std::array<qint64, 2> uploaded_image_keys_{};
    std::array<bool, 2> image_dirty_{{true, true}};
    bool camera_dirty_{true};
    std::unique_ptr<QOpenGLShaderProgram> program_;
    std::shared_ptr<const fs::MeshResult> mesh_;
    bool dirty_{true}, right_camera_visible_{false};
    int mode_{1}, count_{}, camera_count_{}, camera_image_count_{};
    float camera_radius_{};
    // Behind the camera, above and to its right, looking into the scene.
    static constexpr float kCaptureYaw = -33.6f, kCapturePitch = 40.f;
    float yaw_{kCaptureYaw}, pitch_{kCapturePitch}, zoom_{1};
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
void MeshView::setPredictedCameras(SharedPredictedCameras cameras) {
    if (canvas_) canvas_->setPredictedCameras(std::move(cameras));
}
void MeshView::setMAClouds(SharedGPUClouds clouds) {
    if (canvas_) canvas_->setMAClouds(std::move(clouds));
}
void MeshView::setShowMAClouds(bool enabled) {
    if (canvas_) canvas_->setShowMAClouds(enabled);
}
void MeshView::setMode(int mode) {
    if (canvas_)
        canvas_->setMode(mode);
}
void MeshView::resetView() {
    if (canvas_)
        canvas_->resetView();
}

void MeshView::resetCaptureView() {
    if (canvas_) canvas_->resetCaptureView();
}

void MeshView::preserveView() {
    if (canvas_) canvas_->preserveView();
}

void MeshView::setProcessing(bool processing) {
    if (canvas_) canvas_->setProcessing(processing);
}

void MeshView::setRightCameraVisible(bool visible) {
    if (canvas_)
        canvas_->setRightCameraVisible(visible);
}

void MeshView::setCamera(const std::optional<fs::MeshCamera>& camera) {
    if (canvas_) canvas_->setCamera(camera);
}

void MeshView::setLivePreview(bool enabled) {
    if (canvas_) canvas_->setLivePreview(enabled);
}

void MeshView::setCameraImage(QImage image) {
    if (canvas_) canvas_->setCameraImage(std::move(image));
}

void MeshView::setRightCameraImage(QImage image) {
    if (canvas_) canvas_->setCameraImage(std::move(image), 1);
}
