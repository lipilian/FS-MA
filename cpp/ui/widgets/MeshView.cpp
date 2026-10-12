#include "MeshView.hpp"
#include <QGuiApplication>
#include <QApplication>
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
#include <QPainter>
#include <QPainterPath>
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
        setMouseTracking(true);
    }
    ~MeshCanvas() override { cleanup(); }
    void setGPUMesh(SharedGPUMesh mesh) {
        const bool had_mesh=hasMesh();
        gpu_mesh_=std::move(mesh);
        if (gpu_mesh_ && gpu_mesh_->point_cloud) mode_=0;
        dirty_=true; failed_=false; announce_gpu_=bool(gpu_mesh_);
        if (gpu_mesh_ && !had_mesh && !preserve_view_) resetView(); else update();
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
        mode_ = show_ma_clouds_ ? 0 : mode;
        dirty_=true;
        update();
    }
    void setMAClouds(SharedGPUClouds clouds) {
        if (ma_clouds_==clouds) return;
        ma_clouds_=std::move(clouds);
        captures_dirty_=true; dirty_=true; failed_=false;
        announce_ma_=ma_clouds_ && !ma_clouds_->empty();
        update();
    }
    void setShowMAClouds(bool enabled) {
        if (show_ma_clouds_==enabled) return;
        show_ma_clouds_=enabled;
        if (enabled) mode_=0;
        // Keep identical framing when comparing both reconstructions.
        preserveView();
        dirty_=true;
        update();
    }
    void setCameraPresentation(quint64 active, quint64 highlighted, quint64 hidden) {
        if (active_camera_id_==active && highlighted_camera_id_==highlighted && hidden_image_id_==hidden) return;
        active_camera_id_=active; highlighted_camera_id_=highlighted; hidden_image_id_=hidden;
        camera_dirty_=true; update();
    }
    void setPredictedCameras(SharedPredictedCameras cameras) {
        if (predicted_cameras_ == cameras) return;
        setHoveredCamera(0);
        pressed_camera_id_=0;
        camera_targets_.clear(); label_targets_.clear();
        predicted_cameras_ = std::move(cameras);
        camera_dirty_ = true;
        if (predicted_cameras_) preserveView();
        update();
    }
    void setCameraInteractionEnabled(bool enabled) {
        camera_interaction_enabled_=enabled;
        if (!enabled) { setHoveredCamera(0); pressed_camera_id_=0; }
    }
    void setCamera(const std::optional<fs::MeshCamera>& camera) {
        if (camera_.has_value() == camera.has_value() &&
            (!camera || (camera_->image_size == camera->image_size && camera_->intrinsics == camera->intrinsics))) return;
        camera_ = camera;
        camera_dirty_ = true;
        if (!hasMesh()) dirty_ = true;
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
        if (!hasMesh()) dirty_ = true;
        update();
    }
    void resetView() {
        if (!hasMesh()) { resetCaptureView(); return; }
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
    QPolygonF cameraImageQuad() const {
        if (!ready_ || failed_ || !camera_quad_valid_) return {};
        QPolygonF quad;
        for (const auto& corner : camera_image_corners_) {
            const auto clip = view_projection_ * QVector4D(corner, 1);
            if (clip.w() <= 0 || !std::isfinite(clip.w())) return {};
            const auto ndc = clip.toVector3DAffine();
            if (!std::isfinite(ndc.x()) || !std::isfinite(ndc.y()) || ndc.z() < -1 || ndc.z() > 1) return {};
            quad << QPointF((ndc.x() + 1) * width() * .5, (1 - ndc.y()) * height() * .5);
        }
        return quad;
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
uniform vec3 center;
uniform float scale;
out vec3 tint;
out vec3 point;
out vec2 uv;
void main() {
    point=position;
    gl_PointSize=1.0;
    tint=color;
    uv=color.xy;
    vec3 world=(cameraToWorld*vec4(position,1.0)).xyz;
    vec3 p=cameraSpace ? (world-center)/scale*vec3(1.0,-1.0,-1.0) : position;
    // Invalid slots have zero camera-space Z. Valid world-space Z may be negative.
    gl_Position=clipInvalid && position.z<=0.0 ? vec4(2.0,2.0,2.0,1.0) : matrix*vec4(p,1.0);
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
uniform bool hoveredCloud;
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
    vec3 color=wireframe ? vec3(0.0) : tint*light;
    if (hoveredCloud) color=mix(color,vec3(0.82,0.68,1.0),0.85);
    fragmentColor=vec4(color,1.0);
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
        for (auto& texture : textures_) texture=createTexture();
        ready_ = true;
        camera_dirty_ = true; image_dirty_.fill(true);
        captures_dirty_=true;
        dirty_ = true; failed_=false; announce_gpu_=bool(gpu_mesh_);
        announce_ma_=ma_clouds_ && !ma_clouds_->empty();
        emit owner_->log(QString("3D renderer: OpenGL %1 (Core), %2")
                             .arg(QString::fromLatin1(reinterpret_cast<const char *>(gl_->glGetString(GL_VERSION))))
                             .arg(QString::fromLatin1(reinterpret_cast<const char *>(gl_->glGetString(GL_RENDERER)))));
    }
    void resizeGL(int width, int height) override {
        if (gl_) gl_->glViewport(0, 0, width, height);
    }
    void paintGL() override {
        if (!gl_) return;
        // QPainter's label overlay changes GL state between frames.
        gl_->glEnable(GL_DEPTH_TEST);
        gl_->glDepthFunc(GL_LESS);
        gl_->glDepthMask(GL_TRUE);
        gl_->glDisable(GL_BLEND);
        gl_->glDisable(GL_SCISSOR_TEST);
        gl_->glClearColor(.067f, .11f, .17f, 1);
        gl_->glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
        if (!ready_ || failed_) return;
        if (dirty_) {
            try {
                if (captures_dirty_) uploadCaptures();
                updateBounds();
                if (gpu_mesh_ && !isCaptured(gpu_mesh_)) uploadGPU();
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
        // Keep the fitted viewpoint stable while adding captures.
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
        view_projection_ = matrix;
        program_->bind();
        program_->setUniformValue("matrix", matrix);
        program_->setUniformValue("center",center_);
        program_->setUniformValue("scale",scale_);
        program_->setUniformValue("textured",false);
        program_->setUniformValue("hoveredCloud",false);
        // Captures have independent VBOs, retained while new views are acquired.
        program_->setUniformValue("cameraSpace",true);
        program_->setUniformValue("clipInvalid",true);
        program_->setUniformValue("shaded",false);
        program_->setUniformValue("wireframe",false);
        CloudResource* hovered_capture=nullptr;
        for (const auto& capture:cloud_resources_) {
            const bool ma_cloud=isMACloud(capture->frame);
            if (show_ma_clouds_ != ma_cloud) continue;
            if (!show_ma_clouds_ && capture->frame==hidden_capture_) continue;
            if (isHoveredFSCloud(capture->frame)) { hovered_capture=capture.get(); continue; }
            const bool surface=!ma_cloud && !capture->frame->point_cloud && mode_!=0;
            program_->setUniformValue("shaded",surface && mode_==1);
            program_->setUniformValue("wireframe",surface && mode_==2);
            program_->setUniformValue("cameraToWorld",cameraPose(capture->frame));
            capture->vao.bind();
            if (surface && mode_==2) gl_->glPolygonMode(GL_FRONT_AND_BACK,GL_LINE);
            gl_->glDrawArrays(surface ? GL_TRIANGLES : GL_POINTS,0,capture->count);
            if (surface && mode_==2) gl_->glPolygonMode(GL_FRONT_AND_BACK,GL_FILL);
            capture->vao.release();
        }
        program_->setUniformValue("cameraToWorld",cameraPose(gpu_mesh_));
        program_->setUniformValue("clipInvalid",true);
        program_->setUniformValue("shaded",mode_==1);
        program_->setUniformValue("wireframe",mode_==2);
        vao_.bind();
        if (gpu_mesh_ && mode_==2) gl_->glPolygonMode(GL_FRONT_AND_BACK,GL_LINE);
        const bool hovered_active=count_>0 && isHoveredFSCloud(gpu_mesh_);
        if (!show_ma_clouds_ && !hovered_active)
            gl_->glDrawArrays(mode_==0 ? GL_POINTS : GL_TRIANGLES,0,count_);
        if (gpu_mesh_ && mode_==2) gl_->glPolygonMode(GL_FRONT_AND_BACK,GL_FILL);
        vao_.release();
        if (hovered_capture || hovered_active) {
            // Draw the matching FS view last, including exactly coincident
            // points. Keep depth testing so nearer surfaces still occlude it.
            program_->setUniformValue("shaded",false);
            program_->setUniformValue("wireframe",false);
            program_->setUniformValue("hoveredCloud",true);
            program_->setUniformValue("cameraToWorld",cameraPose(hovered_capture ? hovered_capture->frame : gpu_mesh_));
            auto& hovered_vao=hovered_capture ? hovered_capture->vao : vao_;
            hovered_vao.bind();
            gl_->glDepthFunc(GL_LEQUAL);
            gl_->glDrawArrays(GL_POINTS,0,hovered_capture ? hovered_capture->count : count_);
            gl_->glDepthFunc(GL_LESS);
            hovered_vao.release();
            program_->setUniformValue("hoveredCloud",false);
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
                if (!camera_images_[camera].isNull()) draw_image(textures_[camera],camera_count_+camera*6);
            for (const auto& texture:predicted_textures_)
                if (texture.first_vertex>=0) draw_image(texture.id,texture.first_vertex);
            gl_->glDrawArrays(GL_LINES,0,camera_count_);
            if (highlight_count_>0) {
                // The active frame stays legible even when MA predicts nearly
                // overlapping camera poses or scene geometry lies in front.
                gl_->glDisable(GL_DEPTH_TEST);
                gl_->glDrawArrays(GL_LINES,highlight_first_,highlight_count_);
                gl_->glEnable(GL_DEPTH_TEST);
            }
            if (hover_count_>0) {
                gl_->glDisable(GL_DEPTH_TEST);
                gl_->glDrawArrays(GL_LINES,hover_first_,hover_count_);
                gl_->glEnable(GL_DEPTH_TEST);
            }
            camera_vao_.release();
        }
        program_->release();
        drawCameraLabels();
        if (announce_gpu_ && gpu_mesh_) {
            announce_gpu_=false;
            const auto error=gl_->glGetError();
            if (error!=GL_NO_ERROR) {
                failed_=true;
                emit owner_->renderFailed(QString("OpenGL mesh draw failed (error %1).").arg(error));
            } else emit owner_->gpuMeshPresented(gpu_mesh_);
        }
        if (announce_ma_ && show_ma_clouds_ && ma_clouds_) {
            announce_ma_=false;
            const auto error=gl_->glGetError();
            if (error!=GL_NO_ERROR) {
                failed_=true;
                emit owner_->renderFailed(QString("OpenGL MA point-cloud draw failed (error %1).").arg(error));
            } else emit owner_->maCloudsPresented(ma_clouds_);
        }
    }
    void mousePressEvent(QMouseEvent *e) override {
        last_=press_position_=e->position();
        dragging_=false;
        pressed_camera_id_=e->button()==Qt::LeftButton ? cameraAt(e->position()) : 0;
    }
    void mouseMoveEvent(QMouseEvent *e) override {
        if (!e->buttons()) { setHoveredCamera(cameraAt(e->position())); return; }
        if (!dragging_ && (e->position()-press_position_).manhattanLength()<QApplication::startDragDistance()) return;
        dragging_=true;
        setHoveredCamera(0);
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
    void mouseReleaseEvent(QMouseEvent *e) override {
        const auto id=cameraAt(e->position());
        const bool click=e->button()==Qt::LeftButton && !dragging_ && pressed_camera_id_ &&
            id==pressed_camera_id_ && (e->position()-press_position_).manhattanLength()<QApplication::startDragDistance();
        pressed_camera_id_=0;
        setHoveredCamera(id);
        if (click) emit owner_->cameraClicked(id);
    }
    void leaveEvent(QEvent *e) override {
        setHoveredCamera(0);
        QOpenGLWidget::leaveEvent(e);
    }
    void wheelEvent(QWheelEvent *e) override {
        setHoveredCamera(0);
        zoom_ = std::clamp(zoom_ * std::pow(1.15f, e->angleDelta().y() / 120.f),
                           .2f, 8.f);
        update();
        e->accept();
    }

  private:
    std::optional<QVector3D> projectCameraPoint(const QVector3D& point) const {
        const auto clip=view_projection_*QVector4D(point,1);
        if (!std::isfinite(clip.w()) || clip.w()<=0) return {};
        const auto ndc=clip.toVector3DAffine();
        if (!std::isfinite(ndc.x()) || !std::isfinite(ndc.y()) || !std::isfinite(ndc.z()) ||
            ndc.z()<-1 || ndc.z()>1) return {};
        return QVector3D((ndc.x()+1)*width()*.5,(1-ndc.y())*height()*.5,ndc.z());
    }
    quint64 cameraAt(QPointF position) const {
        if (!camera_interaction_enabled_ || !ready_ || failed_ || !predicted_cameras_ || predicted_cameras_->size()<2) return 0;
        // Labels are separated on screen, so even identical poses stay selectable.
        for (const auto& label:label_targets_)
            if (label.rect.contains(position)) return label.image_id;
        quint64 closest_line=0, closest_plane=0;
        qreal line_distance=36, plane_depth=std::numeric_limits<qreal>::max();
        for (const auto& target:camera_targets_) {
            for (size_t i=0;i+1<target.lines.size();i+=2) {
                const auto a=projectCameraPoint(target.lines[i]), b=projectCameraPoint(target.lines[i+1]);
                if (!a || !b) continue;
                const QPointF start(a->x(),a->y()), delta(b->x()-a->x(),b->y()-a->y());
                const auto length=QPointF::dotProduct(delta,delta);
                const auto t=length>0 ? std::clamp(QPointF::dotProduct(position-start,delta)/length,qreal(0),qreal(1)) : 0;
                const auto offset=position-start-t*delta;
                const auto distance=QPointF::dotProduct(offset,offset);
                if (distance<line_distance) { line_distance=distance; closest_line=target.image_id; }
            }
            QPolygonF quad;
            qreal depth=0;
            // Each frustum edge pair is origin->corner, corner->next corner.
            for (size_t i : {1,5,9,13}) {
                const auto corner=projectCameraPoint(target.lines[i]);
                if (!corner) break;
                quad<<QPointF(corner->x(),corner->y()); depth+=corner->z();
            }
            if (quad.size()==4 && quad.containsPoint(position,Qt::OddEvenFill) && depth<plane_depth) {
                plane_depth=depth; closest_plane=target.image_id;
            }
        }
        return closest_line ? closest_line : closest_plane;
    }
    void setHoveredCamera(quint64 id) {
        if (hovered_camera_id_==id) return;
        hovered_camera_id_=id;
        setCursor(id ? Qt::PointingHandCursor : Qt::ArrowCursor);
        camera_dirty_=true; update();
    }
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
            : !captured_clouds_.empty() || gpu_mesh_;
    }
    bool isMACloud(const SharedGPUMesh& mesh) const {
        return ma_clouds_ && std::find(ma_clouds_->begin(),ma_clouds_->end(),mesh)!=ma_clouds_->end();
    }
    bool isHoveredFSCloud(const SharedGPUMesh& mesh) const {
        return !show_ma_clouds_ && hovered_camera_id_ && mesh && mesh->point_cloud &&
            mesh->image_id==hovered_camera_id_ && !isMACloud(mesh);
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
            if (gl_) {
                gl_->glDeleteTextures(int(textures_.size()), textures_.data());
                for (const auto& texture:predicted_textures_) gl_->glDeleteTextures(1,&texture.id);
            }
            predicted_textures_.clear();
            textures_ = {}; texture_sizes_ = {};
            uploaded_image_keys_ = {};
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
            if (ma_clouds_) for (const auto& frame:*ma_clouds_) add_gpu(frame);
        } else {
            for (const auto& cloud:captured_clouds_) if (cloud!=hidden_capture_) add_gpu(cloud);
            add_gpu(gpu_mesh_);
        }
        if (!any) {
            center_={}; scale_=1;
            const auto camera=activeCamera();
            if (validCamera(camera)) {
                const auto& k=camera->intrinsics;
                const float z=.06f;
                const float width=float(camera->image_size.width*z/k(0,0));
                const float height=float(camera->image_size.height*z/k(1,1));
                center_=QVector3D(float((camera->image_size.width*.5-k(0,2))*z/k(0,0)),
                                  float((camera->image_size.height*.5-k(1,2))*z/k(1,1)),z*.8f);
                if (right_camera_visible_) center_.setX(center_.x()+width*.6f);
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
            if (!frame || !(frame->stats.point_count || frame->stats.triangle_count)) continue;
            const auto existing=std::find_if(cloud_resources_.begin(),cloud_resources_.end(),
                [&](const auto& resource) { return resource->frame==frame; });
            if (existing!=cloud_resources_.end()) continue;
            auto resource=std::make_unique<CloudResource>(); resource->frame=frame;
            try {
                if (!resource->buffer.create() || !resource->vao.create())
                    throw std::runtime_error("Unable to create captured geometry buffer/array");
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
        uploadTexture(textures_[camera],camera_images_[camera],texture_sizes_[camera],uploaded_image_keys_[camera]);
    }
    std::optional<fs::MeshCamera> activeCamera() const {
        return camera_ ? camera_ : gpu_mesh_ ? std::optional<fs::MeshCamera>(gpu_mesh_->camera) : std::nullopt;
    }
    static bool validCamera(const std::optional<fs::MeshCamera>& camera) {
        if (!camera || camera->image_size.width<=0 || camera->image_size.height<=0) return false;
        const auto& k=camera->intrinsics;
        return std::isfinite(k(0,0)) && k(0,0)>0 && std::isfinite(k(1,1)) && k(1,1)>0 &&
               std::isfinite(k(0,2)) && std::isfinite(k(1,2));
    }
    void uploadCamera() {
        // Camera frustums use a separate buffer from CUDA/OpenGL geometry.
        camera_labels_.clear();
        camera_targets_.clear();
        const bool predicted=predicted_cameras_ && !predicted_cameras_->empty();
        const auto metadata = predicted ? std::nullopt : activeCamera();
        const cv::Vec3f center(center_.x(), center_.y(), center_.z());
        const auto active=std::find_if(captured_clouds_.begin(),captured_clouds_.end(),
            [&](const auto& frame) { return frame && frame->image_id==active_camera_id_; });
        const int active_view=active==captured_clouds_.end() ? std::max(1,int(captured_clouds_.size()))
            : int(active-captured_clouds_.begin())+1;
        auto data = cameraData(metadata, center, scale_,camera_count_,camera_radius_,
                               cv::Matx44d::eye(),cameraColor(active_camera_id_),right_camera_visible_ ? 2 : 1,active_view);
        camera_image_count_ = (int(data.size()/6) - camera_count_) / 6;
        const int base_highlight=active_camera_id_ && active_camera_id_==highlighted_camera_id_ ? camera_count_ : 0;
        highlight_first_=0; highlight_count_=0;
        hover_first_=0; hover_count_=0;
        // Reuse the exact image-plane vertices supplied to OpenGL. Project only
        // these four corners on the CPU for the 2D capture transfer animation.
        camera_quad_valid_ = camera_image_count_ > 0;
        if (camera_quad_valid_) rememberCameraQuad(data,camera_count_);
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
        if (predicted) {
            for (size_t i=0;i<predicted_cameras_->size();++i) {
                const auto& frame=(*predicted_cameras_)[i];
                auto& texture=predicted_textures_[i];
                int count=0; float radius=0;
                const auto predicted=cameraData(frame.camera,center,scale_,count,radius,
                    frame.camera_to_world,cameraColor(frame.image_id),frame.image.isNull() ? 0 : 1,int(i)+1,frame.image_id);
                if (count>=16) {
                    CameraTarget target{frame.image_id,{}};
                    for (int j=0;j<count;++j)
                        target.lines.emplace_back(predicted[j*6],predicted[j*6+1],predicted[j*6+2]);
                    camera_targets_.push_back(std::move(target));
                }
                if (frame.image_id && frame.image_id==highlighted_camera_id_) {
                    highlight_first_=int(lines.size()/6); highlight_count_=count;
                }
                if (frame.image_id && frame.image_id==hovered_camera_id_) {
                    hover_first_=int(lines.size()/6); hover_count_=count;
                }
                lines.insert(lines.end(),predicted.begin(),predicted.begin()+count*6);
                texture.first_vertex=-1;
                if (int(predicted.size())>count*6) {
                    if (frame.image_id==active_camera_id_) { rememberCameraQuad(predicted,count); camera_quad_valid_=true; }
                    if (frame.image_id!=hidden_image_id_) texture.first_vertex=int(images.size()/6);
                    images.insert(images.end(),predicted.begin()+count*6,predicted.end());
                    uploadTexture(texture.id,frame.image,texture.size,texture.image_key);
                }
                camera_radius_=std::max(camera_radius_,radius);
            }
        } else {
            for (std::size_t i=0;i<captured_clouds_.size();++i) {
                const auto& frame=captured_clouds_[i];
                if (!frame) continue;
                if (metadata && frame->image_id==active_camera_id_) continue;
                int count=0; float radius=0;
                const auto captured=cameraData(frame->camera,center,scale_,count,radius,
                    cv::Matx44d::eye(),cameraColor(frame->image_id),0,int(i)+1);
                if (frame->image_id && frame->image_id==highlighted_camera_id_) {
                    highlight_first_=int(lines.size()/6); highlight_count_=count;
                }
                lines.insert(lines.end(),captured.begin(),captured.begin()+count*6);
                camera_radius_=std::max(camera_radius_,radius);
            }
        }
        data.insert(data.begin(),lines.begin(),lines.end());
        if (base_highlight) { highlight_first_=int(lines.size()/6); highlight_count_=base_highlight; }
        camera_count_+=int(lines.size()/6);
        for (auto& texture:predicted_textures_)
            if (texture.first_vertex>=0) texture.first_vertex+=int(data.size()/6);
        data.insert(data.end(),images.begin(),images.end());
        camera_buffer_.bind();
        camera_buffer_.allocate(data.data(), int(data.size() * sizeof(float)));
        camera_buffer_.release();
        camera_dirty_ = false;
    }
    cv::Vec3f cameraColor(quint64 id) const {
        if (id && id==hovered_camera_id_) return cv::Vec3f(.65f,.3f,1.f);
        return id && id==highlighted_camera_id_ ? cv::Vec3f(.1f,1.f,.25f) : cv::Vec3f(.1f,.45f,1.f);
    }
    void drawCameraLabels() {
        label_targets_.clear();
        if (camera_labels_.empty()) return;
        QPainter painter(this);
        painter.setRenderHint(QPainter::Antialiasing);
        QFont label_font=font(); label_font.setPixelSize(15); label_font.setWeight(QFont::DemiBold);
        const QFontMetricsF metrics(label_font);
        std::vector<QRectF> occupied;
        for (const auto& label:camera_labels_) {
            const auto clip=view_projection_*QVector4D(label.anchor,1);
            if (!std::isfinite(clip.w()) || clip.w()<=0) continue;
            const auto ndc=clip.toVector3DAffine();
            if (!std::isfinite(ndc.x()) || !std::isfinite(ndc.y()) || !std::isfinite(ndc.z()) ||
                ndc.z()<-1 || ndc.z()>1) continue;
            const QPointF anchor((ndc.x()+1)*width()*.5,(1-ndc.y())*height()*.5);
            const qreal text_width=metrics.horizontalAdvance(label.text);
            QRectF rect(anchor.x()-text_width*.5-3,anchor.y()-metrics.height()-3,text_width+6,metrics.height()+6);
            if (!rect.intersects(QRectF(this->rect()))) continue;
            // Nearby/overlapping camera poses still need distinct view numbers.
            while (std::any_of(occupied.begin(),occupied.end(),[&](const auto& other) { return rect.intersects(other); }))
                rect.translate(0,-rect.height()-2);
            occupied.push_back(rect);
            if (label.image_id) label_targets_.push_back({label.image_id,rect});
            QPainterPath text;
            text.addText(QPointF(rect.left()+3,rect.top()+3+metrics.ascent()),label_font,label.text);
            painter.strokePath(text,QPen(QColor(17,28,43),3,Qt::SolidLine,Qt::RoundCap,Qt::RoundJoin));
            painter.fillPath(text,label.color);
        }
    }
    void rememberCameraQuad(const std::vector<float>& data, int lines) {
        const int vertices[] = {0,1,2,5};
        for (int i=0;i<4;++i) {
            const auto offset=(lines+vertices[i])*6;
            camera_image_corners_[i]=QVector3D(data[offset],data[offset+1],data[offset+2]);
        }
    }
    std::vector<float> cameraData(const std::optional<fs::MeshCamera>& metadata, cv::Vec3f center, float scale,
                                  int& line_count, float& radius,
                                  const cv::Matx44d& pose = cv::Matx44d::eye(),
                                  const cv::Vec3f& color = cv::Vec3f(.1f,.45f,1.f), int image_count = 1, int view_number = 1,
                                  quint64 image_id = 0) {
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
                // Keep the captured image on a short frustum near its camera.
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
                // Replace the triangular up marker with a screen-facing label
                // anchored above the image plane. Use the capture slot, not ID,
                // so retakes retain their view number.
                const auto top = (corners[0] + corners[1]) * .5f;
                const auto up = top + cv::Vec3f(0, -z * .2f, 0);
                const auto anchor=(to_world(up)-center)/scale;
                radius=std::max(radius,float(cv::norm(anchor)));
                camera_labels_.push_back({QVector3D(anchor[0],-anchor[1],-anchor[2]),
                    QString("view %1").arg(view_number),QColor::fromRgbF(color[0],color[1],color[2]),image_id});
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
    bool announce_ma_{false};
    quint64 active_camera_id_{0}, highlighted_camera_id_{0}, hidden_image_id_{0};
    bool show_ma_clouds_{false};
    SharedPredictedCameras predicted_cameras_;
    struct CameraLabel { QVector3D anchor; QString text; QColor color; quint64 image_id; };
    std::vector<CameraLabel> camera_labels_;
    struct CameraTarget { quint64 image_id; std::vector<QVector3D> lines; };
    struct LabelTarget { quint64 image_id; QRectF rect; };
    std::vector<CameraTarget> camera_targets_;
    std::vector<LabelTarget> label_targets_;
    bool camera_interaction_enabled_{false}, dragging_{false};
    quint64 hovered_camera_id_{0}, pressed_camera_id_{0};
    QPointF press_position_;
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
    std::optional<fs::MeshCamera> camera_;
    std::array<QImage, 2> camera_images_;
    std::array<GLuint, 2> textures_{};
    std::array<QSize, 2> texture_sizes_;
    std::array<qint64, 2> uploaded_image_keys_{};
    std::array<bool, 2> image_dirty_{{true, true}};
    bool camera_dirty_{true};
    bool camera_quad_valid_{false};
    std::array<QVector3D,4> camera_image_corners_;
    QMatrix4x4 view_projection_;
    std::unique_ptr<QOpenGLShaderProgram> program_;
    bool dirty_{true}, right_camera_visible_{false};
    int mode_{1}, count_{}, camera_count_{}, camera_image_count_{};
    int highlight_first_{0}, highlight_count_{0};
    int hover_first_{0}, hover_count_{0};
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
void MeshView::setGPUMesh(SharedGPUMesh mesh) {
    if (canvas_) canvas_->setGPUMesh(std::move(mesh));
    else if (mesh) emit renderFailed("GPU mesh rendering requires a hardware OpenGL display session.");
}
void MeshView::setCapturedClouds(std::vector<SharedGPUMesh> clouds, SharedGPUMesh hidden) {
    if (canvas_) canvas_->setCapturedClouds(std::move(clouds),std::move(hidden));
}
void MeshView::setPredictedCameras(SharedPredictedCameras cameras) {
    if (canvas_) canvas_->setPredictedCameras(std::move(cameras));
}
void MeshView::setCameraInteractionEnabled(bool enabled) {
    if (canvas_) canvas_->setCameraInteractionEnabled(enabled);
}
void MeshView::setMAClouds(SharedGPUClouds clouds) {
    if (canvas_) canvas_->setMAClouds(std::move(clouds));
    else if (offscreen_ma_clouds_!=clouds) {
        offscreen_ma_clouds_=std::move(clouds);
        if (offscreen_ma_clouds_ && !offscreen_ma_clouds_->empty())
            emit renderFailed("MA point-cloud rendering requires a hardware OpenGL display session.");
    }
}
void MeshView::setCameraPresentation(quint64 active, quint64 highlighted, quint64 hidden) {
    if (canvas_) canvas_->setCameraPresentation(active,highlighted,hidden);
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

QPolygonF MeshView::cameraImageQuad() const {
    if (!canvas_) return {};
    return canvas_->cameraImageQuad().translated(canvas_->mapTo(this, QPoint()));
}
