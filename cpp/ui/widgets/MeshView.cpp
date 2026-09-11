#include "MeshView.hpp"
#include <QGuiApplication>
#include <QLabel>
#include <QMatrix4x4>
#include <QMouseEvent>
#include <QOpenGLBuffer>
#include <QOpenGLContext>
#include <QOpenGLFunctions>
#include <QOpenGLShaderProgram>
#include <QOpenGLWidget>
#include <QVBoxLayout>
#include <QWheelEvent>
#include <algorithm>
#include <cmath>

class MeshCanvas : public QOpenGLWidget, protected QOpenGLFunctions {
  public:
    explicit MeshCanvas(QWidget *parent)
        : QOpenGLWidget(parent), buffer_(QOpenGLBuffer::VertexBuffer) {
        setMinimumSize(260, 240);
        QSurfaceFormat requested;
        requested.setVersion(2, 1);
        requested.setDepthBufferSize(24);
        setFormat(requested);
    }
    ~MeshCanvas() override { cleanup(); }
    void setMesh(std::shared_ptr<const fs::MeshResult> mesh) {
        mesh_ = std::move(mesh);
        dirty_ = true;
        resetView();
    }
    void setMode(int mode) {
        mode_ = mode;
        dirty_ = true;
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
        initializeOpenGLFunctions();
        glEnable(GL_DEPTH_TEST);
        connect(
            context(), &QOpenGLContext::aboutToBeDestroyed, this,
            [this] { cleanup(); }, Qt::DirectConnection);
        program_ = std::make_unique<QOpenGLShaderProgram>();
        const char *vertex =
            "attribute vec3 position; attribute vec3 color; uniform mat4 "
            "matrix; varying vec3 tint; void main(){ "
            "gl_Position=matrix*vec4(position,1.0); tint=color;}";
        const char *fragment =
            "#ifdef GL_ES\nprecision mediump float;\n#endif\nvarying vec3 "
            "tint; void main(){gl_FragColor=vec4(tint,1.0);}";
        if (!program_->addShaderFromSourceCode(QOpenGLShader::Vertex, vertex) ||
            !program_->addShaderFromSourceCode(QOpenGLShader::Fragment,
                                               fragment) ||
            !program_->link()) {
            auto *error =
                new QLabel("Unable to initialize the 3D renderer", this);
            error->setStyleSheet("color: white;");
            error->adjustSize();
            error->show();
            program_.reset();
            return;
        }
        buffer_.create();
        dirty_ = true;
    }
    void resizeGL(int width, int height) override {
        glViewport(0, 0, width, height);
    }
    void paintGL() override {
        glClearColor(.067f, .11f, .17f, 1);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
        if (!program_ || !mesh_ || mesh_->vertices.empty())
            return;
        if (dirty_)
            upload();
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
        buffer_.bind();
        const int pos = program_->attributeLocation("position"),
                  col = program_->attributeLocation("color");
        program_->enableAttributeArray(pos);
        program_->enableAttributeArray(col);
        program_->setAttributeBuffer(pos, GL_FLOAT, 0, 3, 6 * sizeof(float));
        program_->setAttributeBuffer(col, GL_FLOAT, 3 * sizeof(float), 3,
                                     6 * sizeof(float));
        glDrawArrays(mode_ == 0   ? GL_POINTS
                     : mode_ == 2 ? GL_LINES
                                  : GL_TRIANGLES,
                     0, count_);
        if (camera_visible_ && camera_count_ > 0)
            glDrawArrays(GL_LINES, count_, camera_count_);
        program_->disableAttributeArray(pos);
        program_->disableAttributeArray(col);
        buffer_.release();
        program_->release();
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
    void cleanup() {
        if (context()) {
            makeCurrent();
            buffer_.destroy();
            program_.reset();
            doneCurrent();
        }
        if (context())
            disconnect(context(), nullptr, this, nullptr);
    }
    void upload() {
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
        camera_count_ = 0;
        camera_radius_ = .87f; // Radius of the mesh's normalized bounding box.
        if (mesh_->camera) {
            const auto &camera = *mesh_->camera;
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
                camera_count_ = int(data.size() / 6) - count_;
            }
        }
        buffer_.bind();
        buffer_.allocate(data.data(), int(data.size() * sizeof(float)));
        buffer_.release();
        dirty_ = false;
    }
    QOpenGLBuffer buffer_;
    std::unique_ptr<QOpenGLShaderProgram> program_;
    std::shared_ptr<const fs::MeshResult> mesh_;
    bool dirty_{true}, camera_visible_{true};
    int mode_{1}, count_{}, camera_count_{};
    float camera_radius_{};
    float yaw_{}, pitch_{}, zoom_{1};
    QPointF last_, pan_;
};

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
