#include "fs/geometry/MeshBuilder.hpp"
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <mutex>
#include <numeric>
#include <opencv2/imgproc.hpp>
#include <stdexcept>
#include <unordered_set>

// Triangle's ABI uses double REAL and int* VOID. Keep definitions local.
extern "C" {
#define REAL double
#define VOID int
#define ANSI_DECLARATORS
#include "triangle.h"
#undef ANSI_DECLARATORS
#undef VOID
#undef REAL
}

namespace fs {
namespace {
// Exact nearest lookup on integer image coordinates. Equal-distance ties use
// the lowest input index; scipy.cKDTree may choose a different tied neighbour.
class PixelTree {
  public:
    explicit PixelTree(const std::vector<cv::Point> &points)
        : points_(points), order_(points.size()) {
        std::iota(order_.begin(), order_.end(), 0);
        build(0, order_.size(), 0);
    }
    int nearest(cv::Point p) const {
        int best = -1;
        long long distance = std::numeric_limits<long long>::max();
        query(0, order_.size(), 0, p, best, distance);
        return best;
    }

  private:
    static int coord(cv::Point p, int axis) { return axis ? p.y : p.x; }
    void build(int lo, int hi, int axis) {
        if (lo >= hi)
            return;
        int mid = lo + (hi - lo) / 2;
        std::nth_element(order_.begin() + lo, order_.begin() + mid,
                         order_.begin() + hi, [&](int a, int b) {
                             const int x = coord(points_[a], axis),
                                       y = coord(points_[b], axis);
                             return x != y ? x < y : a < b;
                         });
        build(lo, mid, 1 - axis);
        build(mid + 1, hi, 1 - axis);
    }
    void query(int lo, int hi, int axis, cv::Point p, int &best,
               long long &distance) const {
        if (lo >= hi)
            return;
        const int mid = lo + (hi - lo) / 2, id = order_[mid];
        const long long dx = p.x - points_[id].x, dy = p.y - points_[id].y,
                        d = dx * dx + dy * dy;
        if (d < distance || (d == distance && id < best)) {
            best = id;
            distance = d;
        }
        const long long delta = coord(p, axis) - coord(points_[id], axis);
        if (delta < 0) {
            query(lo, mid, 1 - axis, p, best, distance);
            if (delta * delta <= distance)
                query(mid + 1, hi, 1 - axis, p, best, distance);
        } else {
            query(mid + 1, hi, 1 - axis, p, best, distance);
            if (delta * delta <= distance)
                query(lo, mid, 1 - axis, p, best, distance);
        }
    }
    const std::vector<cv::Point> &points_;
    std::vector<int> order_;
};

struct TriangleOutput {
    triangulateio io{};
    ~TriangleOutput() {
        // holelist/regionlist alias inputs (both null here); never free
        // aliases.
        for (void *p : {static_cast<void *>(io.pointlist),
                        static_cast<void *>(io.pointattributelist),
                        static_cast<void *>(io.pointmarkerlist),
                        static_cast<void *>(io.trianglelist),
                        static_cast<void *>(io.triangleattributelist),
                        static_cast<void *>(io.neighborlist),
                        static_cast<void *>(io.segmentlist),
                        static_cast<void *>(io.segmentmarkerlist),
                        static_cast<void *>(io.edgelist),
                        static_cast<void *>(io.edgemarkerlist),
                        static_cast<void *>(io.normlist)})
            if (p)
                trifree(static_cast<int *>(p));
    }
};
std::mutex
    triangle_mutex; // Triangle has process-global predicate/random state.
std::uint64_t edgeKey(int a, int b) {
    if (a > b)
        std::swap(a, b);
    return (std::uint64_t(std::uint32_t(a)) << 32) | std::uint32_t(b);
}
// Nearest-sample snapping can turn an otherwise simple contour into a
// closed walk that revisits vertices. Split at those visits instead of
// discarding the whole component. Keep every nonzero-area cycle (including
// lobes joined at one vertex); discard only backtracking/collinear spurs.
// Each retained edge comes from the original walk: global deduplication would
// instead invent shortcuts between unrelated parts of the boundary.
std::vector<std::vector<int>> boundaryCycles(const std::vector<int> &walk,
                                            const std::vector<cv::Point> &uv) {
    std::vector<std::vector<int>> cycles;
    if (walk.empty())
        return cycles;
    std::vector<int> position(uv.size(), -1), path;
    const auto visit = [&](int id) {
        const int start = position[id];
        if (start < 0) {
            position[id] = int(path.size());
            path.push_back(id);
            return;
        }
        if (path.size() - start >= 3) {
            double twice_area = 0;
            for (size_t i = start; i < path.size(); ++i) {
                const auto a = uv[path[i]];
                const auto b = uv[i + 1 < path.size() ? path[i + 1] : id];
                twice_area += double(a.x) * b.y - double(a.y) * b.x;
            }
            if (twice_area != 0)
                cycles.emplace_back(path.begin() + start, path.end());
        }
        for (size_t i = start + 1; i < path.size(); ++i)
            position[path[i]] = -1;
        path.resize(start + 1);
    };
    for (int id : walk)
        visit(id);
    visit(walk.front());
    return cycles;
}
bool valid(cv::Vec3f p) {
    return std::isfinite(p[0]) && std::isfinite(p[1]) && std::isfinite(p[2]) &&
           p[2] > 0;
}
} // namespace

MeshResult build_constrained_mesh(const cv::Mat &xyz, const cv::Mat &selection,
                                  const cv::Mat &rgb, double max_edge_m,
                                  double max_depth_jump_m,
                                  const std::function<void()> &checkpoint) {
    if (xyz.empty() || xyz.type() != CV_32FC3 || selection.type() != CV_8UC1 ||
        rgb.type() != CV_8UC3 || selection.size() != xyz.size() ||
        rgb.size() != xyz.size() ||
        xyz.total() > std::numeric_limits<int>::max() ||
        !std::isfinite(max_edge_m) || max_edge_m <= 0 ||
        !std::isfinite(max_depth_jump_m) || max_depth_jump_m <= 0)
        throw std::invalid_argument(
            "Mesh needs aligned XYZ, binary mask and RGB "
            "with positive finite thresholds");
    const auto check = [&] {
        if (checkpoint)
            checkpoint();
    };
    check();
    cv::Mat labels;
    const int components =
        cv::connectedComponents(selection, labels, 8, CV_32S);
    std::vector<std::vector<cv::Point>> retained(components);
    for (int y = 0; y < xyz.rows; ++y)
        for (int x = 0; x < xyz.cols; ++x) {
            const int c = labels.at<int>(y, x);
            if (c && valid(xyz.at<cv::Vec3f>(y, x)))
                retained[c].emplace_back(x, y);
        }
    MeshResult result;
    cv::Mat input_at(xyz.size(), CV_32S, cv::Scalar(-1));
    for (int component = 1; component < components; ++component) {
        check();
        const auto &uv = retained[component];
        if (uv.size() < 3) {
            ++result.skipped_components;
            continue;
        }
        std::vector<std::vector<cv::Point>> contours;
        cv::findContours(labels == component, contours, cv::RETR_EXTERNAL,
                         cv::CHAIN_APPROX_NONE);
        if (contours.empty()) {
            ++result.skipped_components;
            continue;
        }
        const auto &contour = *std::max_element(
            contours.begin(), contours.end(), [](const auto &a, const auto &b) {
                return cv::contourArea(a) < cv::contourArea(b);
            });
        PixelTree tree(uv);
        std::vector<int> loop;
        for (auto p : contour) {
            int id = tree.nearest(p);
            if (loop.empty() || id != loop.back())
                loop.push_back(id);
        }
        if (loop.size() > 1 && loop.front() == loop.back())
            loop.pop_back();
        const auto cycles = boundaryCycles(loop, uv);
        if (cycles.empty()) {
            ++result.skipped_components;
            continue;
        }
        std::vector<double> points;
        points.reserve(uv.size() * 2);
        for (size_t i = 0; i < uv.size(); ++i) {
            points.push_back(uv[i].x);
            points.push_back(uv[i].y);
            input_at.at<int>(uv[i]) = int(i);
        }
        std::vector<int> segments;
        segments.reserve(loop.size() * 2);
        std::unordered_set<std::uint64_t> required;
        for (const auto &cycle : cycles)
            for (size_t i = 0; i < cycle.size(); ++i) {
                const int a = cycle[i], b = cycle[(i + 1) % cycle.size()];
                if (required.insert(edgeKey(a, b)).second) {
                    segments.push_back(a);
                    segments.push_back(b);
                }
            }
        triangulateio input{};
        input.pointlist = points.data();
        input.numberofpoints = int(uv.size());
        input.segmentlist = segments.data();
        input.numberofsegments = int(segments.size() / 2);
        TriangleOutput output;
        {
            std::lock_guard<std::mutex> lock(triangle_mutex);
            char options[] = "pzQ";
            triangulate(options, &input, &output.io, nullptr);
        }
        check();
        const auto &out = output.io;
        if (!out.trianglelist || !out.numberoftriangles) {
            ++result.skipped_components;
            continue;
        }
        std::vector<int> mapped(out.numberofpoints);
        for (int i = 0; i < out.numberofpoints; ++i) {
            const double x = out.pointlist[2 * i], y = out.pointlist[2 * i + 1];
            if (!std::isfinite(x) || !std::isfinite(y) || x < 0 || y < 0 ||
                x >= xyz.cols || y >= xyz.rows)
                throw std::runtime_error(
                    "CDT returned a vertex outside the image");
            const cv::Point p(cvRound(x), cvRound(y));
            if (p.x >= xyz.cols || p.y >= xyz.rows ||
                std::hypot(x - p.x, y - p.y) > 1e-6 ||
                labels.at<int>(p) != component || input_at.at<int>(p) < 0)
                throw std::runtime_error(
                    "CDT inserted a vertex without a retained XYZ sample");
            mapped[i] = input_at.at<int>(p);
        }
        std::vector<cv::Vec3i> faces;
        faces.reserve(out.numberoftriangles);
        std::vector<int> remap(uv.size(), -1);
        for (int i = 0; i < out.numberoftriangles; ++i) {
            if (i % 4096 == 0)
                check();
            cv::Vec3i f;
            for (int k = 0; k < 3; ++k) {
                int id = out.trianglelist[i * out.numberofcorners + k];
                if (id < 0 || id >= out.numberofpoints)
                    throw std::runtime_error(
                        "CDT returned an invalid triangle index");
                f[k] = mapped[id];
            }
            // Check the constraint contract BEFORE optional 3D filtering.
            for (int k = 0; k < 3; ++k)
                required.erase(edgeKey(f[k], f[(k + 1) % 3]));
            const cv::Point sum = uv[f[0]] + uv[f[1]] + uv[f[2]];
            if (labels.at<int>((sum.y + 1) / 3, (sum.x + 1) / 3) != component)
                continue;
            cv::Vec3d p[3] = {xyz.at<cv::Vec3f>(uv[f[0]]),
                              xyz.at<cv::Vec3f>(uv[f[1]]),
                              xyz.at<cv::Vec3f>(uv[f[2]])};
            if (std::max({p[0][2], p[1][2], p[2][2]}) -
                    std::min({p[0][2], p[1][2], p[2][2]}) >
                max_depth_jump_m)
                continue;
            bool keep = true;
            for (int k = 0; k < 3; ++k)
                if (cv::norm(p[k] - p[(k + 1) % 3]) > max_edge_m)
                    keep = false;
            if (!keep)
                continue;
            faces.push_back(f);
            for (int k = 0; k < 3; ++k)
                remap[f[k]] = 0;
        }
        if (!required.empty())
            throw std::runtime_error(
                "CDT did not preserve every exterior contour segment");
        for (size_t i = 0; i < uv.size(); ++i)
            if (remap[i] == 0) {
                remap[i] = int(result.vertices.size());
                result.vertices.push_back(xyz.at<cv::Vec3f>(uv[i]));
                result.colors.push_back(rgb.at<cv::Vec3b>(uv[i]));
            }
        for (auto f : faces) {
            for (int k = 0; k < 3; ++k)
                f[k] = remap[f[k]];
            result.triangles.push_back(f);
            const cv::Vec3d a = result.vertices[f[0]],
                            b = result.vertices[f[1]],
                            c = result.vertices[f[2]];
            result.area_m2 += .5 * cv::norm((b - a).cross(c - a));
        }
    }
    check();
    if (result.triangles.empty())
        throw std::runtime_error(
            "No mesh triangles survived; inspect the mask or "
            "adjust mesh thresholds");
    return result;
}
} // namespace fs
