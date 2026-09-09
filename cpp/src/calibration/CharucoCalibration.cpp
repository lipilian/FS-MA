#include "fs/calibration/CharucoCalibration.hpp"
#include "fs/stereo/StereoFrame.hpp"
#include <opencv2/calib3d.hpp>
#include <opencv2/imgproc.hpp>
#include <algorithm>
#include <cmath>
#include <map>
#include <stdexcept>

namespace fs::calibration {
namespace {
const std::vector<std::string> names{
    "DICT_4X4_50", "DICT_4X4_100", "DICT_4X4_250", "DICT_4X4_1000",
    "DICT_5X5_50", "DICT_5X5_100", "DICT_5X5_250", "DICT_5X5_1000",
    "DICT_6X6_50", "DICT_6X6_100", "DICT_6X6_250", "DICT_6X6_1000",
    "DICT_7X7_50", "DICT_7X7_100", "DICT_7X7_250", "DICT_7X7_1000"};
struct Matched {
    std::vector<cv::Point3f> object;
    std::vector<cv::Point2f> left, right;
};
Matched match(const Sample& sample, const cv::Ptr<cv::aruco::CharucoBoard>& board) {
    if (sample.left.ids.size() != sample.left.corners.size() ||
        sample.right.ids.size() != sample.right.corners.size())
        throw std::invalid_argument("Invalid corner/ID arrays");
    Matched m;
    std::map<int, cv::Point2f> right;
    for (std::size_t i = 0; i < sample.right.ids.size(); ++i)
        right.emplace(sample.right.ids[i], sample.right.corners[i]);
    for (std::size_t i = 0; i < sample.left.ids.size(); ++i) {
        const int id = sample.left.ids[i];
        const auto it = right.find(id);
        if (it == right.end() || id < 0 || id >= int(board->chessboardCorners.size())) continue;
        m.object.push_back(board->chessboardCorners[id]);
        m.left.push_back(sample.left.corners[i]);
        m.right.push_back(it->second);
    }
    return m;
}
void validate_images(const Sample& sample, cv::Size size) {
    for (const auto* image : {&sample.images.left.rgb, &sample.images.right.rgb})
        if (image->empty() || image->type() != CV_8UC3 || image->size() != size)
            throw std::invalid_argument("Samples must contain equal-size RGB8 images");
}
double rms(const std::vector<cv::Point2f>& a, const std::vector<cv::Point2f>& b) {
    double sum = 0;
    for (std::size_t i = 0; i < a.size(); ++i) { const auto d = a[i] - b[i]; sum += d.dot(d); }
    return std::sqrt(sum / a.size());
}
void valid_quality(const Quality& q) {
    for (double x : {q.left, q.right, q.stereo})
        if (!std::isfinite(x) || x < 0) throw std::invalid_argument("Invalid RMS value");
}
}
std::vector<std::string> dictionary_names() { return names; }
cv::Ptr<cv::aruco::CharucoBoard> make_board(const BoardConfig& c) {
    const auto it = std::find(names.begin(), names.end(), c.dictionary_name);
    if (it == names.end()) throw std::invalid_argument("Unsupported ArUco dictionary");
    if (c.squares_x < 3 || c.squares_y < 3 || c.squares_x > 30 || c.squares_y > 30 ||
        !std::isfinite(c.square_length_m) || !std::isfinite(c.marker_length_m) ||
        c.square_length_m <= 0 || c.marker_length_m <= 0 || c.marker_length_m >= c.square_length_m)
        throw std::invalid_argument("Board needs 3–30 squares per axis and 0 < marker < square");
    const auto dictionary = cv::aruco::getPredefinedDictionary(int(it - names.begin()));
    if (c.squares_x * c.squares_y / 2 > dictionary->bytesList.rows)
        throw std::invalid_argument("Dictionary has too few markers for this board");
    return cv::aruco::CharucoBoard::create(c.squares_x, c.squares_y,
        float(c.square_length_m), float(c.marker_length_m), dictionary);
}
bool same_board(const BoardConfig& a, const BoardConfig& b) {
    return a.squares_x == b.squares_x && a.squares_y == b.squares_y &&
        std::abs(a.square_length_m - b.square_length_m) < 1e-8 &&
        std::abs(a.marker_length_m - b.marker_length_m) < 1e-8 && a.dictionary_name == b.dictionary_name;
}
Detection detect(const cv::Mat& rgb, const cv::Ptr<cv::aruco::CharucoBoard>& board) {
    cv::Mat gray, enhanced;
    cv::cvtColor(rgb, gray, cv::COLOR_RGB2GRAY);
    cv::createCLAHE(2.0, {8, 8})->apply(gray, enhanced);
    Detection result;
    cv::aruco::detectMarkers(enhanced, board->dictionary, result.marker_corners, result.marker_ids);
    result.markers = int(result.marker_ids.size());
    if (!result.marker_ids.empty()) cv::aruco::interpolateCornersCharuco(
        result.marker_corners, result.marker_ids, enhanced, board, result.corners, result.ids);
    return result;
}
cv::Mat annotated(const cv::Mat& rgb, const Detection& d) {
    cv::Mat output = rgb.clone();
    if (!d.marker_ids.empty())
        cv::aruco::drawDetectedMarkers(output, d.marker_corners, d.marker_ids, {0, 255, 0});
    if (!d.ids.empty()) cv::aruco::drawDetectedCornersCharuco(output, d.corners, d.ids, {60, 235, 170});
    return output;
}
std::size_t common_corners(const Detection& a, const Detection& b) {
    return std::count_if(a.ids.begin(), a.ids.end(), [&](int id) {
        return std::find(b.ids.begin(), b.ids.end(), id) != b.ids.end();
    });
}
std::uint64_t arrival_skew(const StereoCameraPair& p) {
    const auto a = p.left.host_arrival_ns, b = p.right.host_arrival_ns;
    return a > b ? a - b : b - a;
}
SessionResult solve(const BoardConfig& config, const std::vector<Sample>& samples) {
    if (samples.size() < 3) throw std::runtime_error("Need at least 3 valid pairs; use varied board poses");
    const auto board = make_board(config);
    SessionResult result;
    result.board = config;
    result.image_size = samples.front().images.left.rgb.size();
    std::vector<std::vector<cv::Point2f>> lc, rc, sl, sr;
    std::vector<std::vector<int>> li, ri;
    std::vector<std::vector<cv::Point3f>> object;
    for (const auto& sample : samples) {
        validate_images(sample, result.image_size);
        const auto m = match(sample, board);
        if (sample.left.ids.size() >= 4) { lc.push_back(sample.left.corners); li.push_back(sample.left.ids); }
        if (sample.right.ids.size() >= 4) { rc.push_back(sample.right.corners); ri.push_back(sample.right.ids); }
        if (m.object.size() >= 4) { object.push_back(m.object); sl.push_back(m.left); sr.push_back(m.right); }
    }
    if (lc.size() < 3 || rc.size() < 3 || object.size() < 3)
        throw std::runtime_error("Need 3 pairs with at least 4 common ChArUco corners");
    auto& c = result.calibration;
    std::vector<cv::Mat> rvecs, tvecs;
    const cv::TermCriteria criteria(cv::TermCriteria::COUNT + cv::TermCriteria::EPS, 100, 1e-6);
    result.solve.left = cv::aruco::calibrateCameraCharuco(lc, li, board, result.image_size,
        c.left_camera_matrix, c.left_distortion, rvecs, tvecs, 0, criteria);
    result.solve.right = cv::aruco::calibrateCameraCharuco(rc, ri, board, result.image_size,
        c.right_camera_matrix, c.right_distortion, rvecs, tvecs, 0, criteria);
    cv::Mat e, f;
    // Legacy JSON field names: values are OpenCV's LEFT-to-RIGHT R,T, unchanged.
    result.solve.stereo = cv::stereoCalibrate(object, sl, sr, c.left_camera_matrix, c.left_distortion,
        c.right_camera_matrix, c.right_distortion, result.image_size,
        c.right_to_left_rotation, c.right_to_left_translation, e, f, cv::CALIB_FIX_INTRINSIC, criteria);
    result.solve.pairs = int(object.size());
    validate(result);
    return result;
}
Quality check(const SessionResult& result, const Sample& sample) {
    validate_images(sample, result.image_size);
    const auto m = match(sample, make_board(result.board));
    if (m.object.size() < 4) throw std::runtime_error("Check needs at least 4 common corners");
    const auto& c = result.calibration;
    cv::Mat rv, tv, rotation, right_rv;
    if (!cv::solvePnP(m.object, m.left, c.left_camera_matrix, c.left_distortion, rv, tv))
        throw std::runtime_error("Cannot estimate board pose");
    std::vector<cv::Point2f> pl, pr;
    cv::projectPoints(m.object, rv, tv, c.left_camera_matrix, c.left_distortion, pl);
    cv::Rodrigues(rv, rotation);
    cv::Rodrigues(c.right_to_left_rotation * rotation, right_rv);
    cv::projectPoints(m.object, right_rv, c.right_to_left_rotation * tv + c.right_to_left_translation.reshape(1, 3),
        c.right_camera_matrix, c.right_distortion, pr);
    Quality q;
    q.left = rms(m.left, pl); q.right = rms(m.right, pr);
    q.stereo = std::sqrt((q.left * q.left + q.right * q.right) / 2);
    q.pairs = 1;
    valid_quality(q);
    return q;
}
void validate(const SessionResult& r) {
    make_board(r.board);
    if (r.image_size.width <= 0 || r.image_size.height <= 0 ||
        r.image_size.width > 16384 || r.image_size.height > 16384 ||
        r.left_serial.empty() || r.right_serial.empty() || r.left_serial == r.right_serial)
        throw std::invalid_argument("Invalid image grid or camera identities");
    // Matrix validation does not require allocating camera-size dummy images.
    const cv::Mat dummy(1, 1, CV_8UC3, cv::Scalar::all(0));
    StereoFrame validated(dummy, dummy, r.calibration);
    valid_quality(r.solve);
    if (r.checked) valid_quality(r.check);
    if (!std::isfinite(r.threshold_px) || r.threshold_px <= 0)
        throw std::invalid_argument("RMS threshold must be positive");
}
std::string serialize(const SessionResult& r) {
    validate(r);
    cv::FileStorage f(".json", cv::FileStorage::WRITE | cv::FileStorage::MEMORY | cv::FileStorage::FORMAT_JSON);
    f << "schema_version" << 1 << "squares_x" << r.board.squares_x << "squares_y" << r.board.squares_y
      << "square_length_m" << r.board.square_length_m << "marker_length_m" << r.board.marker_length_m
      << "dictionary_name" << r.board.dictionary_name << "image_width" << r.image_size.width
      << "image_height" << r.image_size.height << "left_serial" << r.left_serial << "right_serial" << r.right_serial
      << "left_rms" << r.solve.left << "right_rms" << r.solve.right << "stereo_rms" << r.solve.stereo
      << "stereo_pair_count" << r.solve.pairs << "threshold_px" << r.threshold_px
      << "checked" << int(r.checked) << "check_left_rms" << r.check.left
      << "check_right_rms" << r.check.right << "check_stereo_rms" << r.check.stereo;
    const auto& c = r.calibration;
    f << "left_camera_matrix" << c.left_camera_matrix << "right_camera_matrix" << c.right_camera_matrix
      << "left_distortion" << c.left_distortion << "right_distortion" << c.right_distortion
      << "right_to_left_rotation" << c.right_to_left_rotation
      << "right_to_left_translation" << c.right_to_left_translation;
    return f.releaseAndGetString();
}
SessionResult load(const std::string& path) {
    cv::FileStorage f(path, cv::FileStorage::READ);
    if (!f.isOpened()) throw std::runtime_error("Cannot open calibration JSON");
    for (const char* key : {"squares_x", "squares_y", "square_length_m", "marker_length_m", "dictionary_name",
        "image_width", "image_height", "left_serial", "right_serial", "left_rms", "right_rms", "stereo_rms", "stereo_pair_count"})
        if (f[key].empty()) throw std::runtime_error(std::string("Missing metadata: ") + key +
            ". Legacy matrix-only files remain usable in the CLI; recalibrate here.");
    SessionResult r;
    f["squares_x"] >> r.board.squares_x; f["squares_y"] >> r.board.squares_y;
    f["square_length_m"] >> r.board.square_length_m; f["marker_length_m"] >> r.board.marker_length_m;
    f["dictionary_name"] >> r.board.dictionary_name;
    f["image_width"] >> r.image_size.width; f["image_height"] >> r.image_size.height;
    f["left_serial"] >> r.left_serial; f["right_serial"] >> r.right_serial;
    f["left_rms"] >> r.solve.left; f["right_rms"] >> r.solve.right; f["stereo_rms"] >> r.solve.stereo;
    f["stereo_pair_count"] >> r.solve.pairs;
    r.calibration = StereoCalibration::from_file(path);
    for (auto* mat : {&r.calibration.left_camera_matrix, &r.calibration.right_camera_matrix,
        &r.calibration.left_distortion, &r.calibration.right_distortion,
        &r.calibration.right_to_left_rotation, &r.calibration.right_to_left_translation}) {
        if (mat->depth() != CV_32F && mat->depth() != CV_64F) throw std::runtime_error("Matrices must be floating point");
        mat->convertTo(*mat, CV_64F);
    }
    validate(r);
    return r;
}
SessionResult clone(const SessionResult& source) {
    SessionResult r = source;
    auto& c = r.calibration;
    c.left_camera_matrix = c.left_camera_matrix.clone(); c.right_camera_matrix = c.right_camera_matrix.clone();
    c.left_distortion = c.left_distortion.clone(); c.right_distortion = c.right_distortion.clone();
    c.right_to_left_rotation = c.right_to_left_rotation.clone();
    c.right_to_left_translation = c.right_to_left_translation.clone();
    return r;
}
} // namespace fs::calibration
