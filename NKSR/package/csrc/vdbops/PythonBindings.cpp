#include <torch/extension.h>
#include <torch/csrc/autograd/python_variable.h>
#include <pybind11/stl.h>
#include <array>

#include "SparseFeatureIndexGrid.h"

using IndexGridHost = SparseFeatureIndexGrid<nanovdb::HostBuffer>;

using IndexGridDevice = SparseFeatureIndexGrid<fvdb::PytorchDeviceBuffer>;


namespace pybind11 {namespace detail {
// Convert through std::array: newer pybind11 array_caster constructs its
// target with nested braces, which NanoVDB's vector types do not support.
template <typename VectorT, typename ScalarT>
struct nanovdb_vec3_caster {
    using ArrayT = std::array<ScalarT, 3>;
    using ArrayCaster = make_caster<ArrayT>;

    PYBIND11_TYPE_CASTER(VectorT, ArrayCaster::name);

    bool load(handle src, bool convert) {
        // Tensor implements iteration but is not a Python sequence, so the
        // std::array caster needs an explicit conversion for tensor inputs.
        object tensor_values;
        if (THPVariable_Check(src.ptr())) {
            const auto& tensor = THPVariable_Unpack(src.ptr());
            if (!convert || tensor.dim() != 1 || tensor.numel() != 3) {
                return false;
            }
            tensor_values = src.attr("tolist")();
            src = tensor_values;
        }
        ArrayCaster caster;
        if (!caster.load(src, convert)) {
            return false;
        }
        auto values = cast_op<ArrayT>(caster);
        value = VectorT(values[0], values[1], values[2]);
        return true;
    }

    static handle cast(const VectorT& src, return_value_policy policy, handle parent) {
        return ArrayCaster::cast(ArrayT{src[0], src[1], src[2]}, policy, parent);
    }
};

template <>
struct type_caster<nanovdb::Coord>
 : nanovdb_vec3_caster<nanovdb::Coord, int32_t> { };

template <typename ScalarT>
struct type_caster<nanovdb::Vec3<ScalarT>>
 : nanovdb_vec3_caster<nanovdb::Vec3<ScalarT>, ScalarT> { };
}}


template <typename GridT>
py::class_<GridT, std::shared_ptr<GridT>> makeBinding(pybind11::handle& m, const char* name) {
    return py::class_<GridT, std::shared_ptr<GridT>>(m, name, py::module_local())
        .def(py::init<double, double, c10::DeviceIndex>(), py::arg("vox_size"), py::arg("vox_origin"), py::arg("device_index") = -1)
        .def("build_from_pointcloud", &GridT::buildFromPointsWithPadding)
        .def("build_from_pointcloud_nearest_voxels", &GridT::buildFromPointsNearestNeighbors)
        .def("build_from_ijk_coords", &GridT::buildFromVoxelCoordinates)
        .def("num_voxels", &GridT::numVoxels)
        .def("origin", &GridT::voxelOrigin)
        .def("device", &GridT::device)
        .def("set_origin", &GridT::setVoxelOrigin)
        .def("voxel_size", &GridT::voxelSize)
        .def("set_voxel_size", &GridT::setVoxelSize)
        .def("active_grid_coords", &GridT::activeGridCoords)
        .def("points_in_active_voxel", &GridT::pointsInGrid)
        .def("ijk_to_index", &GridT::ijkToIndex)
        .def("splat_trilinear",
            [](const std::shared_ptr<GridT>& grid, const torch::Tensor& points,
               const torch::Tensor& pointsData, bool returnCounts) {
                    std::vector<torch::Tensor> ret = grid->splatTrilinear(points, pointsData, returnCounts);
                    if (ret.size() == 1) {
                        return py::cast(ret[0]);
                    } else {
                        return py::cast(ret);
                    }
            }, py::arg("points"), py::arg("points_data"), py::arg("return_counts") = false)
        .def("sample_trilinear",
             [](const std::shared_ptr<GridT>& grid, const torch::Tensor& points,
                const torch::Tensor& gridData, bool returnGrad) {
                 std::vector<torch::Tensor> ret = grid->sampleTrilinear(points, gridData, returnGrad);
                 if (ret.size() == 1) {
                     return py::cast(ret[0]);
                 } else {
                     return py::cast(ret);
                 }
             }, py::arg("points"), py::arg("grid_data"), py::arg("return_grad") = false)
        .def("sample_bezier",
             [](const std::shared_ptr<GridT>& grid, const torch::Tensor& points,
                const torch::Tensor& gridData, bool returnGrad) {
                 std::vector<torch::Tensor> ret = grid->sampleBezier(points, gridData, returnGrad);
                 if (ret.size() == 1) {
                     return py::cast(ret[0]);
                 } else {
                     return py::cast(ret);
                 }
             }, py::arg("points"), py::arg("grid_data"), py::arg("return_grad") = false)
        .def("grid_to_world", &GridT::gridToWorld)
        .def("world_to_grid", &GridT::worldToGrid)
        .def("coarsened_grid", &GridT::coarsenedGrid, py::arg("branching_factor") = 2)
        .def("subdivided_grid", [](std::shared_ptr<GridT>& coarseGrid, unsigned subdivFactor, const torch::optional<torch::Tensor>& mask) {
            torch::Tensor inputMask = torch::zeros(0, torch::TensorOptions().dtype(torch::kBool));
            if (mask.has_value()) {
                inputMask = mask.value();
            }
            return coarseGrid->subdividedGrid(subdivFactor, inputMask);
        }, py::arg("branching_factor") = 2, py::arg("mask") = nullptr)
        .def("dual_grid", &GridT::dualGrid)
        .def("subdivide", [](std::shared_ptr<GridT>& coarseGrid, std::shared_ptr<GridT>& fineGrid, unsigned upsampleFactor, torch::Tensor& coarseData) {
                fvdb::checkDevice<typename GridT::BufferT>(coarseData, coarseGrid->device().index());
                TORCH_CHECK(coarseGrid->device() == fineGrid->device(), "coarse_grid and fine_grid must be on the same device");
                return fvdb::UpsampleNanovdbGrid<typename GridT::BufferT>::apply(coarseGrid->nanovdbGrid(),
                                                                                 fineGrid->nanovdbGrid(),
                                                                                 upsampleFactor,
                                                                                 coarseData)[0];
        })
        .def("max_pool", [](std::shared_ptr<GridT>& fineGrid,
                            std::shared_ptr<GridT>& coarseGrid,
                            unsigned poolFactor,
                            torch::Tensor& fineData) {
                TORCH_CHECK(coarseGrid->device() == fineGrid->device(), "coarse_grid and fine_grid must be on the same device");
                fvdb::checkDevice<typename GridT::BufferT>(fineData, fineGrid->device().index());
                return fvdb::MaxPoolNanovdbGrid<typename GridT::BufferT>::apply(fineGrid->nanovdbGrid(),
                                                                                coarseGrid->nanovdbGrid(),
                                                                                poolFactor,
                                                                                fineData)[0];
        });
}


void pybind_vdbops(py::module& m) {
    makeBinding<IndexGridHost>(m, "_CpuIndexGrid")
    // TODO: Francis: This API is hacky and I will fix it when I refactor all the device stuff
    .def("to_cuda", [](std::shared_ptr<IndexGridHost>& grid, torch::optional<c10::DeviceIndex> deviceIndex) {

        c10::DeviceIndex actualDeviceIndex = fvdb::defaultDeviceIndex<fvdb::PytorchDeviceBuffer>();
        if (deviceIndex.has_value()) {
            actualDeviceIndex = deviceIndex.value();
        }
        std::shared_ptr<IndexGridDevice> ret = std::make_shared<IndexGridDevice>(grid->voxelSize(), grid->voxelOrigin(), actualDeviceIndex);
        ret->moveFrom(*grid);
        return ret;
    });
    makeBinding<IndexGridDevice>(m, "_CudaIndexGrid")
    // TODO: Francis: This API is hacky and I will fix it when I refactor all the device stuff
    .def("to_cpu", [](std::shared_ptr<IndexGridDevice>& grid) {
        c10::DeviceIndex actualDeviceIndex = fvdb::defaultDeviceIndex<nanovdb::HostBuffer>();
        std::shared_ptr<IndexGridHost> ret = std::make_shared<IndexGridHost>(grid->voxelSize(), grid->voxelOrigin(), actualDeviceIndex);
        ret->moveFrom(*grid);
        return ret;
    });

}
