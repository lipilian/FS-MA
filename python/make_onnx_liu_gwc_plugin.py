"""Export the fixed 800x960 FoundationStereo model with a GWC TensorRT node.

The exported node is:
    foundation_stereo::GWCVolume(left_feature, right_feature)

It must be parsed with libGWCVolumePlugin.so loaded and registered.
"""

from pathlib import Path
import sys

import torch
from omegaconf import OmegaConf


REPO_ROOT = Path(__file__).resolve().parents[1]
FOUNDATION_STEREO_ROOT = REPO_ROOT / "FoundationStereo"
WEIGHT_DIR = REPO_ROOT / "weight" / "23-51-11"
OUTPUT_PATH = REPO_ROOT / "onnx" / "foundationstereo_800x960_gwc_plugin.onnx"

sys.path.insert(0, str(FOUNDATION_STEREO_ROOT))

import core.foundation_stereo as foundation_stereo_module
from core.foundation_stereo import FoundationStereo
from core.submodule import build_gwc_volume as eager_build_gwc_volume


FEATURE_SHAPE = (1, 224, 200, 240)
VOLUME_SHAPE = (1, 8, 48, 200, 240)
MAX_DISPARITY = 48
NUM_GROUPS = 8


class GWCVolumeExportOp(torch.autograd.Function):
    """Use eager GWC for tracing, but emit one TensorRT custom ONNX node."""

    @staticmethod
    def forward(ctx, left_feature, right_feature):
        return eager_build_gwc_volume(
            left_feature,
            right_feature,
            maxdisp=MAX_DISPARITY,
            num_groups=NUM_GROUPS,
        )

    @staticmethod
    def symbolic(g, left_feature, right_feature):
        volume = g.op("foundation_stereo::GWCVolume", left_feature, right_feature)
        # The custom operation changes rank 4 features into a rank 5 volume.
        return volume.setType(left_feature.type().with_sizes(VOLUME_SHAPE))


def export_build_gwc_volume(refimg_fea, targetimg_fea, maxdisp, num_groups, stride=1):
    """Fixed-shape export replacement for FoundationStereo's GWC Python loop."""
    assert (maxdisp, num_groups, stride) == (MAX_DISPARITY, NUM_GROUPS, 1)
    assert tuple(refimg_fea.shape) == FEATURE_SHAPE
    assert tuple(targetimg_fea.shape) == FEATURE_SHAPE

    # GWCVolumePlugin currently accepts FP32 LINEAR tensors only.  The casts
    # become explicit ONNX Cast nodes if autocast made the features FP16.
    return GWCVolumeExportOp.apply(refimg_fea.float(), targetimg_fea.float())


class FoundationStereoOnnx(FoundationStereo):
    @torch.no_grad()
    def forward(self, left, right):
        with torch.amp.autocast("cuda", enabled=True):
            return FoundationStereo.forward(
                self,
                left,
                right,
                iters=self.args.valid_iters,
                test_mode=True,
            )


def main():
    cfg = OmegaConf.load(WEIGHT_DIR / "cfg.yaml")
    cfg["vit_size"] = "vitl"
    cfg["height"] = 800
    cfg["width"] = 960
    cfg["valid_iters"] = 32

    model = FoundationStereoOnnx(cfg)
    checkpoint = torch.load(
        WEIGHT_DIR / "model_best_bp2.pth",
        weights_only=False,
        map_location="cpu",
    )
    model.load_state_dict(checkpoint["model"])
    model.cuda().eval()

    # FoundationStereo.forward imported build_gwc_volume into its module scope.
    # Patch that export-only alias, not core/submodule.py, so normal PyTorch
    # inference remains unchanged.
    original_build_gwc_volume = foundation_stereo_module.build_gwc_volume
    foundation_stereo_module.build_gwc_volume = export_build_gwc_volume

    left_img = torch.randn(1, 3, cfg.height, cfg.width, device="cuda", dtype=torch.float32)
    right_img = torch.randn(1, 3, cfg.height, cfg.width, device="cuda", dtype=torch.float32)
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    try:
        torch.onnx.export(
            model,
            (left_img, right_img),
            OUTPUT_PATH,
            input_names=["left", "right"],
            output_names=["disp"],
            custom_opsets={"foundation_stereo": 1},
            dynamo=False,
        )
    finally:
        foundation_stereo_module.build_gwc_volume = original_build_gwc_volume

    print(f"Exported plugin-ready ONNX: {OUTPUT_PATH}")
    print("Custom node: foundation_stereo::GWCVolume")
    print(f"Plugin interface: {FEATURE_SHAPE} x 2 -> {VOLUME_SHAPE}")


if __name__ == "__main__":
    main()
