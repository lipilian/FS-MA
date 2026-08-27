from pathlib import Path
import sys
REPO_ROOT = Path(__file__).resolve().parents[1]
FOUNDATION_STEREO_ROOT = REPO_ROOT / "FoundationStereo"
WEIGHT_DIR = REPO_ROOT / "weight" / "23-51-11"
OUTPUT_PATH = REPO_ROOT / "onnx" / "coarse_engine_512x512_initial_disp_liu.onnx"
sys.path.insert(0, str(FOUNDATION_STEREO_ROOT))

from omegaconf import OmegaConf
from core.foundation_stereo import FoundationStereo
import torch
class FoundationStereoOnnx(FoundationStereo):
    def __init__(self, args):
        super().__init__(args)

    @torch.no_grad()
    def forward(self, left, right):
        """ Removes extra outputs and hyper-parameters """
        with torch.amp.autocast('cuda', enabled=True):
            disp = FoundationStereo.forward(self, left, right, iters=self.args.valid_iters, test_mode=True)
        return disp
cfg = OmegaConf.load(WEIGHT_DIR / "cfg.yaml")
cfg['vit_size'] = 'vitl'
cfg['height'] = 512
cfg['width'] = 512
model = FoundationStereoOnnx(cfg)
ckpt = torch.load(WEIGHT_DIR / "model_best_bp2.pth", weights_only=False, map_location = 'cpu') # except model tensor, it also contains trainning metadata.
model.load_state_dict(ckpt['model'])
model.cuda()
model.eval()
left_img = torch.randn(1, 3, cfg.height, cfg.width).cuda().float()
right_img = torch.randn(1, 3, cfg.height, cfg.width).cuda().float()

torch.onnx.export(
        model,
        (left_img, right_img),
        OUTPUT_PATH,
        input_names = ['left', 'right'],
        output_names = ['disp'],
        opset_version = 16,
        dynamo = False,
    )
