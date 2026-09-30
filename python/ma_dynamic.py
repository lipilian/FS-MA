"""Tensor-only, single-scene MapAnything forward with dynamic view count.

Input: float32 [V, 7, H, W], 2 <= V <= 5, normalized RGB / unit camera
rays / metric distance along rays. Output heads are RAW, before adaptors.
The wrapped checkpoint remains compatible with its original forward/infer.
"""
from __future__ import annotations

import torch
from torch import nn
from mapanything.utils.geometry import apply_log_to_norm
from uniception.models.encoders import (
    EncoderGlobalRepInput, ViTEncoderInput, ViTEncoderNonImageInput,
)
from uniception.models.prediction_heads.base import (
    PredictionHeadInput, PredictionHeadLayeredInput, PredictionHeadTokenInput,
)

OUTPUT_NAMES = ("depths", "poses", "scale")


class MapAnythingDynamicRaw(nn.Module):
    """Reuse the current DINOv2 / alternating-attention / DPT+pose weights.

    B is fixed at one. The first view is the reference view. V stays a symbolic
    tensor dimension throughout; there is no Python loop over input views.
    Precision is a construction-time setting, not a runtime ONNX input.
    """

    def __init__(self, model, precision="fp32", height=434, width=518):
        super().__init__()
        if precision not in ("fp32", "bf16"):
            raise ValueError("precision must be fp32 or bf16")
        if height <= 0 or width <= 0 or height % 14 or width % 14:
            raise ValueError("H/W must be positive multiples of 14")
        tr = model.info_sharing
        if (model.pred_head_type != "dpt+pose"
                or model.scene_rep_type != "raydirs+depth+pose+confidence+mask"
                or model.info_sharing_type != "alternating_attention"
                or model.info_sharing_return_type != "intermediate_features"
                or not model.use_encoder_features_for_dpt
                or tuple(tr.indices) != (7, 11)
                or tr.intermediates_only
                or tr.custom_positional_encoding is not None
                or tr.use_pe_for_non_reference_views
                or model.encoder.data_norm_type != "dinov2"):
            raise ValueError("Unsupported checkpoint: this adapter targets the local MapAnything configuration")
        if model.dtype != torch.float32:
            raise ValueError("Keep checkpoint weights in FP32; use precision='bf16' for autocast")
        self.model = model
        self.precision = precision
        self.height, self.width = height, width
        self.eval()

    def _fuse_geometry(self, features, rays, depth):
        # Match the original temporary buffers' dtype, including any rounding
        # before encoding. Disabling autocast alone does not cast tensors.
        feature_dtype = features.dtype
        rays = rays.to(feature_dtype).contiguous()
        ray_features = self.model.ray_dirs_encoder(
            ViTEncoderNonImageInput(data=rays)
        ).features
        features = features + ray_features

        # Original reductions run over NHWC depth (with a singleton channel).
        depth = depth.permute(0, 2, 3, 1).contiguous()
        valid = depth > 0
        weighted = depth * valid
        if self.precision == "bf16":
            # CUDA selects a different floating-point reduction tree when
            # reducing V rows together. A one-ULP change can cross a BF16
            # rounding boundary in attention. Keep the original single-view
            # sum shape using five fixed scalar reductions (our declared upper
            # bound), not a Python loop or a variable-length list of views.
            # Repeated trailing rows are discarded; all feature computation
            # and attention still use the true symbolic V.
            indices = torch.arange(5, device=depth.device).clamp(max=depth.shape[0] - 1)
            rows = weighted.index_select(0, indices)
            sums = torch.cat((rows[0:1].sum((1, 2, 3)),
                              rows[1:2].sum((1, 2, 3)),
                              rows[2:3].sum((1, 2, 3)),
                              rows[3:4].sum((1, 2, 3)),
                              rows[4:5].sum((1, 2, 3))))[:depth.shape[0]]
        else:
            sums = weighted.sum((1, 2, 3))
        factor = (sums / (valid.sum((1, 2, 3)) + 1e-8)).clamp(min=1e-8)
        normalized = (depth / factor[:, None, None, None]).to(feature_dtype)
        factor = factor.to(feature_dtype)
        encoded = apply_log_to_norm(normalized).permute(0, 3, 1, 2).contiguous()
        depth_features = self.model.depth_encoder(
            ViTEncoderNonImageInput(data=encoded)
        ).features
        scale_features = self.model.depth_scale_encoder(
            EncoderGlobalRepInput(data=torch.log(factor + 1e-8).unsqueeze(-1))
        ).features
        features = features + depth_features + scale_features[:, :, None, None]
        # No pose priors; the original pose contributions are masked to zero.
        features = self.model.fusion_norm_layer(features.permute(0, 2, 3, 1).contiguous())
        return features.permute(0, 3, 1, 2).contiguous()

    def _share_views(self, features, registers):
        tr = self.model.info_sharing
        v, c, h, w = features.shape
        tokens = features.flatten(2)
        if self.model.use_register_tokens_from_encoder and registers is not None:
            tokens = torch.cat((tokens, registers), dim=2)
        tokens_per_view = tokens.shape[2]
        tokens = tokens.transpose(1, 2).reshape(1, v * tokens_per_view, c).contiguous()
        scale_token = self.model.scale_token.reshape(1, 1, c)
        tokens = tr.proj_embed(torch.cat((tokens, scale_token), dim=1))
        if tr.distinguish_ref_and_non_ref_views:
            reference = tokens[:, :tokens_per_view] + tr.view_pos_table[0].reshape(1, 1, tr.dim)
            tokens = torch.cat((reference, tokens[:, tokens_per_view:]), dim=1)

        intermediate = []
        for index, block in enumerate(tr.self_attention_blocks):
            if index % 2 == 0:
                tokens = block(tokens, None)
            else:
                global_tokens = tokens[:, v * tokens_per_view:]
                frames = tokens[:, :v * tokens_per_view].reshape(v, tokens_per_view, tr.dim).contiguous()
                frames = block(frames, None)
                tokens = torch.cat((frames.reshape(1, v * tokens_per_view, tr.dim), global_tokens), dim=1)
            if index in tr.indices:
                normalized = tr.norm(tokens) if tr.norm_intermediate else tokens
                intermediate.append(self._spatial_features(normalized, v, tokens_per_view, h, w))
        tokens = tr.norm(tokens)
        final = self._spatial_features(tokens, v, tokens_per_view, h, w)
        scale = tokens[:, v * tokens_per_view:].transpose(1, 2).contiguous()
        return final, intermediate, scale

    @staticmethod
    def _spatial_features(tokens, v, tokens_per_view, h, w):
        frames = tokens[:, :v * tokens_per_view].reshape(v, tokens_per_view, -1)
        return frames[:, :h * w].transpose(1, 2).reshape(v, -1, h, w).contiguous()

    def forward(self, inputs):
        torch._assert(inputs.ndim == 4, "Expected [V,7,H,W]")
        torch._assert(inputs.shape[0] >= 2, "At least two views required")
        torch._assert(inputs.shape[0] <= 5, "At most five views supported")
        torch._assert(inputs.shape[1] == 7, "Expected RGB/rays/depth: seven channels")
        torch._assert(inputs.shape[2] == self.height, "Unexpected height")
        torch._assert(inputs.shape[3] == self.width, "Unexpected width")
        torch._assert(inputs.dtype == torch.float32, "Inputs must be FP32")
        with torch.autocast(inputs.device.type, dtype=torch.bfloat16, enabled=self.precision == "bf16"):
            enc = self.model.encoder(ViTEncoderInput(
                image=inputs[:, :3].contiguous(), data_norm_type=self.model.encoder.data_norm_type,
            ))
            with torch.autocast(inputs.device.type, enabled=False):
                features = self._fuse_geometry(enc.features, inputs[:, 3:6], inputs[:, 6:7])
            final, intermediate, scale_features = self._share_views(features, enc.registers)
            with torch.autocast(inputs.device.type, enabled=False):
                # Explicit FP32 head interface; original FP32 residual/norm path
                # also supplies FP32 here. Validate against raw-head hooks.
                head_features = [features.float(), intermediate[0].float(),
                                 intermediate[1].float(), final.float()]
                dense = self.model.dense_head(PredictionHeadLayeredInput(
                    list_features=head_features, target_output_shape=(self.height, self.width),
                )).decoded_channels
                poses = self.model.pose_head(PredictionHeadInput(last_feature=final.float())).decoded_channels
                scale = self.model.scale_head(PredictionHeadTokenInput(last_feature=scale_features.float())).decoded_channels
        return {"depths": dense.float(), "poses": poses.float(), "scale": scale.float()}
