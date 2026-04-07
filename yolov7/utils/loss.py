"""Loss functions module."""

import torch
from torch import nn


# pylint: disable=too-many-instance-attributes
class SigmoidBin(nn.Module):
    """Sigmoid binning module for regression with bin-based classification.
    
    This module combines bin classification with regression refinement for
    precise value prediction within defined ranges.
    
    Attributes:
        bin_count: Number of bins for classification.
        length: Total output length (bin_count + 1).
        min_val: Minimum value of the range.
        max_val: Maximum value of the range.
        scale: Range scale (max_val - min_val).
        shift: Half of the scale.
        use_loss_regression: Whether to use regression in loss calculation.
        use_fw_regression: Whether to use regression in forward pass.
        reg_scale: Scale factor for regression.
        BCE_weight: Weight for BCE loss.
        smooth_eps: Smoothing epsilon for targets.
    """
    stride = None  # strides computed during build
    export = False  # onnx export

    # pylint: disable=too-many-arguments, too-many-positional-arguments
    def __init__(self, bin_count=10, min_val=0.0, max_val=1.0, reg_scale=2.0,
                 use_loss_regression=True, use_fw_regression=True,
                 BCE_weight=1.0, smooth_eps=0.0):
        super().__init__()

        self.bin_count = bin_count
        self.length = bin_count + 1
        self.min_val = min_val
        self.max_val = max_val
        self.scale = float(max_val - min_val)
        self.shift = self.scale / 2.0

        self.use_loss_regression = use_loss_regression
        self.use_fw_regression = use_fw_regression
        self.reg_scale = reg_scale
        self.BCE_weight = BCE_weight

        start = min_val + (self.scale / 2.0) / self.bin_count
        end = max_val - (self.scale / 2.0) / self.bin_count
        step = self.scale / self.bin_count
        self.step = step
        #print(f" start = {start}, end = {end}, step = {step} ")

        bins = torch.range(start, end + 0.0001, step).float()
        self.register_buffer('bins', bins)

        self.cp = 1.0 - 0.5 * smooth_eps
        self.cn = 0.5 * smooth_eps

        self.BCEbins = nn.BCEWithLogitsLoss(pos_weight=torch.Tensor([BCE_weight]))
        self.MSELoss = nn.MSELoss()

    def get_length(self):
        """Return the length of the output."""
        return self.length

    def forward(self, pred):
        """Forward pass to compute regression result from predictions.
        
        Args:
            pred: Prediction tensor with shape [..., length].
            
        Returns:
            Clamped regression result within [min_val, max_val] range.
        """
        assert pred.shape[-1] == self.length, \
            f"pred.shape[-1]={pred.shape[-1]} is not equal to self.length={self.length}"

        pred_reg = (pred[..., 0] * self.reg_scale - self.reg_scale/2.0) * self.step
        pred_bin = pred[..., 1:(1+self.bin_count)]

        _, bin_idx = torch.max(pred_bin, dim=-1)
        bin_bias = self.bins[bin_idx]

        if self.use_fw_regression:
            result = pred_reg + bin_bias
        else:
            result = bin_bias
        result = result.clamp(min=self.min_val, max=self.max_val)

        return result

    # pylint: disable=too-many-locals
    def training_loss(self, pred, target):
        """Compute training loss for bin classification and regression.
        
        Args:
            pred: Prediction tensor with shape [..., length].
            target: Target tensor with shape [...].
            
        Returns:
            Tuple of (loss, out_result) where loss is the computed loss
            and out_result is the clamped regression result.
        """
        assert pred.shape[-1] == self.length, \
            f"pred.shape[-1]={pred.shape[-1]} is not equal to self.length={self.length}"

        assert pred.shape[0] == target.shape[0], \
            f"pred.shape[0]={pred.shape[0]} is not equal to target.shape[0]={target.shape[0]}"
        device = pred.device

        pred_reg = (pred[..., 0].sigmoid() * self.reg_scale - self.reg_scale/2.0) * self.step
        pred_bin = pred[..., 1:(1+self.bin_count)]

        diff_bin_target = torch.abs(target[..., None] - self.bins)
        _, bin_idx = torch.min(diff_bin_target, dim=-1)

        bin_bias = self.bins[bin_idx]
        bin_bias.requires_grad = False
        result = pred_reg + bin_bias

        target_bins = torch.full_like(pred_bin, self.cn, device=device)  # targets
        n = pred.shape[0]
        target_bins[range(n), bin_idx] = self.cp

        loss_bin = self.BCEbins(pred_bin, target_bins) # BCE

        if self.use_loss_regression:
            loss_regression = self.MSELoss(result, target)  # MSE
            loss = loss_bin + loss_regression
        else:
            loss = loss_bin

        out_result = result.clamp(min=self.min_val, max=self.max_val)

        return loss, out_result
