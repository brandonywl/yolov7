"""Experimental models for YOLOv7."""

import numpy as np
import torch
from torch import nn

from yolov7.models.common import Conv


class CrossConv(nn.Module):
    """Cross Convolution Downsample."""

    def __init__(self, c1, c2, k=3, s=1, g=1, e=1.0, shortcut=False):  # pylint: disable=too-many-arguments, too-many-positional-arguments
        """Initialize CrossConv.

        Args:
            c1: Input channels
            c2: Output channels
            k: Kernel size
            s: Stride
            g: Groups
            e: Expansion factor
            shortcut: Use shortcut connection
        """
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, (1, k), (1, s))
        self.cv2 = Conv(c_, c2, (k, 1), (s, 1), g=g)
        self.add = shortcut and c1 == c2

    def forward(self, x):
        """Forward pass through CrossConv."""
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))


class MixConv2d(nn.Module):
    """Mixed Depthwise Conv https://arxiv.org/abs/1907.09595."""

    def __init__(self, c1, c2, k=(1, 3), s=1, equal_ch=True):  # pylint: disable=too-many-arguments, too-many-positional-arguments
        """Initialize MixConv2d.

        Args:
            c1: Input channels
            c2: Output channels
            k: Kernel sizes
            s: Stride
            equal_ch: Equal channels per group
        """
        super().__init__()
        groups = len(k)
        if equal_ch:  # equal c_ per group
            i = torch.linspace(0, groups - 1E-6, c2).floor()  # c2 indices
            c_ = [(i == g).sum() for g in range(groups)]  # intermediate channels
        else:  # equal weight.numel() per group
            b = [c2] + [0] * groups
            a = np.eye(groups + 1, groups, k=-1)
            a -= np.roll(a, 1, axis=1)
            a *= np.array(k) ** 2
            a[0] = 1
            c_ = np.linalg.lstsq(a, b, rcond=None)[0].round()  # solve for equal weight indices, ax = b

        self.m = nn.ModuleList([nn.Conv2d(c1, int(c_[g]), k[g], s, k[g] // 2, bias=False) for g in range(groups)])
        self.bn = nn.BatchNorm2d(c2)
        self.act = nn.LeakyReLU(0.1, inplace=True)

    def forward(self, x):
        """Forward pass through MixConv2d."""
        return x + self.act(self.bn(torch.cat([m(x) for m in self.m], 1)))


class Ensemble(nn.ModuleList):
    """Ensemble of models."""

    def __init__(self):
        """Initialize Ensemble."""
        super().__init__()

    def forward(self, x, augment=False):
        """Forward pass through Ensemble.

        Args:
            x: Input tensor
            augment: Whether to augment

        Returns:
            Tuple of (output, None)
        """
        y = []
        for module in self:
            y.append(module(x, augment)[0])
        # y = torch.stack(y).max(0)[0]  # max ensemble
        # y = torch.stack(y).mean(0)  # mean ensemble
        y = torch.cat(y, 1)  # nms ensemble
        return y, None  # inference, train output


def attempt_load_state_dict(models, weights, map_location=None):
    """Load an ensemble of models weights=[a,b,c] or a single model weights=[a] or weights=a.

    Args:
        models: Model or list of models
        weights: Weight file path or list of paths
        map_location: Device to load weights to

    Returns:
        Loaded model(s) and class names
    """
    ensemble_model = Ensemble()
    models = models if isinstance(models, list) else [models]
    weights = weights if isinstance(weights, list) else [weights]
    class_names = []
    for i, w in enumerate(weights):
        checkpoint = torch.load(w, map_location=map_location)
        model = models[i]
        model.fuse()
        model.load_state_dict(checkpoint['state_dict'])
        model.eval()
        ensemble_model.append(model)
        class_names.append(checkpoint['class_names'])

    # Compatibility updates
    for m in ensemble_model.modules():
        if isinstance(m, (nn.Hardswish, nn.LeakyReLU, nn.ReLU, nn.ReLU6, nn.SiLU)):
            m.inplace = True  # pytorch 1.7.0 compatibility
        elif isinstance(m, nn.Upsample):
            m.recompute_scale_factor = None  # torch 1.11.0 compatibility
        elif isinstance(m, Conv):
            m._non_persistent_buffers_set = set()  # pytorch 1.6.0 compatibility pylint: disable=protected-access

    if len(ensemble_model) == 1:
        return ensemble_model[-1], class_names[-1]  # return model

    print(f"Ensemble created with {weights}")
    for k in ['names', 'stride']:
        setattr(ensemble_model, k, getattr(ensemble_model[-1], k))
    return ensemble_model, class_names  # return ensemble
