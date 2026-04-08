"""YOLOR PyTorch utils."""

import copy
import logging
import math
import time

import torch
from torch import nn
import torch.nn.functional as F

try:
    import thop  # for FLOPS computation
except ImportError:
    thop = None

logger = logging.getLogger(__name__)


def time_synchronized():
    """Return pytorch-accurate time."""
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    return time.time()


def initialize_weights(model):
    """Initialize model weights."""
    for m in model.modules():
        t = type(m)
        if t is nn.Conv2d:
            pass  # nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        elif t is nn.BatchNorm2d:
            m.eps = 1e-3
            m.momentum = 0.03
        elif t in [nn.Hardswish, nn.LeakyReLU, nn.ReLU, nn.ReLU6]:
            m.inplace = True


def fuse_conv_and_bn(conv, bn):
    """Fuse convolution and batchnorm layers.
    
    https://tehnokv.com/posts/fusing-batchnorm-and-conv/
    """
    fusedconv = nn.Conv2d(conv.in_channels,
                          conv.out_channels,
                          kernel_size=conv.kernel_size,
                          stride=conv.stride,
                          padding=conv.padding,
                          groups=conv.groups,
                          bias=True).requires_grad_(False).to(conv.weight.device)

    # prepare filters
    w_conv = conv.weight.clone().view(conv.out_channels, -1)
    w_bn = torch.diag(bn.weight.div(torch.sqrt(bn.eps + bn.running_var)))
    fusedconv.weight.copy_(torch.mm(w_bn, w_conv).view(fusedconv.weight.shape))

    # prepare spatial bias
    b_conv = torch.zeros(conv.weight.size(0), device=conv.weight.device) if conv.bias is None else conv.bias
    b_bn = bn.bias - bn.weight.mul(bn.running_mean).div(torch.sqrt(bn.running_var + bn.eps))
    fusedconv.bias.copy_(torch.mm(w_bn, b_conv.reshape(-1, 1)).reshape(-1) + b_bn)

    return fusedconv


def model_info(model, verbose=False, img_size=640):
    """Log model information.
    
    Args:
        model: PyTorch model to inspect
        verbose: If True, print detailed parameter information
        img_size: Image size for FLOPS calculation
    """
    n_p = sum(x.numel() for x in model.parameters())  # number parameters
    n_g = sum(x.numel() for x in model.parameters() if x.requires_grad)  # number gradients
    if verbose:
        print(f"{'layer':>5} {'name':>40} {'gradient':>9} {'parameters':>12} {'shape':>20} {'mu':>10} {'sigma':>10}")
        for i, (name, p) in enumerate(model.named_parameters()):
            name = name.replace('module_list.', '')
            print(
                f"{i:5g} {name:>40} {str(p.requires_grad):>9} "
                f"{p.numel():12g} {str(list(p.shape)):>20} {p.mean():10.3g} {p.std():10.3g}"
            )

    try:  # FLOPS
        if thop is None:
            raise ImportError("thop not available")
        stride = max(int(model.stride.max()), 32) if hasattr(model, 'stride') else 32
        img = torch.zeros((1, model.yaml.get('ch', 3), stride, stride), device=next(model.parameters()).device)  # input
        flops = thop.profile(copy.deepcopy(model), inputs=(img,), verbose=False)[0] / 1E9 * 2  # stride GFLOPS
        img_size = img_size if isinstance(img_size, list) else [img_size, img_size]  # expand if int/float
        fs = f', {flops * img_size[0] / stride * img_size[1] / stride:.1f} GFLOPS'  # 640x640 GFLOPS
    except (ImportError, OSError):
        fs = ''

    logger.info("Model Summary: %d layers, %d parameters, %d gradients%s",
                len(list(model.modules())), n_p, n_g, fs)


def scale_img(img, ratio=1.0, same_shape=False, gs=32):  # img(16,3,256,416)
    """Scales img(bs,3,y,x) by ratio constrained to gs-multiple.
    
    Args:
        img: Input image tensor
        ratio: Scaling ratio
        same_shape: If True, maintain original shape
        gs: Grid size for rounding
    
    Returns:
        Scaled image tensor
    """
    if ratio == 1.0:
        return img
    h, w = img.shape[2:]
    s = (int(h * ratio), int(w * ratio))  # new size
    img = F.interpolate(img, size=s, mode='bilinear', align_corners=False)  # resize
    if not same_shape:  # pad/crop img
        h, w = [math.ceil(x * ratio / gs) * gs for x in (h, w)]
    return F.pad(img, [0, w - s[1], 0, h - s[0]], value=0.447)  # value = imagenet mean  # pylint: disable=not-callable


def copy_attr(a, b, include=(), exclude=()):
    """Copy attributes from b to a, with optional include/exclude lists.
    
    Args:
        a: Target object to copy attributes to
        b: Source object to copy attributes from
        include: Tuple of attribute names to include (empty means all)
        exclude: Tuple of attribute names to exclude
    
    Returns:
        None: Modifies object a in place
    """
    for k, v in b.__dict__.items():
        if (include and k not in include) or k.startswith('_') or k in exclude:
            continue
        setattr(a, k, v)

# pylint: disable=too-few-public-methods
class BatchNormXd(torch.nn.modules.batchnorm.BatchNorm):
    """BatchNorm class that accepts any number of dimensions."""

    def _check_input_dim(self, _):
        # The only difference between BatchNorm1d, BatchNorm2d, BatchNorm3d, etc
        # is this method that is overwritten by the sub-class
        # This original goal of this method was for tensor sanity checks
        # If you're ok bypassing those sanity checks (eg. if you trust your inference
        # to provide the right dimensional inputs), then you can just use this method
        # for easy conversion from SyncBatchNorm
        # (unfortunately, SyncBatchNorm does not store the original class - if it did
        #  we could return the one that was originally created)
        return
# pylint: disable=attribute-defined-outside-init
def revert_sync_batchnorm(module):
    """Revert SyncBatchNorm to regular BatchNorm.
    
    This is similar to the function in PyTorch:
    https://github.com/pytorch/pytorch/blob/c8b3686a3e4ba63dc59e5dcfe5db3430df256833/torch/nn/modules/batchnorm.py#L679
    
    Args:
        module: Module to convert
    
    Returns:
        Module with SyncBatchNorm replaced by BatchNormXd
    """
    module_output = module
    if isinstance(module, torch.nn.modules.batchnorm.SyncBatchNorm):
        module_output = BatchNormXd(module.num_features,
                                                  module.eps, module.momentum,
                                                  module.affine,
                                                  module.track_running_stats)
        if module.affine:
            with torch.no_grad():
                module_output.weight = module.weight
                module_output.bias = module.bias
        module_output.running_mean = module.running_mean
        module_output.running_var = module.running_var
        module_output.num_batches_tracked = module.num_batches_tracked
        if hasattr(module, "qconfig"):
            module_output.qconfig = module.qconfig
    for name, child in module.named_children():
        module_output.add_module(name, revert_sync_batchnorm(child))
    del module
    return module_output


class TracedModel(nn.Module):
    """A traced PyTorch model for TorchScript export."""

    def __init__(self, model=None, device=None, img_size=(640, 640)):
        super().__init__()

        print(" Convert model to Traced-model... ")
        self.stride = model.stride
        self.names = model.names
        self.model = model

        self.model = revert_sync_batchnorm(self.model)
        self.model.to('cpu')
        self.model.eval()

        self.detect_layer = self.model.model[-1]
        self.model.traced = True

        rand_example = torch.rand(1, 3, img_size, img_size)

        traced_script_module = torch.jit.trace(self.model, rand_example, strict=False)
        #traced_script_module = torch.jit.script(self.model)
        traced_script_module.save("traced_model.pt")
        print(" traced_script_module saved! ")
        self.model = traced_script_module
        self.model.to(device)
        self.detect_layer.to(device)
        print(" model is traced! \n")

    def forward(self, x, augment=None, profile=None):  # pylint: disable=unused-argument
        """Forward pass through the traced model.
        
        Args:
            x: Input tensor
            augment: If True, apply test-time augmentation (unused)
            profile: If True, enable profiling (unused)
        
        Returns:
            Model output
        """
        out = self.model(x)
        out = self.detect_layer(out)
        return out
