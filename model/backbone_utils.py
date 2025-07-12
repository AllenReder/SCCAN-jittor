import jittor as jt
from jittor import nn
import jittor.nn as F
from model.backbone_res import *


class FrozenBatchNorm2d(jt.Module):
    """
    BatchNorm2d where the batch statistics and the affine parameters are fixed.
    """

    def __init__(self, n):
        super(FrozenBatchNorm2d, self).__init__()
        self.weight = jt.ones(n)
        self.bias = jt.zeros(n)
        self.running_mean = jt.zeros(n)
        self.running_var = jt.ones(n)

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        num_batches_tracked_key = prefix + 'num_batches_tracked'
        if num_batches_tracked_key in state_dict:
            del state_dict[num_batches_tracked_key]

    def execute(self, x):
        # move reshapes to the beginning
        # to make it fuser-friendly
        w = self.weight.reshape(1, -1, 1, 1)
        b = self.bias.reshape(1, -1, 1, 1)
        rv = self.running_var.reshape(1, -1, 1, 1)
        rm = self.running_mean.reshape(1, -1, 1, 1)
        eps = 1e-5
        scale = w * jt.rsqrt(rv + eps)
        bias = b - rm * scale
        return x * scale + bias


class BackboneBase(nn.Module):

    def __init__(self, backbone: nn.Module, train_backbone: bool, num_channels: int, return_interm_layers: bool):
        super().__init__()
        for name, parameter in backbone.named_parameters():
            if not train_backbone or 'layer2' not in name and 'layer3' not in name and 'layer4' not in name:
                parameter.requires_grad = False
        if return_interm_layers:
            return_layers = {"layer1": "0", "layer2": "1", "layer3": "2", "layer4": "3"}
        else:
            return_layers = {'layer4': "0"}
        self.backbone = backbone
        self.num_channels = num_channels

    def execute(self, x):
        x = self.backbone(x)
        return x


resnets_dict = {
    'resnet50': (resnet50, 'initmodel/resnet50_v2.pth'),
    'resnet101': (resnet101, 'initmodel/resnet101_v2.pth'),
}


class Backbone(BackboneBase):
    """ResNet backbone with frozen BatchNorm."""

    def __init__(self, name: str,
                 train_backbone: bool,
                 return_interm_layers: bool,
                 dilation: list):
        backbone = resnets_dict[name][0](
            replace_stride_with_dilation=dilation,
            pretrained=resnets_dict[name][1], norm_layer=FrozenBatchNorm2d)
        num_channels = 512 if name in ('resnet18', 'resnet34') else 2048
        super().__init__(backbone, train_backbone, num_channels, return_interm_layers)


if __name__ == "__main__":
    backbone = Backbone('resnet50', train_backbone=False,
                        return_interm_layers=True, dilation=[False, True, True])
    print(backbone)
    x = jt.randn(1, 3, 224, 224)
    out = backbone(x)
    print(out.keys())

    # print(backbone.backbone.conv1.weight)
