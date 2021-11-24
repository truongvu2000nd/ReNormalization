import torchvision
import torch.nn as nn

from functools import partial

from norm_layer import LogGroupNorm, LogBatchNorm
from models import MLP


def create_mlp_model(norm_type, **kwargs):
    assert norm_type in ["bn", "ln", "gn"]

    if norm_type == "bn":
        norm_layer = partial(LogBatchNorm, input_type="1d")
    elif norm_type == "ln":
        norm_layer = partial(LogGroupNorm, 1)
    else:
        norm_layer = partial(LogGroupNorm, 32)
    model = MLP(out_dim=10, norm_layer=norm_layer, **kwargs)
    return model


def create_resnet_model(norm_type):
    assert norm_type in ["bn", "ln", "gn"]

    if norm_type == "bn":
        norm_layer = LogBatchNorm
    elif norm_type == "ln":
        norm_layer = partial(LogGroupNorm, 1)
    else:
        norm_layer = partial(LogGroupNorm, 32)
    model = torchvision.models.resnet18(pretrained=False, num_classes=10, norm_layer=norm_layer)
    model.conv1 = nn.Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
    model.maxpool = nn.Identity()
    return model


if __name__ == '__main__':
    from torchinfo import summary

    model = create_resnet_model("bn")
    summary(model, (1, 3, 32, 32))
    # track_buffers = ["before_mean", "before_var", "after_mean", "after_var", "after_affine_mean", "after_affine_var"]
    # log_dict = {}
    # for buffer in track_buffers:
    #     log_dict[buffer] = {} 
    
    # for name, p in model.named_buffers():
    #     print(name)
            