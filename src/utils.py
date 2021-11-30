import torchvision
import torch.nn as nn

from functools import partial

from norm_layer import LogGroupNorm, LogBatchNorm, LogReGroupNorm
from models import MLP


# def create_mlp_model(norm_type, **kwargs):
#     assert norm_type in ["bn", "ln", "gn"]

#     if norm_type == "bn":
#         norm_layer = partial(LogBatchNorm, input_type="1d")
#     elif norm_type == "ln":
#         norm_layer = partial(LogGroupNorm, 1)
#     elif norm_type == "gn":
#         norm_layer = partial(LogGroupNorm, 32)
#     else:
#         raise NotImplementedError

#     model = MLP(out_dim=10, norm_layer=norm_layer, **kwargs)
#     return model


def create_resnet18_model(norm_type, **kwargs):
    assert norm_type in ["bn", "ln", "gn", "reln", "regn"]

    if norm_type == "bn":
        norm_layer = LogBatchNorm
    elif norm_type == "ln":
        norm_layer = partial(LogGroupNorm, 1, **kwargs)
    elif norm_type == "gn":
        norm_layer = partial(LogGroupNorm, 32, **kwargs)
    elif norm_type == "reln":
        norm_layer = partial(LogReGroupNorm, 1, **kwargs)
    elif norm_type == "regn":
        norm_layer = partial(LogReGroupNorm, 32, **kwargs)
    else:
        raise NotImplementedError

    model = torchvision.models.resnet18(pretrained=False, num_classes=10, norm_layer=norm_layer)
    model.conv1 = nn.Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
    model.maxpool = nn.Identity()
    return model


# def create_vgg16_model(norm_type, **kwargs):
#     assert norm_type in ["bn", "ln", "gn"]

#     if norm_type == "bn":
#         norm_layer = LogBatchNorm
#     elif norm_type == "ln":
#         norm_layer = partial(LogGroupNorm, 1, **kwargs)
#     elif norm_type == "gn":
#         norm_layer = partial(LogGroupNorm, 32, **kwargs)
#     elif norm_type == "reln":
#         norm_layer = partial(LogReGroupNorm, 1, **kwargs)
#     elif norm_type == "regn":
#         norm_layer = partial(LogReGroupNorm, 32, **kwargs)
#     else:
#         raise NotImplementedError

#     model = torchvision.models.vgg16(pretrained=False, num_classes=10, norm_layer=norm_layer)
#     return model


if __name__ == '__main__':
    from torchinfo import summary

    model = create_resnet18_model("regn", r=1.2)
    # summary(model, (1, 3, 32, 32))
    # print(list(name for name,_ in model.named_buffers()))
            