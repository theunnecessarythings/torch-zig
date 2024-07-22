from torchvision import models

models = {
    "alexnet": models.alexnet,
    "convnext_tiny": models.convnext_tiny,
    "convnext_small": models.convnext_small,
    "convnext_large": models.convnext_large,
    "convnext_base": models.convnext_base,
    "densenet121": models.densenet121,
    "densenet169": models.densenet169,
    "densenet201": models.densenet201,
    "densenet161": models.densenet161,
    "efficientnet_b0": models.efficientnet_b0,
    "efficientnet_b1": models.efficientnet_b1,
    "efficientnet_b2": models.efficientnet_b2,
    "efficientnet_b3": models.efficientnet_b3,
    "efficientnet_b4": models.efficientnet_b4,
    "efficientnet_b5": models.efficientnet_b5,
    "efficientnet_b6": models.efficientnet_b6,
    "efficientnet_b7": models.efficientnet_b7,
    "efficientnet_v2s": models.efficientnet_v2_s,
    "efficientnet_v2m": models.efficientnet_v2_m,
    "efficientnet_v2l": models.efficientnet_v2_l,
    "googlenet": models.googlenet,
    "inception_v3": models.inception_v3,
    "mnasnet0_5": models.mnasnet0_5,
    "mnasnet0_75": models.mnasnet0_75,
    "mnasnet1_0": models.mnasnet1_0,
    "mnasnet1_3": models.mnasnet1_3,
    "mobilenet_v2": models.mobilenet_v2,
    "mobilenet_v3_large": models.mobilenet_v3_large,
    "mobilenet_v3_small": models.mobilenet_v3_small,
    "resnet18": models.resnet18,
    "resnet34": models.resnet34,
    "resnet50": models.resnet50,
    "resnet101": models.resnet101,
    "resnet152": models.resnet152,
    "resnext50_32x4d": models.resnext50_32x4d,
    "resnext101_32x8d": models.resnext101_32x8d,
    "wide_resnet50_2": models.wide_resnet50_2,
    "wide_resnet101_2": models.wide_resnet101_2,
    "shufflenet_v2_x0_5": models.shufflenet_v2_x0_5,
    "shufflenet_v2_x1_0": models.shufflenet_v2_x1_0,
    "shufflenet_v2_x1_5": models.shufflenet_v2_x1_5,
    "shufflenet_v2_x2_0": models.shufflenet_v2_x2_0,
    "squeezenet1_0": models.squeezenet1_0,
    "squeezenet1_1": models.squeezenet1_1,
    "vgg11": models.vgg11,
    "vgg11_bn": models.vgg11_bn,
    "vgg13": models.vgg13,
    "vgg13_bn": models.vgg13_bn,
    "vgg16": models.vgg16,
    "vgg16_bn": models.vgg16_bn,
    "vgg19": models.vgg19,
    "vgg19_bn": models.vgg19_bn,
    "vit_b_16": models.vit_b_16,
    "vit_b_32": models.vit_b_32,
    "vit_l_16": models.vit_l_16,
    "vit_l_32": models.vit_l_32,
    "vit_h_14": models.vit_h_14,
    "swin_t": models.swin_t,
    "swin_s": models.swin_s,
    "swin_b": models.swin_b,
    "swin_v2_t": models.swin_v2_t,
    "swin_v2_s": models.swin_v2_s,
    "swin_v2_b": models.swin_v2_b,
}

param_dict = []

for model_name, model_fn in models.items():
    model = model_fn()
    params = []
    for key, param in model.state_dict().items():
        params.append([key, param.shape])
    param_dict.append([model_name, params])

import json

with open("params.json", "w") as f:
    json.dump(param_dict, f)
