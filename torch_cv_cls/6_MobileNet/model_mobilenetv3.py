# !/usr/bin/env python
# -*-coding:utf-8 -*-
"""
# @File       : model_mobilenetv2.py
# @version    ：python 3.9
# @Software   : PyCharm
# @Description：
"""
# ================【功能：】====================
from torch import nn, Tensor
import torch
from torch.nn import functional as F
from functools import partial
from typing import Callable, List, Optional


def _make_divisible(ch, divisor=8, min_ch=None):
    """
    the function is taken from the origianl tf repo
    It ensures that all layers have a channel number that is divisible by 8
    It can be seen here:
    # https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet.py
    """
    if min_ch is None:
        min_ch = divisor
    new_ch = max(min_ch, int(ch + divisor / 2) // divisor * divisor)
    if new_ch < 0.9 * ch:
        new_ch += divisor
    return new_ch


class ConvBNActivation(nn.Sequential):
    def __init__(self, in_channel,
                 out_channel,
                 kernel_size=3,
                 stride=1,
                 groups=1,
                 norm_layer: Optional[Callable[..., nn.Module]] = None,
                 activation_layer: Optional[Callable[..., nn.Module]] = None):
        padding = (kernel_size - 1) // 2
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if activation_layer is None:
            activation_layer = nn.ReLU6
        super(ConvBNActivation, self).__init__(
            nn.Conv2d(in_channels=in_channel,
                      out_channels=out_channel,
                      kernel_size=kernel_size,
                      stride=stride,
                      padding=padding,
                      groups=groups,
                      bias=False),
            norm_layer(out_channel),
            nn.ReLU6(inplace=True)
        )

class SqueezeExcitation(nn.Module):
    def __init__(self, input_c, squeeze_factor: int = 4):
        super(SqueezeExcitation, self).__init__()
        squeeze_c = _make_divisible(input_c // squeeze_factor, 8)
        self.fc1 = nn.Conv2d(input_c, squeeze_c, 1)
        self.fc2 = nn.Conv2d(squeeze_c, input_c, 1)
    def forward(self, x: Tensor):
        scale = F.adaptive_avg_pool2d(x, output_size=(1, 1))
        scale = self.fc1(scale)
        scale = F.relu(scale, inplace=True)
        scale = self.fc2(scale)
        scale = F.hardsigmoid(scale, inplace=True)
        return scale * x



class InvertedResidualConfig:
    def __init__(self, input_c,
                 kernel,
                 expanded_c,
                 out_c,
                 use_se,
                 activation,
                 stride,
                 width_multi):
        self.input_c = self.adjust_channels(input_c, width_multi)
        self.kernel = kernel
        self.expanded_c = self.adjust_channels(expanded_c, width_multi)
        self.out_c = self.adjust_channels(out_c, width_multi)
        self.use_se = use_se
        self.use_hs = activation == "HS"  # 是否使用h-swish激活
        self.stride = stride

    @staticmethod
    def adjust_channels(channes, width_multi):
        return _make_divisible(channes * width_multi, 8)



class InvertedResidual(nn.Module):
    def __init__(self,
                 cnf: InvertedResidualConfig,
                 norm_layer: Callable[..., nn.Module]):
        super(InvertedResidual, self).__init__()

        if cnf.stride not in [1, 2]:
            raise ValueError("illegal stride value.")
        self.use_res_connect = (cnf.stride == 1 and cnf.input_c == cnf.out_c)

        layers = []
        activation_layer = nn.Hardswish if cnf.use_hs else nn.ReLU

        # expand
        if cnf.expanded_c != cnf.input_c:
            layers.append(ConvBNActivation(cnf.input_c,
                                           cnf.expanded_c,
                                           kernel_size=1,
                                           norm_layer=norm_layer,
                                           activation_layer=activation_layer))
        # depthwise
        layers.append(ConvBNActivation(cnf.expanded_c,
                                       cnf.expanded_c,
                                       kernel_size=cnf.kernel,
                                       stride=cnf.stride,
                                       groups=cnf.expanded_c,
                                       norm_layer=norm_layer,
                                       activation_layer=activation_layer))

        if cnf.use_se:
            layers.append(SqueezeExcitation(cnf.expanded_c))

        # project
        layers.append(ConvBNActivation(cnf.expanded_c,
                                       cnf.out_c,
                                       kernel_size=1,
                                       norm_layer=norm_layer,
                                       activation_layer=nn.Identity))
        self.block = nn.Sequential(*layers)
        self.out_channels = cnf.out_c
        self.is_strided = cnf.stride > 1

    def forward(self, x:Tensor):
        result = self.block(x)
        if self.use_res_connect:
            result += x
        return result


class MobileNetV3(nn.Module):
    def __init__(self,
                 inverted_residual_setting,
                 last_channel,
                 num_classes,
                 block: Optional[Callable[..., nn.Module]] = None,
                 norm_layer: Optional[Callable[..., nn.Module]] = None):
        super(MobileNetV3, self).__init__()

        if not inverted_residual_setting:
            raise ValueError("The inverted_residual_setting should not be empty")
        elif not (isinstance(inverted_residual_setting, List) and all([isinstance(s, InvertedResidualConfig) for s in inverted_residual_setting])):
            raise TypeError("The inverted_residual_setting should be List[InvertedResidualConfig]")

        if block is None:
            block = InvertedResidual

        if norm_layer is None:
            norm_layer = partial(nn.BatchNorm2d, eps=0.001, momentum=0.01)

        layers = []
        # build first layer
        firstconv_output_c = inverted_residual_setting[0].input_c
        layers.append(ConvBNActivation(3,
                                       firstconv_output_c,
                                       kernel_size=3,
                                       stride=2,
                                       norm_layer=norm_layer,
                                       activation_layer=nn.Hardswish))
        # build inverted residual block
        for cnf in inverted_residual_setting:
            layers.append(block(cnf, norm_layer))

        # bulid last serval layers
        lastconv_input_c = inverted_residual_setting[-1].out_c
        lastconv_output_c = 6 *  lastconv_input_c
        layers.append(ConvBNActivation(lastconv_input_c,
                                       lastconv_output_c,
                                       kernel_size=1,
                                       norm_layer=norm_layer,
                                       activation_layer=nn.Hardswish))
        self.features = nn.Sequential(*layers)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(nn.Linear(lastconv_output_c, last_channel),
                                        nn.Hardswish(inplace=True),
                                        nn.Dropout(p=0.2, inplace=True),
                                        nn.Linear(last_channel, num_classes))

        # initial weights
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
                elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                    nn.init.ones_(m.weight)
                    nn.init.zeros_(m.bias)
                elif isinstance(m, nn.Linear):
                    nn.init.normal_(m.weigh, 0, 0.01)
                    nn.init.zeros_(m.bias)

    def forward(self, x: Tensor):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

def mobilenet_v3_large(num_classes, reduced_tail):
    """
    Constructs a large MobileNetV3 architecture
    weights_link:
        https://download.pytorch.org/models/mobilenet_v3_large-8738ca79.pth
    args:
        num_claeese: number of classes
        reduced_tail: if True, reduces the channel counts of all feature layer
        between C4 and C5 by 2. It is used to reduce the channel redundancy in the
        backbone for Detection and Segmentation
    """
    width_multi =1.0
    # partial 把一个函数的某些参数固定住， 生成一个新的参数
    bneck_conf = partial(InvertedResidualConfig, width_multi=width_multi)
    adjust_channels = partial(InvertedResidualConfig.adjust_channels, width_multi=width_multi)

    reduce_divider = 2 if reduced_tail else 1

    inverted_residual_setting = [
        # input_c, kernel , expanded_c, out_c, use_se, activation , stride
        bneck_conf(16, 3, 16, 16, False, 'RE', 1),
        bneck_conf(16, 3, 64, 24, False, 'RE', 2),
        bneck_conf(24, 3, 72, 24, False, 'RE', 1),
        bneck_conf(24, 5, 72, 40, True, 'RE', 2),
        bneck_conf(40, 5, 120, 40, True, 'RE', 1),
        bneck_conf(40, 5, 120, 40, True, 'RE', 1),
        bneck_conf(40, 3, 240, 80, False, 'HS', 2),
        bneck_conf(80, 3, 200, 80, False, 'HS', 1),
        bneck_conf(80, 3, 184, 80, False, 'HS', 1),
        bneck_conf(80, 3, 184, 80, False, 'HS', 1),
        bneck_conf(80, 3, 480, 112, True, 'HS', 1),
        bneck_conf(112, 3, 672, 112, True, 'HS', 1),
        bneck_conf(112, 5, 672, 160 // reduce_divider, True, 'HS', 2),
        bneck_conf(160 // reduce_divider, 5, 960 // reduce_divider, 160 // reduce_divider, True, 'HS', 1),
        bneck_conf(160 // reduce_divider, 5, 960 // reduce_divider, 160 // reduce_divider, True, 'HS', 1)]
    last_channel = adjust_channels(1280 // reduce_divider)
    return MobileNetV3(inverted_residual_setting=inverted_residual_setting,
                       last_channel=last_channel,
                       num_classes=num_classes)


def mobilenet_v3_small(num_classes, reduced_tail):
    """
    Constructs a large MobileNetV3 architecture
    weights_link:
       https://download.pytorch.org/models/mobilenet_v3_small-047dcff4.pth
    args:
        num_claeese: number of classes
        reduced_tail: if True, reduces the channel counts of all feature layer
        between C4 and C5 by 2. It is used to reduce the channel redundancy in the
        backbone for Detection and Segmentation
    """
    width_multi = 1.0
    # partial 把一个函数的某些参数固定住， 生成一个新的参数
    bneck_conf = partial(InvertedResidualConfig, width_multi=width_multi)
    adjust_channels = partial(InvertedResidualConfig.adjust_channels, width_multi=width_multi)

    reduce_divider = 2 if reduced_tail else 1

    inverted_residual_setting = [
        # input_c, kernel , expanded_c, out_c, use_se, activation , stride
        bneck_conf(16, 3, 16, 16, True, 'RE', 2),
        bneck_conf(16, 3, 72, 24, False, 'RE', 2),
        bneck_conf(24, 3, 88, 24, False, 'RE', 1),
        bneck_conf(24, 5, 96, 40, True, 'HS', 2),
        bneck_conf(40, 5, 240, 40, True, 'HS', 1),
        bneck_conf(40, 5, 240, 40, True, 'HS', 1),
        bneck_conf(40, 5, 120, 48, True, 'HS', 1),
        bneck_conf(48, 5, 144, 48, True, 'HS', 1),
        bneck_conf(48, 5, 288, 96 // reduce_divider, True, 'HS', 2),
        bneck_conf(96 // reduce_divider, 5, 576 // reduce_divider, 96 // reduce_divider, True, 'HS', 1),
        bneck_conf(96 // reduce_divider, 5, 576 // reduce_divider, 96 // reduce_divider, True, 'HS', 1)]
    last_channel = adjust_channels(1024 // reduce_divider)
    return MobileNetV3(inverted_residual_setting=inverted_residual_setting,
                       last_channel=last_channel,
                       num_classes=num_classes)

# input = torch.rand((8, 3, 224, 224))
# model = mobilenet_v3_small(num_classes=5, reduced_tail=True)
# output = model(input)
# print(output)