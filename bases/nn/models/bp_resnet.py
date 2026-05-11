from copy import deepcopy
import torch
import torch.nn as nn
import math
from .utils import is_fc, is_conv, Scaler
from .base_model import BaseModel
from bases.nn.conv2d import Bias_DenseConv2d as DenseConv2d
from bases.nn.linear import Bias_DenseLinear as DenseLinear 
import torch.nn.init as init

def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1, bern=False):
    """3x3 convolution with padding"""
    return DenseConv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                       padding=dilation, groups=groups, use_bias=False, dilation=dilation,bern = bern)


def conv1x1(in_planes, out_planes, stride=1, bern=False):
    """1x1 convolution"""
    return DenseConv2d(in_planes, out_planes, kernel_size=1, stride=stride, use_bias=False,bern = bern)


def conv1x1_no_prune(in_planes, out_planes, stride=1):
    """1x1 convolution, no pruning"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None, bern=False, model_rate=1):
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride,bern = bern)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes,bern = bern)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride
        self.scaler = Scaler(model_rate ** 0.5) if model_rate != 1 else nn.Identity()

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.scaler(out)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.scaler(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    # Bottleneck in torchvision places the stride for downsampling at 3x3 convolution(self.conv2)
    # while conventional implementation places the stride at the first 1x1 convolution(self.conv1)
    # according to "Deep residual learning for image recognition"https://arxiv.org/abs/1512.03385.
    # This variant is also known as ResNet V1.5 and improves accuracy according to
    # https://ngc.nvidia.com/catalog/model-scripts/nvidia:resnet_50_v1_5_for_pytorch.

    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None, bern=False, model_rate=1):
        super(Bottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, width,bern = bern)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation,bern = bern)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion, bern = bern)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride
        self.scaler = Scaler(model_rate ** 0.5) if model_rate != 1 else nn.Identity()

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.scaler(out)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.scaler(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.scaler(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class ResNet(BaseModel):
    def __init__(self, dict_module: dict = None, block=BasicBlock, layers=(2, 2, 2, 2), num_classes=1000,
                 zero_init_residual=False, groups=1, width_per_group=64, replace_stride_with_dilation=None,
                 norm_layer=None, bern=False, model_rate=1, layer_planes=None):
        self.output_layer_prefix = "fc."
        new_arch = dict_module is None
        if new_arch:
            dict_module = dict()
            if norm_layer is None:
                norm_layer = nn.BatchNorm2d
            self._norm_layer = norm_layer

            if layer_planes is None:
                base_planes = [64, 128, 256, 512]
                scale = model_rate ** 0.5
                layer_planes = [max(1, int(int(math.ceil(scale * p)) / 4) * 4) for p in base_planes]
            self.inplanes = layer_planes[0]
            self.dilation = 1
            if replace_stride_with_dilation is None:
                # each element in the tuple indicates if we should replace
                # the 2x2 stride with a dilated convolution instead
                replace_stride_with_dilation = [False, False, False]
            if len(replace_stride_with_dilation) != 3:
                raise ValueError("replace_stride_with_dilation should be None "
                                 "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
            self.groups = groups
            self.base_width = width_per_group
            dict_module["conv1"] = DenseConv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3,
                                               use_bias=False, bern=bern)
            dict_module["bn1"] = norm_layer(self.inplanes)
            dict_module["relu"] = nn.ReLU(inplace=True)
            dict_module["maxpool"] = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
            dict_module["layer1"] = self._make_layer(block, layer_planes[0], layers[0], bern=bern, model_rate=model_rate)
            dict_module["layer2"] = self._make_layer(block, layer_planes[1], layers[1], stride=2,
                                                     dilate=replace_stride_with_dilation[0], bern=bern, model_rate=model_rate)
            dict_module["layer3"] = self._make_layer(block, layer_planes[2], layers[2], stride=2,
                                                     dilate=replace_stride_with_dilation[1], bern=bern, model_rate=model_rate)
            dict_module["layer4"] = self._make_layer(block, layer_planes[3], layers[3], stride=2,
                                                     dilate=replace_stride_with_dilation[2], bern=bern, model_rate=model_rate)
            dict_module["avgpool"] = nn.AdaptiveAvgPool2d((1, 1))
            dict_module["fc"] = DenseLinear(layer_planes[3] * block.expansion, num_classes, bern=bern)

            self.dict_module = dict_module

        super(ResNet, self).__init__(nn.functional.cross_entropy, dict_module)

        if new_arch:
            self.scaler = Scaler(model_rate ** 0.5) if model_rate != 1 else nn.Identity()
            self.reset_parameters(zero_init_residual)
            init.kaiming_uniform_(self.fc.weight, a=1.5)

    def reset_parameters(self, zero_init_residual):
        for m in self.modules():
            if isinstance(m, DenseConv2d) or isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False, bern=False, model_rate=1):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1_no_prune(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer, bern=bern, model_rate=model_rate))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer, bern=bern, model_rate=model_rate))

        return nn.Sequential(*layers)

    def _forward_impl(self, x):
        # See note [TorchScript super()]
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.scaler(x)
        x = self.fc(x)

        return x

    def forward(self, x):
        return self._forward_impl(x)

    def collect_layers(self):
        self.get_param_layers(self.param_layers, self.param_layer_prefixes)
        self.prunable_layers = [layer for layer in self.param_layers if is_conv(layer) or is_fc(layer)]
        self.prunable_layer_prefixes = [pfx for ly, pfx in zip(self.param_layers, self.param_layer_prefixes) if
                                        is_conv(ly) or is_fc(ly)]

    @staticmethod
    def _block_to_sparse(block):
        assert isinstance(block, BasicBlock) or isinstance(block, Bottleneck)
        new_block = deepcopy(block)
        new_block.conv1 = block.conv1.to_sparse()
        new_block.conv2 = block.conv2.to_sparse()
        if isinstance(block, Bottleneck):
            new_block.conv3 = block.conv3.to_sparse()
        return new_block

    def to_sparse(self):
        new_dict = {}
        for key, module in self.dict_module.items():
            if hasattr(module, "to_sparse"):
                new_dict[key] = module.to_sparse()
                if isinstance(module, DenseLinear):
                    new_dict[key].transpose = True
            elif isinstance(module, nn.Sequential):
                blocks = [self._block_to_sparse(block) for block in module]
                new_dict[key] = nn.Sequential(*blocks)
            else:
                new_dict[key] = deepcopy(module)
        return self.__class__(new_dict)


def _resnet(block, layers, num_classes, bern=False, model_rate=1, layer_planes=None, **kwargs):
    model = ResNet(
        None,
        block,
        layers,
        num_classes=num_classes,
        bern=bern,
        model_rate=model_rate,
        layer_planes=layer_planes,
        **kwargs,
    )
    return model

def resnet10(num_classes=1000, bern=False, model_rate=1, layer_planes=None, norm_layer=None) -> ResNet:
    """ResNet-10: [1, 1, 1, 1]"""
    return _resnet(
        BasicBlock,
        [1, 1, 1, 1],
        num_classes,
        bern=bern,
        model_rate=model_rate,
        layer_planes=layer_planes,
        norm_layer=norm_layer,
    )

def resnet14(num_classes=1000, bern=False, model_rate=1, layer_planes=None, norm_layer=None) -> ResNet:
    """ResNet-14: [1, 1, 2, 2]"""
    return _resnet(
        BasicBlock,
        [1, 1, 2, 2],
        num_classes,
        bern=bern,
        model_rate=model_rate,
        layer_planes=layer_planes,
        norm_layer=norm_layer,
    )

def resnet18(num_classes=1000, bern=False, model_rate=1, layer_planes=None, norm_layer=None) -> ResNet:
    r"""ResNet-18 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_
    """
    return _resnet(
        BasicBlock,
        [2, 2, 2, 2],
        num_classes,
        bern=bern,
        model_rate=model_rate,
        layer_planes=layer_planes,
        norm_layer=norm_layer,
    )


def resnet34(num_classes=1000) -> ResNet:
    r"""ResNet-34 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_.
    """
    return _resnet(BasicBlock, [3, 4, 6, 3], num_classes)


def resnet50(num_classes=1000) -> ResNet:
    r"""ResNet-50 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_
    """
    return _resnet(Bottleneck, [3, 4, 6, 3], num_classes)


def resnet101(num_classes=1000) -> ResNet:
    r"""ResNet-101 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_.
    """
    return _resnet(Bottleneck, [3, 4, 23, 3], num_classes)


def resnet152(num_classes=1000) -> ResNet:
    r"""ResNet-152 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_.
    """
    return _resnet(Bottleneck, [3, 8, 36, 3], num_classes)


def resnext50_32x4d(num_classes=1000) -> ResNet:
    r"""ResNeXt-50 32x4d model from
    `"Aggregated Residual Transformation for Deep Neural Networks" <https://arxiv.org/pdf/1611.05431.pdf>`_.
    """
    kwargs = {'groups': 32,
              'width_per_group': 4}
    return _resnet(Bottleneck, [3, 4, 6, 3], num_classes, **kwargs)


def resnext101_32x8d(num_classes=1000) -> ResNet:
    r"""ResNeXt-101 32x8d model from
    `"Aggregated Residual Transformation for Deep Neural Networks" <https://arxiv.org/pdf/1611.05431.pdf>`_.
    """
    kwargs = {'groups': 32,
              'width_per_group': 8}
    return _resnet(Bottleneck, [3, 4, 23, 3], num_classes, **kwargs)


def wide_resnet50_2(num_classes=1000) -> ResNet:
    r"""Wide ResNet-50-2 model from
    `"Wide Residual Networks" <https://arxiv.org/pdf/1605.07146.pdf>`_.
    The model is the same as ResNet except for the bottleneck number of channels
    which is twice larger in every block. The number of channels in outer 1x1
    convolutions is the same, e.g. last block in ResNet-50 has 2048-512-2048
    channels, and in Wide ResNet-50-2 has 2048-1024-2048.
    """
    kwargs = {'width_per_group': 64 * 2}
    return _resnet(Bottleneck, [3, 4, 6, 3], num_classes, **kwargs)


def wide_resnet101_2(num_classes=1000) -> ResNet:
    r"""Wide ResNet-101-2 model from
    `"Wide Residual Networks" <https://arxiv.org/pdf/1605.07146.pdf>`_.
    The model is the same as ResNet except for the bottleneck number of channels
    which is twice larger in every block. The number of channels in outer 1x1
    convolutions is the same, e.g. last block in ResNet-50 has 2048-512-2048
    channels, and in Wide ResNet-50-2 has 2048-1024-2048.
    """
    kwargs = {'width_per_group': 64 * 2}
    return _resnet(Bottleneck, [3, 4, 23, 3], num_classes, **kwargs)
