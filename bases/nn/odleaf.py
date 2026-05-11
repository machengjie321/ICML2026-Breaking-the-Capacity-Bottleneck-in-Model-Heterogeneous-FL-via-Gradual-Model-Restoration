import numpy as np
from torch import nn
import torch.nn as nn
import torch.nn.functional as F
from bases.nn.odlayer import ODLinear,ODWrapper,ODConv2d,ODGroupNorm
import torch
from torch.nn.functional import binary_cross_entropy_with_logits

class Conv2(nn.Module):
    def __init__(self):
        super(Conv2, self).__init__()
        dict_module = dict()
        hidden_sizes=[32, 64, 2048]
        self.features = nn.Sequential(ODWrapper(ODConv2d(1, hidden_sizes[0], kernel_size=5, padding=2, )),  # 32x28x2
      
                                nn.ReLU(inplace=True),
                                nn.MaxPool2d(2, stride=2),  # 32x14x14
                                ODWrapper(ODConv2d(hidden_sizes[0], hidden_sizes[1], kernel_size=5, padding=2, )),  # 64x14x14
                                nn.ReLU(inplace=True),
                                nn.MaxPool2d(2, stride=2))  # 64x7x7
            
            # print(hidden_sizes)
        self.classifier = nn.Sequential(ODWrapper(ODLinear(hidden_sizes[1] * 7 * 7, hidden_sizes[2])),
                                    nn.ReLU(inplace=True),
                                    ODLinear(hidden_sizes[2], 62, mode="fan_out"))
        
        self.output_layer_prefix = self.output_layer_prefix = "classifier.2."





    def forward(self, x, p=1.0):
        for layer in self.features:
            if isinstance(layer, ODWrapper):
                x = layer(x, p=p)
            else:
                x = layer(x)
        x = x.view(x.size(0), -1)
        for layer in self.classifier:
            if isinstance(layer, ODWrapper):
                x = layer(x, p=p)
            else:
                x = layer(x)
        return x
    
    def loss(self, inputs, labels: torch.IntTensor, rate) -> torch.FloatTensor:
        return binary_cross_entropy_with_logits(self(inputs,rate), labels,)
    
    @torch.no_grad()
    def evaluate(self, test_loader, mode="sum"):
        assert mode in ["sum", "mean"], "mode must be sum or mean"
        self.eval()
        test_loss = 0
        n_correct = 0
        n_total = 0
        device = next(self.parameters()).device

        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = self(inputs)
            batch_loss = binary_cross_entropy_with_logits(outputs, labels)
            test_loss += batch_loss.item()
            labels_predicted = torch.argmax(outputs, dim=1)
            if labels.dim() == 2:
                labels = torch.argmax(labels, dim=1)
            n_total += labels.size(0)
            n_correct += torch.sum(torch.eq(labels_predicted, labels)).item()

        if mode == "mean":
            test_loss /= n_total
        self.train()
        return test_loss, n_correct / n_total


class VGG11(nn.Module):
    def __init__(self,num_classes=10):
        super(VGG11, self).__init__()
        self.config = [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M']
        self.batch_norm = False
        self.features = self._make_feature_layers()
        self.classifier = nn.Sequential(ODWrapper(ODLinear(self.config[-2], self.config[-2], a=0,)),
                                    nn.ReLU(inplace=True),
                                    ODWrapper(ODLinear(self.config[-2], self.config[-2], a=0)),
                                    nn.ReLU(inplace=True),
                                    ODLinear(self.config[-2], num_classes, a=1.5, mode="fan_out"))
        self.output_layer_prefix = self.output_layer_prefix = "classifier.4."





    def _make_feature_layers(self):
        layers = []
        in_channels = 3
        for param in self.config:
            if param == 'M':
                layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
            else:
                if self.batch_norm:     
                    layers.extend([ODWrapper(ODConv2d(in_channels, param, kernel_size=3, padding=1)),

                                   nn.BatchNorm2d(param),
                                   nn.ReLU(inplace=True)])
                else:
                    layers.extend([ODWrapper(ODConv2d(in_channels, param, kernel_size=3, padding=1)),
                                   nn.ReLU(inplace=True)])
                in_channels = param

        return nn.Sequential(*layers)

    def forward(self, x, p=1.0):
        for layer in self.features:
            x = layer(x, p=p) if isinstance(layer, ODWrapper) else layer(x)
        x = x.view(x.size(0), -1)
        for layer in self.classifier:
            x = layer(x, p=p) if isinstance(layer, ODWrapper) else layer(x)
        return x
    
    def loss(self, inputs, labels: torch.IntTensor, rate) -> torch.FloatTensor:
        return binary_cross_entropy_with_logits(self(inputs,rate), labels,)
    
    @torch.no_grad()
    def evaluate(self, test_loader, mode="sum"):
        assert mode in ["sum", "mean"], "mode must be sum or mean"
        self.eval()
        test_loss = 0
        n_correct = 0
        n_total = 0
        device = next(self.parameters()).device

        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = self(inputs)
            batch_loss = binary_cross_entropy_with_logits(outputs, labels)
            test_loss += batch_loss.item()
            labels_predicted = torch.argmax(outputs, dim=1)
            if labels.dim() == 2:
                labels = torch.argmax(labels, dim=1)
            n_total += labels.size(0)
            n_correct += torch.sum(torch.eq(labels_predicted, labels)).item()

        if mode == "mean":
            test_loss /= n_total
        self.train()
        return test_loss, n_correct / n_total






def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    return ODWrapper(ODConv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                               padding=dilation, groups=groups, dilation=dilation, norm=True))

def conv1x1(in_planes, out_planes, stride=1):
    return ODWrapper(ODConv2d(in_planes, out_planes, kernel_size=1, stride=stride, norm=True))

def conv1x1_no_prune(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, norm_layer=None):
        super(BasicBlock, self).__init__()


        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = ODGroupNorm(4, planes)

        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = ODGroupNorm(4, planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x, p=1.0):
        identity = x

        out = self.conv1(x,p)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out,p)
        out = self.bn2(out)

        if self.downsample is not None:
            for layer in self.downsample:
                identity = layer(identity, p=p) if isinstance(layer, ODWrapper) else layer(identity)

        out += identity
        out = self.relu(out)

        return out



    
class ResNet(nn.Module):
    def __init__(self, block=BasicBlock, layers=(2, 2, 2, 2), num_classes=1000, ):
        super(ResNet, self).__init__()


        self.inplanes = 64
        self.conv1 = ODWrapper(ODConv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3, norm=True))
        self.bn1 = ODGroupNorm(4, self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(block, 64, layers[0],)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2,)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2,)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2, )

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = ODLinear(512 * block.expansion, num_classes)
        self.output_layer_prefix = "fc"

    def _make_layer(self, block, planes, blocks, stride=1, norm_layer=None):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                ODGroupNorm(4, planes * block.expansion)
            )

        layers = [ODWrapper(block(self.inplanes, planes, stride, downsample))]
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(ODWrapper(block(self.inplanes, planes,)))

        return nn.Sequential(*layers)
    
    def loss(self, inputs, labels: torch.IntTensor, rate=1.0) -> torch.FloatTensor:
        return nn.functional.cross_entropy(self(inputs,rate), labels,)

    def forward(self, x, p=1.0):


        x = self.conv1(x,p)

            
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        for layer in self.layer1:
            x = layer(x, p=p) if isinstance(layer, ODWrapper) else layer(x)
        for layer in self.layer2:
            x = layer(x, p=p) if isinstance(layer, ODWrapper) else layer(x)
        for layer in self.layer3:
            x = layer(x, p=p) if isinstance(layer, ODWrapper) else layer(x)
        for layer in self.layer4:
            x = layer(x, p=p) if isinstance(layer, ODWrapper) else layer(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x
    
    @torch.no_grad()
    def evaluate(self, test_loader, mode="sum"):
        assert mode in ["sum", "mean"], "mode must be sum or mean"
        self.eval()
        test_loss = 0
        n_correct = 0
        n_total = 0
        device = next(self.parameters()).device

        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = self(inputs)
            
            batch_loss = nn.functional.cross_entropy(outputs, labels)
            test_loss += batch_loss.item()
            labels_predicted = torch.argmax(outputs, dim=1)
            if labels.dim() == 2:
                labels = torch.argmax(labels, dim=1)
            n_total += labels.size(0)
            n_correct += torch.sum(torch.eq(labels_predicted, labels)).item()

        if mode == "mean":
            test_loss /= n_total
        self.train()
        return test_loss, n_correct / n_total


def resnet18(num_classes=1000):
    return ResNet(BasicBlock, [2, 2, 2, 2], num_classes=1000)








