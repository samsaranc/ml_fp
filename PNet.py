from torch.autograd import Variable
from torch.utils.data.sampler import Sampler
import numpy as np
import torch.nn as nn
import torch
from  torch.nn.parameter import Parameter
import math, random
import torch.utils.model_zoo as model_zoo

__all__ = ['ResNet', 'resnetP']

model_urls = {
'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth'
}

def conv3x3(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)

class BasicBlock(nn.Module):
    expansion = 1
    
    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride
        
        self.drop_rate = 0.4
        self.drop_out = nn.Dropout(p=self.drop_rate)
        
    def d_rate(self,r):
        self.drop_rate = r
        self.drop_out = nn.Dropout(p=r)
        
    def forward(self, x):
        residual = x
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        
        out = self.drop_out(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        
        
        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out

class ResNet(nn.Module):
    def __init__(self, block, layers, num_classes=1000):
        self.inplanes = 64
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AvgPool2d(7)
        self.fc = nn.Linear(512 * block.expansion, num_classes)
        
        self.drop_rate = 0
        self.drop_out = nn.Dropout(p=self.drop_rate)
    
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
                
    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)
    
    def d_rate(self,r):
        print('drop rate: {}'.format(r))
        self.drop_rate = r
        self.drop_out = nn.Dropout(p=r)
        
        self.layer1[0].d_rate(r)
        self.layer2[0].d_rate(r)
        self.layer3[0].d_rate(r)
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.drop_out(x)
        x = self.layer2(x)
        x = self.drop_out(x)
        x = self.layer3(x)
        x = self.drop_out(x)
        x = self.layer4(x)
        
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x

def resnetP(F,pretrained=True, **kwargs):
    """Constructs a ResNet-18 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(BasicBlock, [2, 2, 2, 2], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet18']))
        print("pretrained? yeet")
    model.fc = nn.Linear(512, F)
    return model
# def resnetP(F):
#     model = ResNet(BasicBlock, [1, 1, 1, 1], num_classes=F)
#     return model

            
class PSampler(Sampler):
    def __init__(self,bookmark,balance=True):
        if balance==False:
            self.idx = np.arange(bookmark[0][0],bookmark[-1][1]).tolist()
            return
        
        self.bookmark = bookmark
        interval_list = []
        sp_list_sp = []
        
        # find the max interval
        len_max = 0
        for b in bookmark:
            interval_list.append(np.arange(b[0],b[1]))
            if b[1]-b[0] > len_max: len_max = b[1]-b[0]
            
        for l in interval_list:
            if l.shape[0]<len_max:
                l_ext = np.random.choice(l,len_max-l.shape[0])
                l_ext = np.concatenate((l, l_ext), axis=0)
                l_ext = np.random.permutation(l_ext)
            else:
                l_ext = np.random.permutation(l)
            sp_list_sp.append(l_ext)

        self.idx = np.vstack(sp_list_sp).T.reshape((1,-1)).flatten().tolist()
            
    def __iter__(self):
        return iter(self.idx)
    
    def __len__(self):
        return len(self.idx)
    
