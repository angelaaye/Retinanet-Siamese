import math
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.model_zoo as model_zoo

import utils
import losses
from anchors import Anchors
from pth_nms import pth_nms
from utils import BasicBlock, Bottleneck, BBoxTransform, ClipBoxes

def nms(dets, thresh):
    """Dispatch to either CPU or GPU NMS implementations.\
    Accept dets as tensor"""
    return pth_nms(dets, thresh)

model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}

class PyramidFeatures(nn.Module):
    '''
    Compute P3 to P5 using output from ResNet stage (C3 to C5) using top-down and lateral connections.
    P6 is obtained from 3x3 stride-2 conv on C5.
    P7 is obtained from ReLU + 3x3 stride-2 conv on P6. 
    '''
    def __init__(self, C3_size, C4_size, C5_size):
        super(PyramidFeatures, self).__init__()
        
        # Bottom-up layers
        self.conv6 = nn.Conv2d(C5_size, 256, kernel_size=3, stride=2, padding=1)
        self.conv7 = nn.Conv2d( 256, 256, kernel_size=3, stride=2, padding=1)

        # Top-down layers
        self.toplayer = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)

        # Lateral layers
        self.latlayer1 = nn.Conv2d(C5_size, 256, kernel_size=1, stride=1, padding=0)
        self.latlayer2 = nn.Conv2d(C4_size, 256, kernel_size=1, stride=1, padding=0)
        self.latlayer3 = nn.Conv2d(C3_size, 256, kernel_size=1, stride=1, padding=0)

        # Upsampling layers
        #self.upsample = nn.Upsample(scale_factor=2, mode='nearest')

    def forward(self, inputs):

        C3, C4, C5 = inputs
        
        P5 = self.latlayer1(C5)
        P5_upsampled = F.interpolate(P5, scale_factor=2, mode='nearest')
        #P5_upsampled = self.upsample(P5)
        P5 = self.toplayer(P5)

        P4 = self.latlayer2(C4)
        P4 = P5_upsampled + P4
        P4_upsampled = F.interpolate(P4, scale_factor=2, mode='nearest')
        #P4_upsampled = self.upsample(P4)
        P4 = self.toplayer(P4)

        P3 = self.latlayer3(C3)
        P3 = P4_upsampled + P3
        P3 = self.toplayer(P3)

        P6 = self.conv6(C5)
        P7 = self.conv7(F.relu(P6))

        return [P3, P4, P5, P6, P7]


class RegressionModel(nn.Module):
    '''
    Network head consists of 3x3 convolution followed by two sibling 1x1 convolutions
    for classification and regressions to each level of the feature pyramid.
    '''
    def __init__(self, num_features_in, num_anchors=9):
        super(RegressionModel, self).__init__()
        
        self.conv1 = nn.Conv2d(num_features_in, 256, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.conv5 = nn.Conv2d(256, num_anchors*4, kernel_size=3, padding=1)

    def forward(self, x):

        out = F.relu(self.conv1(x))
        out = F.relu(self.conv2(out))
        out = F.relu(self.conv3(out))
        out = F.relu(self.conv4(out))
        out = self.conv5(out)

        # out is Barch x Channels x W x H, with C = 4*num_anchors
        out = out.permute(0, 2, 3, 1)

        # [N, 9*4,H,W] -> [N,H,W, 9*4] -> [N,H*W*9, 4]
        return out.contiguous().view(out.shape[0], -1, 4)

class SiameseNetwork(torch.nn.Module):
    '''
    Implements the Siamese Network defined in Koch's paper. For more detail, please visit
    http://www.cs.utoronto.ca/~gkoch/files/msc-thesis.pdf
    The network consists of 4 convolutional layers, each followed by relu and max pool.
    Lastly, the L1 distance between two images are compared to output a sigmoid output.
    '''
    def __init__(self):
        super(SiameseNetwork, self).__init__()

        self.conv1 = torch.nn.Conv2d(1, 64, kernel_size=10)
        self.conv2 = torch.nn.Conv2d(64, 128, kernel_size=7)
        self.conv3 = torch.nn.Conv2d(128, 128, kernel_size=4)
        self.conv4 = torch.nn.Conv2d(128, 256, kernel_size=4)
        self.pool = torch.nn.MaxPool2d(kernel_size=2)
        self.fc1 = torch.nn.Linear(256 * 6 * 6, 4096)
        self.fc2 = torch.nn.Linear(4096, 1)
        self.sigmoid = torch.nn.Sigmoid()

    def sub_forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = F.relu(self.conv3(x))
        x = self.pool(x)
        x = F.relu(self.conv4(x))
        # x = self.pool(x)         
        # x = x.view(x.shape[0], -1)  
        x = x.view(-1, 256 * 6 * 6)
        x = self.sigmoid(self.fc1(x))
        return x

    def forward(self, x, y):
        x1 = self.sub_forward(x)
        x2 = self.sub_forward(y)
        # L1 distance followed by Sigmoid
        pred = self.sigmoid(self.fc2(torch.abs(x1-x2)))   

        return pred

class ClassificationModel(nn.Module):
    '''
    Network head consists of 3x3 convolution followed by two sibling 1x1 convolutions
    for classification and regressions to each level of the feature pyramid.
    '''
    def __init__(self, num_features_in, num_anchors=9, num_classes=80):
        super(ClassificationModel, self).__init__()

        self.num_classes = num_classes
        self.num_anchors = num_anchors
        
        self.conv1 = nn.Conv2d(num_features_in, 256, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(256, 256, kernel_size=3, padding=1)

        self.conv5 = nn.Conv2d(256, num_anchors*num_classes, kernel_size=3, padding=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):

        out = F.relu(self.conv1(x))
        out = F.relu(self.conv2(out))
        out = F.relu(self.conv3(out))
        out = F.relu(self.conv4(out))
        out = self.sigmoid(self.conv5(out))

        # out is B x C x W x H, with C = n_classes + n_anchors
        out = out.permute(0, 2, 3, 1)
        # batch_size, width, height, channels = out1.shape

        # out2 = out1.view(batch_size, width, height, self.num_anchors, self.num_classes)

        # contiguous() makes a copy of the tensor
        # view() is equivalent to reshape; -1 fills the unknown dimension
        # [N,9*20,H,W] -> [N,H,W,9*20] -> [N,H*W*9,20]
        # return out2.contiguous().view(x.shape[0], -1, self.num_classes)
        return out.contiguous().view(x.shape[0], -1, self.num_classes) #(batch_size, num_anchors*width*height, num_classes)

class ResNet(nn.Module):

    def __init__(self, num_classes, block, layers):
        super(ResNet, self).__init__()
        self.inplanes = 64
        
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        # self.relu = nn.ReLU(inplace=True)
        # self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, planes=64, blocks=layers[0], stride=1)
        self.layer2 = self._make_layer(block, planes=128, blocks=layers[1], stride=2)
        self.layer3 = self._make_layer(block, planes=256, blocks=layers[2], stride=2)
        self.layer4 = self._make_layer(block, planes=512, blocks=layers[3], stride=2)

        if block == BasicBlock:
            fpn_sizes = [self.layer2[layers[1]-1].conv2.out_channels, self.layer3[layers[2]-1].conv2.out_channels, self.layer4[layers[3]-1].conv2.out_channels]
        elif block == Bottleneck:
            fpn_sizes = [self.layer2[layers[1]-1].conv3.out_channels, self.layer3[layers[2]-1].conv3.out_channels, self.layer4[layers[3]-1].conv3.out_channels]

        self.fpn = PyramidFeatures(fpn_sizes[0], fpn_sizes[1], fpn_sizes[2])
        self.regressionModel = RegressionModel(256)
        self.classificationModel = ClassificationModel(256, num_classes=num_classes)
        self.siameseNetwork = SiameseNetwork()

        self.anchors = Anchors()

        self.regressBoxes = BBoxTransform()

        self.clipBoxes = ClipBoxes()
        
        self.focalLoss = losses.FocalLossModified()

        self.cropBoxes = utils.CropBoxes()
                
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

        prior = 0.01
        
        self.classificationModel.conv5.weight.data.fill_(0)
        self.classificationModel.conv5.bias.data.fill_(-math.log((1.0-prior)/prior))

        self.regressionModel.conv5.weight.data.fill_(0)
        self.regressionModel.conv5.bias.data.fill_(0)

        self.freeze_bn()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        # block = Bottleneck, defined in utils.py
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))  # stride = 1, downsample = None

        return nn.Sequential(*layers)

    def freeze_bn(self):
        '''Freeze BatchNorm layers.'''
        for layer in self.modules():
            if isinstance(layer, nn.BatchNorm2d):
                layer.eval()

    def forward(self, inputs):

        if self.training:
            img_batch, annotations, pairs = inputs
        else:
            img_batch, pairs = inputs
            
        c1 = F.relu(self.bn1(self.conv1(img_batch)))  # 2, 64, 320, 320
        c1 = F.max_pool2d(c1, kernel_size=3, stride=2, padding=1) #2, 64, 160, 160
        c2 = self.layer1(c1) # c2 torch.Size([2, 64, 160, 160])
        c3 = self.layer2(c2) # c3 torch.Size([2, 128, 80, 80])
        c4 = self.layer3(c3) # c4 torch.Size([2, 256, 40, 40]) 
        c5 = self.layer4(c4) # c5 torch.Size([2, 512, 20, 20])
        features = self.fpn([c3, c4, c5])

        regression = torch.cat([self.regressionModel(feature) for feature in features], dim=1)
        
        classification = torch.cat([self.classificationModel(feature) for feature in features], dim=1)

        anchors = self.anchors(img_batch)

        if self.training:
            cropped_imgs, stacked_pairs, labels, classification_loss, regression_loss = self.focalLoss(classification, regression, anchors, inputs)
            out = self.siameseNetwork(cropped_imgs.cuda(), stacked_pairs)
            siamese_loss = F.binary_cross_entropy_with_logits(out, labels.cuda())
            return siamese_loss, classification_loss, regression_loss
        else:
            transformed_anchors = self.regressBoxes(anchors, regression)
            transformed_anchors = self.clipBoxes(transformed_anchors, img_batch)

            scores = torch.max(classification, dim=2, keepdim=True)[0]

            scores_over_thresh = (scores>0.05)[0, :, 0]

            if scores_over_thresh.sum() == 0:
                # no boxes to NMS, just return
                return [torch.zeros(0), torch.zeros(0), torch.zeros(0, 4), torch.zeros(0)]

            classification = classification[:, scores_over_thresh, :]
            transformed_anchors = transformed_anchors[:, scores_over_thresh, :]
            scores = scores[:, scores_over_thresh, :]

            anchors_nms_idx = nms(torch.cat([transformed_anchors, scores], dim=2)[0, :, :], 0.5)
            nms_scores, nms_class = classification[0, anchors_nms_idx, :].max(dim=1)

            cropped_imgs = self.cropBoxes(img_batch[0, :, :, :], transformed_anchors[0, anchors_nms_idx, :])
            stacked_pairs = pairs[0, :, :, :].unsqueeze(0).repeat(cropped_imgs.shape[0], 1, 1, 1)
            similarity = self.siameseNetwork(cropped_imgs.cuda(), stacked_pairs)
            return [nms_scores, nms_class, transformed_anchors[0, anchors_nms_idx, :], similarity]



def resnet18(num_classes, pretrained=False, **kwargs):
    """Constructs a ResNet-18 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(num_classes, BasicBlock, [2, 2, 2, 2], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet18'], model_dir='.'), strict=False)
    return model


def resnet34(num_classes, pretrained=False, **kwargs):
    """Constructs a ResNet-34 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(num_classes, BasicBlock, [3, 4, 6, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet34'], model_dir='.'), strict=False)
    return model


def resnet50(num_classes, pretrained=False, **kwargs):
    """Constructs a ResNet-50 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(num_classes, Bottleneck, [3, 4, 6, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet50'], model_dir='.'), strict=False)
    return model

def resnet101(num_classes, pretrained=False, **kwargs):
    """Constructs a ResNet-101 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(num_classes, Bottleneck, [3, 4, 23, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet101'], model_dir='.'), strict=False)
    return model


def resnet152(num_classes, pretrained=False, **kwargs):
    """Constructs a ResNet-152 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(num_classes, Bottleneck, [3, 8, 36, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet152'], model_dir='.'), strict=False)
    return model