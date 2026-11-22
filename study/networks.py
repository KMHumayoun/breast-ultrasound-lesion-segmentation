"""Compact U-Net and documented torchvision DeepLab transfer baselines."""
import torch
from torch import nn
from torch.nn import functional as F
from torchvision.models.segmentation import deeplabv3_mobilenet_v3_large,DeepLabV3_MobileNet_V3_Large_Weights


class Block(nn.Sequential):
    def __init__(self,inputs,outputs):
        super().__init__(nn.Conv2d(inputs,outputs,3,padding=1,bias=False),nn.GroupNorm(4,outputs),nn.ReLU(),
                         nn.Conv2d(outputs,outputs,3,padding=1,bias=False),nn.GroupNorm(4,outputs),nn.ReLU())


class CompactUNet(nn.Module):
    def __init__(self):
        super().__init__();self.e1=Block(1,16);self.e2=Block(16,32);self.e3=Block(32,64);self.bottom=Block(64,128)
        self.d3=Block(192,64);self.d2=Block(96,32);self.d1=Block(48,16);self.head=nn.Conv2d(16,1,1)

    def forward(self,x):
        a=self.e1(x);b=self.e2(F.max_pool2d(a,2));c=self.e3(F.max_pool2d(b,2));z=self.bottom(F.max_pool2d(c,2))
        z=self.d3(torch.cat([F.interpolate(z,size=c.shape[-2:],mode='bilinear',align_corners=False),c],dim=1))
        z=self.d2(torch.cat([F.interpolate(z,size=b.shape[-2:],mode='bilinear',align_corners=False),b],dim=1))
        z=self.d1(torch.cat([F.interpolate(z,size=a.shape[-2:],mode='bilinear',align_corners=False),a],dim=1))
        return self.head(z)


class BinaryDeepLab(nn.Module):
    def __init__(self,pretrained=False,frozen_backbone=False):
        super().__init__();weights=DeepLabV3_MobileNet_V3_Large_Weights.COCO_WITH_VOC_LABELS_V1 if pretrained else None
        self.net=deeplabv3_mobilenet_v3_large(weights=weights,weights_backbone=None,progress=True)
        self.net.classifier[-1]=nn.Conv2d(self.net.classifier[-1].in_channels,1,1);self.net.aux_classifier=None
        if frozen_backbone:
            for parameter in self.net.backbone.parameters():parameter.requires_grad=False
        self.register_buffer('mean',torch.tensor([.485,.456,.406]).view(1,3,1,1))
        self.register_buffer('std',torch.tensor([.229,.224,.225]).view(1,3,1,1))

    def forward(self,x):return self.net((x.repeat(1,3,1,1)-self.mean)/self.std)['out']


def freeze_batch_statistics(model):
    # Small medical batches are unsuitable for updating pretrained BatchNorm statistics.
    for module in model.modules():
        if isinstance(module,nn.BatchNorm2d):module.eval()


def make_model(name):
    if name in ['unet_scratch','unet_no_augmentation']:return CompactUNet()
    if name=='deeplab_scratch':return BinaryDeepLab()
    if name=='deeplab_pretrained_frozen':return BinaryDeepLab(True,True)
    if name=='deeplab_pretrained_finetune':return BinaryDeepLab(True,False)
    raise ValueError(name)


def segmentation_loss(logits,mask):
    probability=torch.sigmoid(logits);intersection=(probability*mask).flatten(1).sum(1)
    dice=(2*intersection+1e-6)/(probability.flatten(1).sum(1)+mask.flatten(1).sum(1)+1e-6)
    return .5*F.binary_cross_entropy_with_logits(logits,mask)+.5*(1-dice).mean()
