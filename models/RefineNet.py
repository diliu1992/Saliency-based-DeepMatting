import torch.nn as nn
import torch.nn.functional as F
import torch
from torchvision import models
from torchvision.models.resnet import model_urls, BasicBlock, Bottleneck
import torch.utils.model_zoo as model_zoo
import time


class RDFNet(nn.Module):
    def __init__(self, img_depths=[512, 256, 128, 64], conv_dims=[512, 256, 256, 256], use_cuda=True,
                 semantic_labels_nbr=2, use_batch_norm=False):
        super(RDFNet, self).__init__()
        self.nbr_paths = len(img_depths)
        self.img_depths = img_depths
        self.use_cuda = use_cuda
        self.use_batch_norm = use_batch_norm
        self.semantic_labels_nbr = semantic_labels_nbr
        self.conv_dims = conv_dims
        if self.use_cuda:
            self.resnet_rgb = ModelResNet().cuda()
            # self.resnet_d = ModelResNet().cuda()
            self.mmfnet = MMFNet(img_depths=self.img_depths, conv_dims=self.conv_dims,
                                 use_batch_norm=self.use_batch_norm).cuda()
            self.refinenet = RefineNet(conv_dims=self.conv_dims, use_cuda=self.use_cuda,
                                       semantic_labels_nbr=self.semantic_labels_nbr,
                                       use_batch_norm=self.use_batch_norm).cuda()
        else:
            self.resnet_rgb = ModelResNet()
            # self.resnet_d = ModelResNet()
            self.mmfnet = MMFNet(img_depths=self.img_depths, conv_dims=self.conv_dims,
                                 use_batch_norm=self.use_batch_norm)
            self.refinenet = RefineNet(conv_dims=self.conv_dims, use_cuda=self.use_cuda,
                                       semantic_labels_nbr=self.semantic_labels_nbr,
                                       use_batch_norm=self.use_batch_norm)

    def forward(self, x):
        self.resnet_out_rgb = self.resnet_rgb(x)
        # self.resnet_out_d = self.resnet_d(y)

        self.inputs_rgb = [self.resnet_rgb.x4, self.resnet_rgb.x3, self.resnet_rgb.x2, self.resnet_rgb.x1]
        # self.inputs_d = [self.resnet_d.x4, self.resnet_d.x3, self.resnet_d.x2, self.resnet_d.x1]

        self.mmfnet_out = self.mmfnet(self.inputs_rgb)
        self.refinenet_out = self.refinenet(self.mmfnet_out)
        out = F.upsample(self.refinenet_out, x.size()[2:], mode='bilinear', align_corners=True)
        return out


class ModelResNet(models.ResNet):
    def __init__(self, **kwargs):
        super(ModelResNet, self).__init__(BasicBlock, [3, 4, 6, 3], **kwargs)
        self.load_state_dict(model_zoo.load_url(model_urls['resnet34']))

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        self.x0 = x
        x = self.layer1(x)
        self.x1 = x
        x = self.layer2(x)
        self.x2 = x
        x = self.layer3(x)
        self.x3 = x
        x = self.layer4(x)
        self.x4 = x
        return x


class MMFNet(nn.Module):
    def __init__(self, img_depths=[512, 256, 128, 64], conv_dims=[512, 256, 256, 256], use_batch_norm=False):
        super(MMFNet, self).__init__()
        self.img_depths = img_depths
        self.use_batch_norm = use_batch_norm
        self.nbr_paths = len(self.img_depths)
        self.pathsRGB = []
        self.pathsD = []
        self.pathsRP = []
        self.cv_dims = conv_dims
        for i in range(self.nbr_paths):
            path_rgb = nn.Sequential(
                nn.Dropout2d(p=0.5),
                nn.Conv2d(self.img_depths[i], self.cv_dims[i], 1, stride=1, padding=0, bias=False)).cuda()
                # RCU(conv_dim=self.cv_dims[i]),
                # RCU(conv_dim=self.cv_dims[i]),
                # nn.Conv2d(self.cv_dims[i], self.cv_dims[i], 3, stride=1, padding=1, bias=False)).cuda()
            self.pathsRGB.append(path_rgb)
            # path_d = nn.Sequential(
            #     nn.Dropout2d(p=0.5),
            #     nn.Conv2d(self.img_depths[i], self.cv_dims[i], 1, stride=1, padding=0, bias=False),
            #     RCU(conv_dim=self.cv_dims[i]),
            #     RCU(conv_dim=self.cv_dims[i]),
            #     nn.Conv2d(self.cv_dims[i], self.cv_dims[i], 3, stride=1, padding=1, bias=False)).cuda()
            # self.pathsD.append(path_d)
            #
            # self.pathsRP.append(
            #     ResPoolBlock(conv_dim=self.cv_dims[i]))
        self.pathsRGB = nn.ModuleList(self.pathsRGB)
        # self.pathsD = nn.ModuleList(self.pathsD)
        # self.pathsRP = nn.ModuleList(self.pathsRP)
        print('Creation of the MMF: OK.')

    def forward(self, x):
        assert (len(x) == self.nbr_paths)
        output = []
        for i in range(len(x)):
            rgb = self.pathsRGB[i](x[i])
            # d = self.pathsD[i](y[i])
            # sum_rgbd = rgb + d
            output.append(rgb)
        return output


class RCU(nn.Module):
    def __init__(self, conv_dim=256):
        super(RCU, self).__init__()
        self.conv_dim = conv_dim
        self.use_cuda = True
        self.use_batch_norm = False
        self.rcu = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(in_channels=self.conv_dim, out_channels=self.conv_dim, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=self.conv_dim, out_channels=self.conv_dim, kernel_size=3, stride=1, padding=1,
                      bias=False),
        ).cuda()

    def forward(self, x):
        out = self.rcu(x)
        residual = x
        output = out + residual
        return output


class RefineNetBlock(nn.Module):
    def __init__(self, conv_dim1=512, conv_dim2=256):
        super(RefineNetBlock, self).__init__()
        self.conv_dim1 = conv_dim1
        self.conv_dim2 = conv_dim2
        self.path1 = nn.Sequential(
            RCU(conv_dim=self.conv_dim1),
            RCU(conv_dim=self.conv_dim1),
            nn.Conv2d(self.conv_dim1, self.conv_dim2, 3, 1, 1, bias=False),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True))
        self.path2 = nn.Sequential(
            RCU(conv_dim=self.conv_dim2),
            RCU(conv_dim=self.conv_dim2),
            nn.Conv2d(in_channels=self.conv_dim2, out_channels=self.conv_dim2, kernel_size=3, stride=1, padding=1,
                      bias=False))
        self.single = nn.Sequential(
            RCU(conv_dim=self.conv_dim1),
            RCU(conv_dim=self.conv_dim1),
            ChainedResPoolBlock(conv_dim=self.conv_dim1),
            RCU(conv_dim=conv_dim1))
        self.chainedResPool = ChainedResPoolBlock(conv_dim=self.conv_dim2)
        self.finalRCU = RCU(conv_dim=conv_dim2)
        self = self.cuda()

    def forward(self, x, *y):
        if y == ():
            output = self.single(x)
        else:
            out_path1 = self.path1(x)
            out_path2 = self.path2(y[0])
            out_sum = out_path1 + out_path2
            output = self.chainedResPool(out_sum)
            output = self.finalRCU(output)
        return output


class RefineNet(nn.Module):
    def __init__(self, conv_dims=[512, 256, 256, 256], use_cuda=True, semantic_labels_nbr=2, use_batch_norm=False):
        super(RefineNet, self).__init__()
        self.conv_dims = conv_dims
        self.use_cuda = use_cuda
        self.use_batch_norm = use_batch_norm
        self.semantic_labels_nbr = semantic_labels_nbr
        self.nbr_paths = len(self.conv_dims)
        self.refineBlocks = []
        self.refineBlocks.append(
            RefineNetBlock(conv_dim1=512, conv_dim2=512))
        self.refineBlocks.append(
            RefineNetBlock(conv_dim1=512, conv_dim2=256))
        self.refineBlocks.append(
            RefineNetBlock(conv_dim1=256, conv_dim2=256))
        self.refineBlocks.append(
            RefineNetBlock(conv_dim1=256, conv_dim2=256))
        self.refineBlocks = nn.ModuleList(self.refineBlocks)
        self.finalRCUs = nn.Sequential(
            RCU(conv_dim=self.conv_dims[3]),
            RCU(conv_dim=self.conv_dims[3]),
            nn.Dropout2d(p=0.5),
            nn.Conv2d(in_channels=conv_dims[3], out_channels=1, kernel_size=1, stride=1,
                      padding=0))
        print('Creation of the RefineNet : OK.')

    def forward(self, x):
        assert (len(x) == self.nbr_paths)
        for i in range(len(x)):
            if i == 0:
                out = self.refineBlocks[i](x[i])
            else:
                out = self.refineBlocks[i](out, x[i])
        out = self.finalRCUs(out)
        return out


class ResPoolBlock(nn.Module):
    def __init__(self, conv_dim=256):
        super(ResPoolBlock, self).__init__()
        self.conv_dim = conv_dim
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=5, stride=1, padding=2).cuda()
        self.cv1 = nn.Conv2d(self.conv_dim, self.conv_dim, 3, stride=1, padding=1, bias=False).cuda()
        # if self.use_cuda:
        self = self.cuda()

    def forward(self, x):
        out_relu = self.relu1(x)
        out_pool1 = self.pool1(out_relu)
        out_cv1 = self.cv1(out_pool1)
        out = out_relu + out_cv1
        return out


class ChainedResPoolBlock(nn.Module):
    def __init__(self, conv_dim=256):
        super(ChainedResPoolBlock, self).__init__()
        self.use_cuda = True
        self.use_batch_norm = True
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=5, stride=1, padding=2)
        self.cv1 = nn.Conv2d(in_channels=conv_dim, out_channels=conv_dim, kernel_size=3, stride=1, padding=1,
                             bias=False)
        self.pool2 = nn.MaxPool2d(kernel_size=5, stride=1, padding=2)
        self.cv2 = nn.Conv2d(in_channels=conv_dim, out_channels=conv_dim, kernel_size=3, stride=1, padding=1,
                             bias=False)
        if self.use_cuda:
            self = self.cuda()

    def forward(self, x):
        out = self.relu1(x)
        out_pool1 = self.pool1(out)
        out_cv1 = self.cv1(out_pool1)
        outsum = out_cv1 + out
        out_pool2 = self.pool2(out_cv1)
        out_cv2 = self.cv2(out_pool2)
        outsum += out_cv2
        return outsum


def test_rdfnet():
    img_dim = 224
    conv_dim = 256
    use_cuda = True
    use_batch_norm = True
    batch_size = 1
    semantic_labels_nbr = 2
    rdfnet = RDFNet(use_cuda=use_cuda, semantic_labels_nbr=semantic_labels_nbr, use_batch_norm=use_batch_norm)
    print(rdfnet)
    # print(refinenet.refinenet.MultiResFusion)
    # rdfnet.getParameters()
    # raise
    inputs_rgb = torch.rand((batch_size, 3, img_dim, img_dim)).cuda()
    inputs_d = torch.rand((batch_size, 3, img_dim, img_dim)).cuda()

    t = time.time()
    outputs = rdfnet(inputs_rgb, inputs_d)
    elt = time.time() - t
    print('ELT : {} sec.'.format(elt))
    print(outputs.size())
    elt = 0.0
    nbr = 10
    for i in range(nbr):
        t = time.time()
        inputs_rgb = torch.rand((batch_size, 3, img_dim, img_dim)).cuda()
        inputs_d = torch.rand((batch_size, 3, img_dim, img_dim)).cuda()
        outputs = rdfnet(inputs_rgb, inputs_d)
        elt += (time.time() - t) / nbr

    print('MEAN ELT : {} sec.'.format(elt))


if __name__ == '__main__':
    test_rdfnet()
