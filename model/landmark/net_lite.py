import torch
import torch.nn as nn
from thop import profile

from utils.general import LOGGER, colorstr


def Conv(inp, oup, stride=1):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 3, stride, 1, bias=False),
        nn.BatchNorm2d(oup),
        nn.ReLU(inplace=True)
    )


def DPConv(inp, oup, kernel=1, stride=1, pad=0):
    return nn.Sequential(
        nn.Conv2d(inp, inp, kernel_size=kernel, stride=stride, padding=pad, groups=inp),
        nn.ReLU(inplace=True),
        nn.Conv2d(inp, oup, kernel_size=1)
    )


def DWConvblock(inp, oup, stride):
    return nn.Sequential(
        nn.Conv2d(inp, inp, 3, stride, 1, groups=inp, bias=False),
        nn.BatchNorm2d(inp),
        nn.ReLU(inplace=True),
        nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
        nn.BatchNorm2d(oup),
        nn.ReLU(inplace=True)
    )


class Facelite(nn.Module):
    def __init__(self):
        """
        :param cfg:  Network related settings.
        :param phase: train or test.
        """
        super(Facelite, self).__init__()
        self.num_classes = 2

        self.conv1 = Conv(3, 16, 2)
        self.conv2 = DWConvblock(16, 32, 1)
        self.conv3 = DWConvblock(32, 32, 2)
        self.conv4 = DWConvblock(32, 32, 1)
        self.conv5 = DWConvblock(32, 64, 2)
        self.conv6 = DWConvblock(64, 64, 1)
        self.conv7 = DWConvblock(64, 64, 1)
        self.conv8 = DWConvblock(64, 64, 1)

        self.conv9 = DWConvblock(64, 128, 2)
        self.conv10 = DWConvblock(128, 128, 1)
        self.conv11 = DWConvblock(128, 128, 1)

        self.conv12 = DWConvblock(128, 256, 2)
        self.conv13 = DWConvblock(256, 256, 1)

        self.conv14 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=64, kernel_size=1),
            nn.ReLU(inplace=True),
            DPConv(64, 256, kernel=3, stride=2, pad=1),
            nn.ReLU(inplace=True)
        )
        self.loc, self.conf, self.lmk = self.multibox(self.num_classes)

    def multibox(self, num_classes):
        loc_layers = []
        conf_layers = []
        lmk_layers = []
        loc_layers += [DPConv(64, 3 * 4, kernel=3, pad=1)]
        conf_layers += [DPConv(64, 3 * num_classes, kernel=3, pad=1)]
        lmk_layers += [DPConv(64, 3 * 10, kernel=3, pad=1)]

        loc_layers += [DPConv(128, 2 * 4, kernel=3, pad=1)]
        conf_layers += [DPConv(128, 2 * num_classes, kernel=3, pad=1)]
        lmk_layers += [DPConv(128, 2 * 10, kernel=3, pad=1)]

        loc_layers += [DPConv(256, 2 * 4, kernel=3, pad=1)]
        conf_layers += [DPConv(256, 2 * num_classes, kernel=3, pad=1)]
        lmk_layers += [DPConv(256, 2 * 10, kernel=3, pad=1)]

        loc_layers += [nn.Conv2d(256, 3 * 4, kernel_size=3, padding=1)]
        conf_layers += [nn.Conv2d(256, 3 * num_classes, kernel_size=3, padding=1)]
        lmk_layers += [nn.Conv2d(256, 3 * 10, kernel_size=3, padding=1)]
        return nn.Sequential(*loc_layers), nn.Sequential(*conf_layers), nn.Sequential(*lmk_layers)

    def forward(self, inputs):
        detections = list()
        loc = list()
        conf = list()
        lmk = list()

        x1 = self.conv1(inputs)
        x2 = self.conv2(x1)
        x3 = self.conv3(x2)
        x4 = self.conv4(x3)
        x5 = self.conv5(x4)
        x6 = self.conv6(x5)
        x7 = self.conv7(x6)
        x8 = self.conv8(x7)
        detections.append(x8)

        x9 = self.conv9(x8)
        x10 = self.conv10(x9)
        x11 = self.conv11(x10)
        detections.append(x11)

        x12 = self.conv12(x11)
        x13 = self.conv13(x12)
        detections.append(x13)

        x14 = self.conv14(x13)
        detections.append(x14)

        for (x, l, c, lm) in zip(detections, self.loc, self.conf, self.lmk):
            loc.append(l(x).permute(0, 2, 3, 1).contiguous())
            conf.append(c(x).permute(0, 2, 3, 1).contiguous())
            lmk.append(lm(x).permute(0, 2, 3, 1).contiguous())

        bbox_regressions = torch.cat([o.view(o.size(0), -1, 4) for o in loc], 1)
        classifications = torch.cat([o.view(o.size(0), -1, 2) for o in conf], 1)
        lmk_regressions = torch.cat([o.view(o.size(0), -1, 10) for o in lmk], 1)
        output = (bbox_regressions, classifications, lmk_regressions)

        return output

    def load(self, model):
        self.load_state_dict(torch.load(model, map_location=lambda storage, loc: storage))

    def save(self, model_path):
        torch.save(self.state_dict(), model_path)
        LOGGER.info('model save in ', model_path)


def create_net(model='Facelite', input=None, print_model=False):
    net = eval(model)()
    if input is None:
        input = torch.randn(1, 3, 240, 320)
    flops, params = profile(net, inputs=(input,))
    LOGGER.info(
        colorstr('create model: ') + "GFLOPS: {}, params: {}".format((flops / (1000 ** 3)), (params / (1000 ** 2))))
    if print_model:
        LOGGER.info(colorstr('model name: ') + f'{model} \n {net}')
    return net


if __name__ == '__main__':
    create_net(model='Facelite', input=torch.randn(1, 3, 600, 600), print_model=True)
