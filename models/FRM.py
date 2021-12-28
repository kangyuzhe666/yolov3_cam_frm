import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter

class PCRC(nn.Module):
    def __init__(self):
        super().__init__()
        self.C1 = nn.Conv2d(3, 512 * 3, kernel_size=1)
        self.R1 = nn.Upsample(None, 2, 'nearest')  # 上采样扩充2倍采用邻近扩充
        self.mcrc = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(512 * 3, 512 * 3, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512 * 3, 512 * 3, kernel_size=3, padding=1),
        )
        self.acrc = nn.Sequential(
            nn.AvgPool2d(kernel_size=2, stride=2),
            nn.Conv2d(512 * 3, 512 * 3, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512 * 3, 512 * 3, kernel_size=3, padding=1),
        )

    def forward(self, x):
        x1 = self.C1(x)
        x2 = self.mcrc(x1)
        x3 = self.acrc(x1)
        return self.R1(x2) + self.R1(x3)

# FRM模块实现输入x为tensor列表形式
class FRM(nn.Module):
    def __init__(self, nc=80, anchors=(), ch=(), inplace=True):
        super().__init__()
        self.R1 = nn.Upsample(None, 2, 'nearest')  #上采样扩充2倍采用邻近扩充
        self.R3 = nn.MaxPool2d(kernel_size=2, stride=2)   #下采样使用最大池化
        self.C1 = nn.Conv2d(1024 + 512 + 256, 3, kernel_size=1)

        self.C2 = nn.Conv2d(512, 1024, kernel_size=1, stride=1)
        self.C3 = nn.Conv2d(512, 256, kernel_size=1, stride=1)
        self.C4 = nn.Conv2d(1, 256, kernel_size=1, stride=1)
        self.C5 = nn.Conv2d(1, 512, kernel_size=1, stride=1)
        self.C6 = nn.Conv2d(1, 1024, kernel_size=1, stride=1)
        self.pcrc = PCRC()

    def forward(self, x):

        x0 = self.R1(x[0])
        x2 = self.R3(x[2])
        x1 = self.C1(torch.cat((x0, x[1], x2), 1))
        Conv_1_1 = torch.split(torch.softmax(x1, dim=0), 1, 1)       #第一维度1为步长进行分割
        Conv_1_2 = torch.split(self.pcrc(x1), 512, 1)                #第一维度512为步长进行分割
        y0 = (x0 * self.C6(Conv_1_1[0])) + (self.C2(Conv_1_2[0]) * x0)
        y1 = (x[1] * self.C5(Conv_1_1[1])) + (Conv_1_2[1] * x[1])
        y2 = (x2 * self.C4(Conv_1_1[2])) + (self.C3(Conv_1_2[2]) * x2)

        y0 = self.R3(y0)
        y2 = self.R1(y2)

        return [y0, y1, y2]


if __name__ == '__main__':
    model = FRM()
    model.cuda()

    img1 = torch.rand(1, 1024, 20, 20)  # 假设输入1张1024*20*20的特征图
    img2 = torch.rand(1, 512, 40, 40)  # 假设输入1张512*40*40的特征图
    img3 = torch.rand(1, 256, 80, 80)  # 假设输入1张256*80*80的特征图
    img1 = img1.cuda()
    img2 = img2.cuda()
    img3 = img3.cuda()

    with SummaryWriter(comment='FRM') as w:
        w.add_graph(model, ([img1, img2, img3],))