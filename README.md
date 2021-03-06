# yolov3_cam_frm
 yolov3基础上添加cam与frm模块参考CONTEXT AUGMENTATION AND FEATURE REFINEMENT
NETWORK FOR TINY OBJECT DETECTION

模型总体结构如下图

![1](images/1.jpg)

cam模块，融合方式采用（c）

![1](images/2.jpg)

根据论文数据c方式效果最好

![1](images/3.jpg)

cam关键代码实现

```python
class CAM(nn.Module):
    def __init__(self, c1=1024, c2=1024, k=3, s=1):
        super().__init__()
        self.c1 = c1
        self.c2 = c2
        self.dilate_1 = nn.Sequential(
            nn.Conv2d(self.c1, self.c1, kernel_size=k, stride=s, padding=1, dilation=1),
            nn.Conv2d(self.c1, c2 // 3, kernel_size=1),
        )
        self.dilate_3 = nn.Sequential(
            nn.Conv2d(self.c1, self.c1, kernel_size=k, stride=s, padding=3, dilation=3),
            nn.Conv2d(self.c1, c2 // 3, kernel_size=1),
        )
        self.dilate_5 = nn.Sequential(
            nn.Conv2d(self.c1, self.c1, kernel_size=k, stride=s, padding=5, dilation=5),
            nn.Conv2d(self.c1, (c2 // 3 + c2 % 3), kernel_size=1),
        )

    def forward(self, x):
        x1 = self.dilate_1(x)
        x2 = self.dilate_3(x)
        x3 = self.dilate_5(x)
        return torch.cat((x1, x2, x3), 1)

```

frm模块如下图，按照论文插图实现一些尺寸对应不上论文描述也比较模糊，故在尺寸不一致时使用了1x1卷积，导致该模块参数量过大

![1](images/4.jpg)

frm模块集成才检测头里面这样对代码修改量最少

![1](images/5.jpg)

frm关键代码

```python
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
```

最终在VOC2007 val的效果，在原有预训练模型的基础上训练300epoch。

![1](images/6.jpg)

总结：对论文的初步实现，相关超参数没有进行修改也没有进行大规模训练实验，仅供参考

yolov3代码参考：https://github.com/ultralytics/yolov3

训练方法参考：https://github.com/ultralytics/yolov3/wiki/Tips-for-Best-Training-Results