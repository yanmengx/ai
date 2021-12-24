import torch
import torch.nn as nn
import torchvision
import torch.utils.data as Data
import math
from torch.autograd import Variable
from torchvision.transforms import Compose, ToTensor, Resize
import gc

# 清理内存
gc.collect()


# 对输入图像进行处理，转换为(224，224)，因为resnet18要求输入为(224，224)，并转化为tensor
def input_transform():
    return Compose([
        Resize(224),
        ToTensor()
    ])


# mnist数据
train_data = torchvision.datasets.MNIST(
    root='./mnist/',
    train=True,
    transform=input_transform(),
    download=True
)

test_data = torchvision.datasets.MNIST(
    root='./mnist/',
    train=False,
    transform=input_transform(),
    download=True
)

BATCH_SIZE = 128
# 批处理
loader = Data.DataLoader(dataset=test_data,
                         batch_size=BATCH_SIZE,
                         shuffle=True,
                         num_workers=0)


def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)


class ResnetBlock(nn.Module):
    # expansion是残差结构中输出维度是输入维度的多少倍，BasicBlock没有升维，所以expansion = 1
    # 残差结构是在求和之后才经过ReLU层
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(ResnetBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x
        # 这里的x是输入的数据，即用来训练的数据
        # 这里是用变量存储函数，直接调用变量
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class ResNet(nn.Module):
    def __init__(self, block, layers, num_classes=10):
        self.inplanes = 64
        super(ResNet, self).__init__()
        # mnist的通道数为1
        self.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AvgPool2d(7, stride=1)
        self.fc = nn.Linear(512 * block.expansion, num_classes)
        # 初始化模型的权重，一种常用的方法，记住直接用
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2./n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion)
            )

        layers = []
        # pytorch好像可以用变量存储函数名，然后直接调用变量后面加参数列表就能运行了，前面的down萨满评论也是这样
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x


def resnet18(pretrained=False, **kwargs):
    # 架构之间的不同在于basicblock和bottlenek之间的不同以及block的输入参数的不同。因为ResNet一般有4个stack，
    # 每一个stack里面都是block的堆叠，所以[3, 4, 6, 3]就是每一个stack里面堆叠block的个数，故而造就了不同深度的ResNet。
    model = ResNet(ResnetBlock, [2, 2, 2, 2], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet18']))
    return model


net = resnet18()

optimizer = torch.optim.Adam(net.parameters(), lr=0.01)
loss_func = torch.nn.CrossEntropyLoss()
for epoch in range(3):
    for step, (batch_x, batch_y) in enumerate(loader):
        b_x = Variable(batch_x)
        b_y = Variable(batch_y)

        predict = net(b_x)
        loss = loss_func(predict, b_y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if step % 5 == 0:
            print('epoch:{}, step:{}, loss:{}'.format(epoch, step, loss))
