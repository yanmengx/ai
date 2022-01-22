import torch
from torch.autograd import Variable
import numpy
import random
import matplotlib.pyplot as plt
from torch import nn

x = torch.unsqueeze(torch.linspace(-1, 1, 100), dim=1)
y = 3*x + 10 + torch.rand(x.size())


# 每一个pytorch模型都要新建一个类，继承nn.Module，然后写构造函数，即__init__,
# 在__init__里面首先要初始化父类，在把模型里所有要用到的层定义好，用变量代表定义好
# 的层，在后面的函数里直接调用变量名后面更括号放参数就行了。然后要定义forward函数，
# 即前向传播函数。
class LinearRegression(nn.Module):
    def __init__(self):
        super(LinearRegression, self).__init__()
        self.linear = nn.Linear(1, 1)

    def forward(self, x):
        out = self.linear(x)
        return out


model = LinearRegression()
# 训练代码首先要定义优化器，可以是adam也可以是sgd，然后定义损失函数，也是在后面直接用变量名调用
optimizer = torch.optim.SGD(model.parameters(), lr=1e-2)
criterion = nn.MSELoss()

num_epoches = 1000
for epoch in range(num_epoches):
    inputs = Variable(x)
    target = Variable(y)

    out = model(inputs)
    loss = criterion(out, target)
    # 每次训练都要写，可以看成固定的必需组合，梯度为0，误差反向传播，更新参数
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    # loss是个tensor，pytorch不能用loss.data[0]，要用loss.item()
    if (epoch+1) % 20 == 0:
        print('Epoch[{}/{}], loss:{:.6f}'.format(epoch+1, num_epoches, loss.item()))

model.eval()
predict = model(Variable(x))
predict = predict.data.numpy()
plt.plot(x.numpy(), y.numpy(), 'ro', label='original data')
plt.plot(x.numpy(), predict, label='fitting line')
plt.show()
