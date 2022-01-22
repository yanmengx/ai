import torch
from torch import nn
from torch.autograd import Variable
import matplotlib.pyplot as plt
import numpy as np

# 生成训练数据
n_data = torch.ones(100, 2)
x0 = torch.normal(2*n_data, 1)
y0 = torch.zeros(100, 1)
x1 = torch.normal(-2*n_data, 1)
y1 = torch.ones(100, 1)
# 把tensor连接在一起
x = torch.cat((x0, x1), 0).type(torch.FloatTensor)
y = torch.cat((y0, y1), 0).type(torch.FloatTensor)


class LogisticRegression(nn.Module):
    def __init__(self):
        # 初始化父类的写法也是固定的
        super(LogisticRegression, self).__init__()
        self.lr = nn.Linear(2, 1)
        self.sm = nn.Sigmoid()

    def forward(self, x):
        x = self.lr(x)
        x = self.sm(x)
        return x


model = LogisticRegression()
criterion = nn.BCELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=1e-3, momentum=0.9)

for epoch in range(10000):
    x_data = Variable(x)
    y_data = Variable(y)

    out = model(x_data)
    loss = criterion(out, y_data)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    mask = out.ge(0.5).float()
    correct = (mask == y_data).sum()
    acc = correct.item()/x_data.size(0)

    if (epoch+1) % 20 == 0:
        print('epoch {} loss is {:.4f} acc is {:.4f}'.format(epoch+1, loss.item(), acc))
# 获取线性层的权重和偏置
w0, w1 = model.lr.weight[0]
w0 = float(w0.item())
w1 = float(w1.item())
b = float(model.lr.bias.item())
plot_x = np.arange(-7, 7, 0.1)
plot_y = (-w0 * plot_x - b) / w1
plt.scatter(x.data.numpy()[:, 0], x.data.numpy()[:, 1], c=y.data.numpy(), s=100, lw=0, cmap='RdYlGn')
plt.plot(plot_x, plot_y)
plt.show()
