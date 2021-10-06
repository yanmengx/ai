import numpy as np
from PIL import Image
from scipy import signal


def twoConv(f, w, method='zero'):
    f = np.array(f)
    # 反转卷积核
    w = np.fliplr(np.flipud(w))
    fx, fy = f.shape
    x, y = w.shape
    x_ = int(x/2)
    y_ = int(y/2)
    # 初始化填补后的图像，填补后的长宽为原图片的长宽加上卷积核的长宽减1，默认用0填补
    padding = np.zeros((fx+x-1, fy+y-1))
    output = np.zeros((fx, fy))
    # 将原图拷贝到填补后图像的对应位置
    padding[x_: fx+x-1-x_, y_:fy+y-1-y_] = f
    # 用边界像素填补
    if method == 'replicate':
        padding[0:x_, y_:fy+y-1-y_] = f[0, :]
        padding[fx+x-1-x_:, y_:fy+y-1-y_] = f[-1, :]

        for i in range(0, y_):
            padding[:, i] = padding[:, x_]
            padding[:, fy+y-1-1-i] = padding[:, fy+y-1-1-y_]
        # 计算卷积
        for i in range(0, fx):
            for j in range(0, fy):
                output[i, j] = np.sum(padding[i:i+x, j:j+y]*w)
    else:
        # 计算卷积
        for i in range(0, fx):
            for j in range(0, fy):
                output[i, j] = np.sum(padding[i:i+x, j:j+y]*w)
    # 处理值大于255，小于0的像素点
    output = output.clip(0, 255)
    # 转化回图片格式
    output = Image.fromarray(output)
    return output


if __name__ == '__main__':
    image = Image.open('cameraman.tif')
    w = np.array([
        [0, 1, 2],
        [2, 2, 0],
        [0, 1, 2]
    ])
    # 0填补计算卷积
    s = twoConv(image, w)
    s.show()
    # 边界填补计算卷积
    s = twoConv(image, w, 'replicate')
    s.show()
    # 与python包里的卷积函数比较
    s = signal.convolve2d(image, w, boundary='symm', mode='same')
    s = Image.fromarray(s)
    s.show()
