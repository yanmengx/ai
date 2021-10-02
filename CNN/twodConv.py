import numpy as np
from PIL import Image


def twoConv(f, w, method='zero'):
    f = np.array(f)
    # 卷积核转置
    # w = w.T
    # w = np.fliplr(np.flipud(w))
    fx, fy = f.shape
    x, y = w.shape
    x_ = int(x/2)
    y_ = int(y/2)
    padding = np.zeros((fx+x-1, fy+y-1))
    output = np.zeros((fx, fy))
    # 复制图片数据
    padding[x_: fx+x-1-x_, y_:fy+y-1-y_] = f

    if method == 'replicate':
        padding[0:x_, y_:fy+y-1-y_] = f[0, :]
        padding[fx+x-1-x_:, y_:fy+y-1-y_] = f[-1, :]
        # 边界填充
        for i in range(0, y_):
            padding[:, i] = padding[:, x_]
            padding[:, fy+y-1-1-i] = padding[:, fy+y-1-1-y_]
        # 卷积操作
        for i in range(0, fx):
            for j in range(0, fy):
                output[i, j] = np.sum(padding[i:i+x, j:j+y]*w)
    else:
        for i in range(0, fx):
            for j in range(0, fy):
                output[i, j] = np.sum(padding[i:i+x, j:j+y]*w)
    # 把小于min的数全部置换为min，大于max的数全部置换为max，在[min, max]之间的数则不变
    output = output.clip(0, 255)
    output = Image.fromarray(output)
    return output


if __name__ == '__main__':
    image = Image.open('cameraman.tif')
    w = np.array([
        [0, 1, 2],
        [2, 2, 0],
        [0, 1, 2]
    ])
    s = twoConv(image, w)
    s.show()
    s = twoConv(image, w, 'replicate')
    s.show()
