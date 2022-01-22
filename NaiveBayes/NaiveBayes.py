import numpy as np
import torch as tc
import os
import gzip


def load_data():
    data_folder = 'mnist/'
    files = ['train-labels-idx1-ubyte.gz', 'train-images-idx3-ubyte.gz',
             't10k-labels-idx1-ubyte.gz', 't10k-images-idx3-ubyte.gz']
    paths = []

    for fname in files:
        paths.append(os.path.join(data_folder, fname))

    with gzip.open(paths[0], 'rb') as lbpath:
        label_train = np.frombuffer(lbpath.read(), np.uint8, offset=8)

    with gzip.open(paths[1], 'rb') as imgpath:
        img_train = np.frombuffer(
            imgpath.read(), np.uint8, offset=16).reshape(len(label_train), -1)

    with gzip.open(paths[2], 'rb') as lbpath:
        label_test = np.frombuffer(lbpath.read(), np.uint8, offset=8)

    with gzip.open(paths[3], 'rb') as imgpath:
        img_test = np.frombuffer(
            imgpath.read(), np.uint8, offset=16).reshape(len(label_test), -1)

    #  图片像素归于0-1之间
    img_train, img_test = tc.tensor(img_train/256, dtype=tc.float64, device='cuda:0'), tc.tensor(
        img_test/256, dtype=tc.float64, device='cuda:0')
    # 图片数据二值化
    img_train, img_test = binarization(img_train), binarization(img_test)
    # 标签转化为torch.long格式
    label_train, label_test = tc.tensor(label_train, dtype=tc.long, device='cuda:0'), tc.tensor(
        label_test, dtype=tc.long, device='cuda:0')
    return img_train, label_train, img_test, label_test


def binarization(picture_set):
    picture_set_out = (picture_set != 0).to(tc.float64)
    return picture_set_out


def Calculate_prior_and_conditional(img_train, lable_train):
    prior_probability = tc.zeros(10, dtype=tc.float64, device='cuda:0')
    conditional_probability = tc.zeros((10, 784, 2), dtype=tc.float64, device='cuda:0')
    for i in range(10):
        a = img_train[lable_train == i, :]
        prior_probability[i] = (lable_train == i).nonzero().shape[0]/lable_train.shape[0]
        conditional_probability[i, :, 0] = tc.div(a.shape[0]-tc.sum(a, dim=0), a.shape[0]) + 1e-40
        conditional_probability[i, :, 1] = tc.div(tc.sum(a, dim=0), a.shape[0]) + 1e-40

    return prior_probability, conditional_probability


def Calculate_probability(test_picture_set, conditional_probability, prior_probability):
    probability = tc.zeros((10, test_picture_set.shape[0]), dtype=tc.float64, device='cuda:0')
    a = (test_picture_set == 0).to(tc.float64)
    b = (test_picture_set == 1).to(tc.float64)
    for i in range(10):
        class_conditional_probability = conditional_probability[i, :, :]
        a0_probability = tc.einsum('nl, l-> nl', [a, class_conditional_probability[:, 0]])
        a1_probability = tc.einsum('ab, b-> ab', [b, class_conditional_probability[:, 1]])
        a_probability = a0_probability + a1_probability
        probability[i, :] = tc.sum(tc.log(a_probability), dim=1) * prior_probability[i]
    return probability


def cul_accuracy(probability, lable_test):
    lable = tc.argmax(probability, dim=0)
    accuracy = ((lable == lable_test).nonzero().squeeze().shape[0] / lable.shape[0])
    return accuracy


if __name__ == '__main__':
    print('读取数据集')
    img_train, lable_train, img_test, lable_test = load_data()
    print('通过训练集计算条件概率及先验概率')
    prior_probability, conditional_probability = Calculate_prior_and_conditional(img_train, lable_train)
    print('预测标签')
    probability = Calculate_probability(img_test, conditional_probability, prior_probability)
    print('计算分类准确率')
    accuracy = cul_accuracy(probability, lable_test)
    print('测试集分类准确率为：', accuracy)