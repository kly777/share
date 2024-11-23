import numpy as np


def sigmoid(x):
    """
    Sigmoid激活函数。
    参数:
    x -- 输入值

    返回:
    Sigmoid函数的输出。
    """
    return 1 / (1 + np.exp(-x))


def mse_loss(y_true: np.ndarray, y_pred: np.ndarray):
    # y_true 和 y_pred 是长度相同的 numpy 数组.
    return ((y_true - y_pred) ** 2).mean()


class Neuron:
    """
    神经元
    """

    def __init__(self, weights, bias):
        self.weights = weights
        self.bias = bias  # 偏移量

    def feedforward(self, inputs):
        # 权重乘以输入，与偏移量相加，然后通过激活函数
        total = np.dot(self.weights, inputs) + self.bias
        return sigmoid(total)


class NeuralNetwork:
    '''
    神经网络:
      - 2 个输入
      - 1 个隐藏层，2 个神经元 (h1, h2)
      - 1 个输出层，1 个神经元 (o1)
    所有神经元有同样的权重和偏移量:
      - w = [0, 1]
      - b = 0
    '''

    def __init__(self):
        weights = np.array([0, 1])
        bias = 0

        self.h1 = Neuron(weights, bias)
        self.h2 = Neuron(weights, bias)
        self.o1 = Neuron(weights, bias)

    def feedforward(self, x):
        out_h1 = self.h1.feedforward(x)
        out_h2 = self.h2.feedforward(x)

        # o1 的输入就是 h1 和 h2 的输出
        out_o1 = self.o1.feedforward(np.array([out_h1, out_h2]))

        return out_o1


network = NeuralNetwork()
x = np.array([2, 3])
print(network.feedforward(x))  # 0.7216325609518421
