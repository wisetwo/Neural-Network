import math
from random import seed, random


# the objective function
# x^2 + y^2 < 1时为1，否则为0
def objective_fn(x, y):
    return 1.0 if x*x + y*y < 1 else 0.0

# >>>[(x,y) for x in range(0,3) for y in range(0,2)]
# [(0, 0), (0, 1), (1, 0), (1, 1), (2, 0), (2, 1)]

# samples - 生成
# print(type(sample_list))  # 输出：<class 'list'>

# # 确定长度
# >>> print(len(sample_list))  # 输出: 121, 11 * 11
# >>> print(len(sample_list[0])) # 输出: 2

# # 确定内部元素类型
# element_types = [type(element) for element in sample_list]
# [<class 'list'>, ...]
# >>> [x for x in range(11)]
# [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
# >>> [x for x in range(0, 11)]
# [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

sample_density = 10
sample_list = [
    [-2.0 + 4 * x/sample_density, -2.0 + 4 * y/sample_density]
    for x in range(sample_density+1)
    for y in range(sample_density+1)
]
dataset = [
    (x, y, objective_fn(x, y))
    for x, y in sample_list
]


# activation function 激活函数
def sigmoid(x):
    return 1 / (1 + math.exp(-x))

# 激活函数的导数，由数学公式可推导
def sigmoid_derivative(x):
    _output = sigmoid(x)
    return _output * (1 - _output)

# loss function
def square_loss(predict, target):
    return (predict-target)**2

# loss function的导数
def square_loss_derivative(predict, target):
    return 2 * (predict-target)


seed(0)
# 固定随机种子，固定random按顺序调用时的输出值
# >>> seed(0)
# >>> [random() for x in range(0, 5)]
# [0.8444218515250481, 0.7579544029403025, 0.420571580830845, 0.25891675029296335, 0.5112747213686085] # 固定输出这些值

# neural network
class Neuron:
    def __init__(self, input_num):
        # 随机初始化权重(-0.5 ~ 0.5)
        self.weights = [random()-0.5 for _ in range(input_num)]
        self.bias = 0.0

        # caches
        # z = wx + b后的一个实数
        self.z_cache = None
        # 一个结果
        self.inputs_cache = None

    def forward(self, inputs):
        # 校验参数长度，但是貌似这个前后参数没差异？
        # assert len(inputs) == len(inputs)
        self.inputs_cache = inputs

        # z = wx + b
        self.z_cache = sum([
            i * w
            for i, w in zip(inputs, self.weights)
        ]) + self.bias
        # sigmoid(wx + b)
        return sigmoid(self.z_cache)

    # 梯度清零
    def zero_grad(self):
        # 偏导数
        self.d_weights = [0.0 for w in self.weights]
        # 偏导数
        self.d_bias = 0.0

    def backward(self, d_a):
        d_loss_z = d_a * sigmoid_derivative(self.z_cache)
        self.d_bias += d_loss_z
        for i in range(len(self.inputs_cache)):
            self.d_weights[i] += d_loss_z * self.inputs_cache[i]
        return [d_loss_z * w for w in self.weights]

    def update_params(self, learning_rate):
        # 梯度下降法更新参数
        self.bias -= learning_rate * self.d_bias
        for i in range(len(self.weights)):
            self.weights[i] -= learning_rate * self.d_weights[i]

    def params(self):
        return self.weights + [self.bias]


class MyNet:
    def __init__(self, input_dimension_num, hidden_shapes):
        # 实际hidden_shapes = [4]
        # >>> [4] + [1]
        # [4, 1] 即隐藏+输出层的神经元个数(不含输入)
        layer_shapes = hidden_shapes + [1]
        # 实际input_dimension_num = 2，即输入维度数量
        # >>> [2] + [4]
        # [2, 4] 即输入+隐藏层的神经元个数
        input_shapes = [input_dimension_num] + hidden_shapes
        self.layers = [
            [
                Neuron(pre_layer_size)
                for _ in range(layer_size)
            ]
            # >>> [[x, y] for x, y in zip([4, 1], [2, 4])]
            # [[4, 2], [1, 4]]
            # >>> list(zip([4, 1], [2, 4]))
            # [(4, 2), (1, 4)] # 注意是tuple
            # >>> list(zip([1, 2, 3], ['a', 'b'], ['x', 'y', 'z']))
            # [(1, 'a', 'x'), (2, 'b', 'y')] # 长度为最短的长度
            for layer_size, pre_layer_size in zip(layer_shapes, input_shapes)
        ]
        # [
        #     [Neuron(2), Neuron(2), Neuron(2), Neuron(2)],
        #     [Neuron(4)]
        # ]
        # print(self.layers)

    def forward(self, inputs):
        print('myNetIns.forward', inputs)
        for layer in self.layers:
            inputs = [
                neuron.forward(inputs)
                for neuron in layer
            ]
        # return the output of the last neuron
        return inputs[0]

    def zero_grad(self):
        print('myNetIns.zero_grad')
        for layer in self.layers:
            for neuron in layer:
                neuron.zero_grad()

    def backward(self, d_loss):
        print('myNetIns.backward', d_loss)
        d_as = [d_loss]
        # >>> list(reversed([1,2,3]))
        # [3, 2, 1]
        for layer in reversed(self.layers):
            da_list = [
                neuron.backward(d_a)
                for neuron, d_a in zip(layer, d_as)
            ]
            # * 运算符用于解包（unpacking）序列或迭代器
            d_as = [sum(da) for da in zip(*da_list)]

    def update_params(self, learning_rate):
        print('myNetIns.update_params', learning_rate)
        for layer in self.layers:
            for neuron in layer:
                neuron.update_params(learning_rate)

    def params(self):
        return [[neuron.params() for neuron in layer]
                for layer in self.layers]


# build neural network
# 输入2个变量，一层隐藏层，含4个神经元
myNetIns = MyNet(2, [4])

# 未经训练的神经网络
# print(myNetIns.forward([0, 0]))
# 标准答案集
targets = [z for x, y, z in dataset]


# train one step
def one_step_in_train(learning_rate):
    myNetIns.zero_grad()

    loss = 0.0
    num_samples = len(dataset)
    for x, y, z in dataset:
        predict = myNetIns.forward([x, y])
        loss += square_loss(predict, z)

        myNetIns.backward(square_loss_derivative(predict, z) / num_samples)

    myNetIns.update_params(learning_rate)
    return loss / num_samples


def train_process(epoch, learning_rate):
    for i in range(epoch):
        loss = one_step_in_train(learning_rate)
        if i == 0 or (i + 1) % 100 == 0 or True:
            print(f"train {i + 1} loss -> {loss:.4f}")


def inference(x, y):
    return myNetIns.forward([x, y])

train_process(1, learning_rate=10)
# train_process(2000, learning_rate=10)

# 测试模型结果
# test_result = inference(1, 2)

# print('test_result = ', test_result)