import numpy as np
from scipy import io


class Layer:
    def __init__(self, name):
        self.name = name
        self.out = None
        self.weights = None
        self.bias = None

    def forward(self, inputs):
        pass

    def backward(self, grad_out):
        pass

    def compute_penalty(self):
        return 0

    def zero_grad(self):
        pass

    def update(self, lr=1e-3):
        pass

    def parameters(self):
        return self.weights, self.bias


class Module:
    def __init__(self, lr=1e-3, r=1):
        self.layers = []
        self.r = r
        self.lr = lr

    def forward(self, x):
        for layer in self.layers:
            x = layer.forward(x)
        return x

    def backward(self, grad_out):
        for layer in reversed(self.layers):
            layer.zero_grad()
            grad_out = layer.backward(grad_out)
            layer.update(self.lr)
        return grad_out

    def parameters(self):
        parameters = None
        for layer in self.layers:
            if layer.weights is not None and layer.bias is not None:
                if parameters is None:
                    parameters = np.concatenate((np.expand_dims(layer.bias, axis=0), layer.weights), axis=0).ravel()
                else:
                    parameters = np.concatenate((parameters,
                                                 np.concatenate((np.expand_dims(layer.bias, axis=0), layer.weights),
                                                                axis=0).ravel()))
            elif layer.weights is not None and layer.bias is None:
                if parameters is None:
                    parameters = layer.weights.ravel()
                else:
                    parameters = np.concatenate((parameters, layer.weights.ravel()))
        return parameters

    def compute_penalty(self):
        penalty = 0.
        for layer in self.layers:
            penalty += layer.compute_penalty()
        return penalty


class Sigmoid(Layer):
    def __init__(self, name):
        super(Sigmoid, self).__init__(name)

    def forward(self, inputs):
        self.out = 1 / (1 + np.exp(-inputs))
        return self.out

    def backward(self, grad_out):
        dx = np.multiply(grad_out, np.multiply(self.out, 1 - self.out))
        return dx

    def compute_penalty(self):
        return 0.


class Relu(Layer):
    def __init__(self, name):
        super(Relu, self).__init__(name)
        self.mask = None

    def forward(self, inputs):
        self.mask = (inputs <= 0)
        self.out = np.maximum(inputs, 0)
        return self.out

    def backward(self, grad_out):
        dz = grad_out.copy()
        dz[self.mask] = 0
        return dz


class Softmax(Layer):
    def __init__(self, name):
        super(Softmax, self).__init__(name)

    def forward(self, inputs):
        input_exp = np.exp(inputs)
        self.out = input_exp / np.sum(input_exp)
        return self.out

    def backward(self, grad_out):
        return grad_out


class FC(Layer):
    def __init__(self, name, in_channels, out_channels, r=1):
        super(FC, self).__init__(name)
        self.weights = np.random.standard_normal((in_channels, out_channels))
        self.bias = np.zeros(out_channels)

        self.grad_w = np.zeros((in_channels, out_channels))
        self.grad_b = np.zeros(out_channels)
        self.r = r
        self.inputs = None

    def forward(self, inputs):
        self.inputs = inputs
        self.out = np.dot(inputs, self.weights) + self.bias
        return self.out

    def backward(self, grad_out):
        self.grad_w = np.dot(self.inputs.T, grad_out) + self.weights
        self.grad_b = np.sum(grad_out, axis=0)
        dx = np.dot(grad_out, self.weights.T)
        return dx

    def zero_grad(self):
        self.grad_b.fill(0)
        self.grad_w.fill(0)

    def update(self, lr=1e-3):
        self.weights -= lr * self.grad_w
        self.grad_b -= lr * self.grad_b

    def compute_penalty(self):
        penalty = np.sum(np.multiply(self.weights, self.weights)) * 2 / self.r
        return penalty


class MSELoss(Module):
    def __init__(self):
        super(MSELoss, self).__init__()


class Loss:
    def __init__(self, name='loss'):
        self.name = name

    @staticmethod
    def update(y_true, y_predict, penalty=0.):
        pos = np.multiply(-y_true, np.log(y_predict))
        neg = np.multiply((1 - y_true), np.log(1 - y_predict))
        loss = (np.sum(pos - neg) + penalty) / len(y_true)
        loss_diff = y_predict - y_true
        return loss, loss_diff


class Accuracy:
    def __init__(self, name):
        self.name = name

    @staticmethod
    def update(y_true, y_predict):
        y_label = np.argmax(y_predict, axis=1) + 1
        return len(y_label[y_true.ravel() == y_label.ravel()]) / len(y_true)


class Net(Module):
    def __init__(self, lr=1e-3, r=1):
        super(Net, self).__init__(lr=lr, r=r)
        self.layers = [FC('FC1', 400, 25),
                       Sigmoid('R1'),
                       FC('FC2', 25, 10),
                       Sigmoid('R2'),
                       ]


class Preprocess:
    def __init__(self, name):
        self.name = name

    @staticmethod
    def one_hot(y):
        array = np.zeros((len(y), y.max()))
        for i, y_i in enumerate(y):
            array[i, y_i - 1] = 1.
        return array


if __name__ == '__main__':
    data = io.loadmat('ex4data1.mat')
    X = data['X']
    y_raw = data['y']
    y = Preprocess.one_hot(y_raw)
    net = Net()
    for epoch in range(50000):
        y_predict = net.forward(X)
        acc = Accuracy.update(y_raw, y_predict)
        penalty = net.compute_penalty()
        loss, loss_diff = Loss.update(y, y_predict, penalty)
        net.backward(loss_diff)
        print('\repoch: {}, loss: {}, acc: {} '.format(epoch, loss, acc), end=' ')
