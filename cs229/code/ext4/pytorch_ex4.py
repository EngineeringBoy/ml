import torch
from scipy import io
import matplotlib.pyplot as plt
import torch.nn.functional as F


class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = torch.nn.Linear(400, 25)
        self.s1 = torch.nn.Sigmoid()
        self.fc2 = torch.nn.Linear(25, 10)
        self.s2 = torch.nn.Sigmoid()

    def forward(self, x):
        x = self.fc1(x)
        x = self.s1(x)
        x = self.fc2(x)
        x = self.s2(x)
        return x


if __name__ == '__main__':
    # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    net = Net()
    # net = net.to(device)
    loss_fn = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(net.parameters())
    data1 = io.loadmat('ex4data1.mat')
    X = torch.from_numpy(data1['X']).to(torch.float32)
    y = torch.from_numpy(data1['y']).to(torch.int64)
    y_one_hot = F.one_hot((y - 1).ravel(), num_classes=10).to(torch.float32)

    for iteration in range(10000):
        y_pred = net(X)
        loss = loss_fn(y_pred, y_one_hot)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        y_label = torch.argmax(y_pred, dim=1) + 1
        acc = len(y[y_label.ravel() == y.ravel()]) / len(y)
        print('\repoch: {}, loss: {}, acc: {}'.format(iteration, loss.detach().numpy(), acc), end=' ')

        # plot
        if iteration % 1000 == 0:
            fc1_weight_vis = net.fc1.weight.detach().numpy().reshape(100, 100)
            plt.figure(0)
            plt.imshow(fc1_weight_vis, cmap='gray')
            plt.show()
