from torch.autograd import Variable
import torch
import torch.nn.functional as F
import numpy as np
from torchvision.datasets import MNIST
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from collections import OrderedDict
from torch.nn.parameter import Parameter
import matplotlib.pyplot as plt
from MogrifierLSTM import MogrifierLSTMCell
from utils.dataset import NCDFDatasets




def gaussianNoise(img, mean, std):
    noise = Variable(img.data.new(img.size()).normal_(mean, std))
    return img + noise


class CustomConv2d(torch.nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1,
                 bias=False, padding_mode='zeros', weight=None):
        super().__init__(in_channels, out_channels, kernel_size, stride=stride,
                 padding=padding, dilation=dilation,
                 bias=bias, padding_mode=padding_mode)
        
        self.padding = kernel_size // 2
        if (weight is not None):
            self.weight = Parameter(weight.permute(1,0,2,3))
        
    def forward(self,input):
        return F.conv2d(input, self.weight, self.bias, self.stride,
                        self.padding, self.dilation)

class WeightSharingCnn(torch.nn.Module):
    def __init__(self):
        super(WeightSharingCnn, self).__init__()
        self.conv_layers = torch.nn.Sequential(OrderedDict([
            ('conv1', CustomConv2d(1, 6, kernel_size=5)),
            ('relu1', torch.nn.ReLU())
        ]))
        self.transp_conv_layers = torch.nn.Sequential(OrderedDict([
            ('transpConv1', CustomConv2d(6, 1, kernel_size=5, weight=self.conv_layers[0].weight)),
            ('transpRelu1', torch.nn.ReLU())
        ]))
    def forward(self, image):
        output = self.conv_layers(image)
        output = self.transp_conv_layers(output)
        return output


data_train = MNIST('./data/mnist',
                   download=True,
                   transform=transforms.Compose([
                       transforms.Resize((32, 32)),
                       transforms.ToTensor()]))
data_test = MNIST('./data/mnist',
                  train=False,
                  download=True,
                  transform=transforms.Compose([
                      transforms.Resize((32, 32)),
                      transforms.ToTensor()]))
data_train_loader = DataLoader(data_train, batch_size=256, shuffle=True, num_workers=8)
data_test_loader = DataLoader(data_test, batch_size=1024, num_workers=8)


class RMSELoss(torch.nn.Module):
    def __init__(self, eps=1e-6):
        super().__init__()
        self.mse = torch.nn.MSELoss()
        self.eps = eps
        
    def forward(self,yhat,y):
        loss = torch.sqrt(self.mse(yhat,y) + self.eps)
        return loss

net = WeightSharingCnn()
net.cuda()
criterion = RMSELoss()
optimizer = torch.optim.Adam(net.parameters(), lr=2e-3)
train_losses, val_losses = [], []
for epoch in range(1,16):
    train_loss = net.train()
    epoch_train_loss = 0.0
    for i, (images, labels) in enumerate(data_train_loader):
        images = images.cuda()
        noiseImages = gaussianNoise(images, 0, 0.1)
        optimizer.zero_grad()
        output = net(noiseImages)
        loss = criterion(output, images)
        loss.backward()
        optimizer.step()
        epoch_train_loss += loss.item()
    print(epoch_train_loss)
    avg_epoch_loss = epoch_train_loss/len(data_train_loader)
    train_losses.append(avg_epoch_loss)
    print('Train - Epoch %d, Batch: %d, Epoch Loss: %f' % (epoch, i, avg_epoch_loss))

    epoch_val_loss = 0.0
    val_loss = net.eval()
    for i, (images, labels) in enumerate(data_test_loader):
        images = images.cuda()
        noiseImages = gaussianNoise(images, 0, 0.1)
        output = net(noiseImages)
        loss = criterion(output, images)
        epoch_val_loss += loss.item()
    avg_loss = epoch_val_loss/len(data_test_loader)
    val_losses.append(avg_loss)
    print('Val Avg. Loss: %f' % (avg_loss))

epochs = np.arange(1,16)
plt.plot(train_losses)
plt.plot(val_losses)
plt.legend(['Train loss', 'Val loss'], loc='upper right')
plt.xlabel("Epochs")
plt.ylabel("RMSE Loss")
plt.show()