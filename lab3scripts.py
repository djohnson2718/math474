import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import matplotlib.pyplot as plt
from torchvision import transforms, utils, datasets
from tqdm import tqdm
from torch.nn.parameter import Parameter
import pdb

assert torch.cuda.is_available(), "You need to request a GPU from Runtime > Change Runtime"

class LinearNetwork(nn.Module):
  def __init__(self,dataset, num_inner_neurons):
    super(LinearNetwork, self).__init__()
    x,y = dataset[0]
    c,h,w = x.size()
    out_dim = 10
    self.num_inner_neurons = num_inner_neurons

    self.net = nn.Sequential(nn.Linear(c*h*w, self.num_inner_neurons), nn.ReLU(), nn.Linear(self.num_inner_neurons, out_dim))

  def forward(self,x):
    n,c,h,w = x.size()
    flattened = x.view(n,c*h*w)
    return self.net(flattened)


class Conv2d(nn.Module):
  def __init__(self,in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True, padding_mode='zeros', device=None, dtype=None, init="xe"):
    self.__dict__.update(locals())
    super(Conv2d, self).__init__()
    self.weight = Parameter(torch.Tensor(out_channels, in_channels, kernel_size[0], kernel_size[1]))
    self.bias = Parameter(torch.Tensor(out_channels))
    if init=="uniform":
      self.weight.data.uniform_(-1,1)
      self.bias.data[:] = 0
    elif init=="orth":
      #print(kernel_size,in_channels,out_channels)
      X = np.random.random((kernel_size[0]*kernel_size[1]*in_channels, out_channels))
      U, _, Vt = np.linalg.svd(X)
      #print(X.shape, U.shape, Vt.shape)
      W = U[0:out_channels,:].reshape(out_channels, in_channels, kernel_size[0], kernel_size[1])
      self.weight.data[:] = torch.tensor(W)
      self.bias.data[:] = 0
      assert np.abs( (W[0,:]*(W[0,:])).sum()-1) < 1e-6, (W[0,:].dot(W[0,:])).sum()-1
      assert np.abs( (W[1,:]*(W[0,:])).sum()) < 1e-6
    elif init=="xe":
      var = 2/in_channels**2
      std = np.sqrt(var)     
      self.weight.data.normal_(0,std)
      self.bias.data[:] = 0
    else:
      raise ValueError("Unknown init type")


    


  def forward(self,x):
    return F.conv2d(x,self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups)

  def extra_repr(self):
    return "?"


class ConvNetwork(nn.Module):
  def __init__(self, dataset,init="xe"):
    super(ConvNetwork,self).__init__()
    x,y = dataset[0]
    c,h,w = x.size()
    out_dim = 10

    self.net = nn.Sequential( Conv2d(c,9,(3,3), padding=(1,1),init=init),
                             torch.nn.ReLU(),
                             Conv2d(9,out_dim,(28,28), padding=(0,0),init=init) )

  def forward(self,x):
    return self.net(x).squeeze(2).squeeze(2)


class ConvNetwork2(nn.Module):
  def __init__(self, dataset,init="xe"):
    super(ConvNetwork2,self).__init__()
    x,y = dataset[0]
    c,h,w = x.size()
    out_dim = 10

    self.net = nn.Sequential( Conv2d(c,9,(3,3), padding=(1,1),init=init),
                             torch.nn.ReLU(),
                             Conv2d(9,20,(3,3), padding=(1,1),init=init),
                             torch.nn.ReLU(),
                             Conv2d(20,out_dim,(28,28), padding=(0,0),init=init))
  def forward(self,x):
    return self.net(x).squeeze(2).squeeze(2)

class ConvNetwork3(nn.Module):
  def __init__(self, dataset,init="xe"):
    super(ConvNetwork3,self).__init__()
    x,y = dataset[0]
    c,h,w = x.size()
    out_dim = 10

    self.net = nn.Sequential( Conv2d(c,9,(3,3), padding=(1,1),init=init),
                             torch.nn.ReLU(),
                             Conv2d(9,10,(3,3), padding=(1,1),init=init),
                             torch.nn.MaxPool2d(2),
                             Conv2d(10,20,(3,3), padding=(1,1),init=init),
                             torch.nn.ReLU(),
                             Conv2d(20,out_dim,(14,14), padding=(0,0),init=init)
                              )

  def forward(self,x):
    return self.net(x).squeeze(2).squeeze(2)
  

    
class FMPDataset(Dataset):
  def __init__(self, root, train=True):
    self.data = datasets.FashionMNIST(root, train=train, transform= transforms.ToTensor(), download = True)

  def __getitem__(self,i):
    return self.data[i]

  def __len__(self):
    return len(self.data)

class CrossEntropyLoss(nn.Module):
  def __init__(self):
    super(CrossEntropyLoss, self).__init__()

  def forward(self, y_hat, y_truth):
    yhe = y_hat.exp()
    score = yhe/yhe.sum(dim=1, keepdim=True)
    score_of_correct = score[range(y_truth.size(0)), y_truth]
    return -torch.log(score_of_correct).mean()

train_dataset = FMPDataset("/tmp/fashionmnist", train=True)
val_dataset = FMPDataset("/tmp/fashionmnist", train=False)

bs = 42
#model = LinearNetwork(train_dataset,1000)
#model = model.cuda()
train_loader = DataLoader(train_dataset, batch_size = bs, pin_memory = True)
validation_loader = DataLoader(val_dataset, batch_size = bs)


objective = torch.nn.CrossEntropyLoss()
#myobjective = CrossEntropyLoss()


def do_it(model):
  model = model.cuda()
  optimizer= optim.SGD(model.parameters(), lr = 1e-4)
  losses = []
  validations = []

  num_epochs = 30 
  loop =tqdm(total=len(train_loader)*num_epochs, position = 0)

  for epoch in range(num_epochs):
    for  batch, (x,y_truth) in enumerate(train_loader):
      x,y_truth = x.cuda(non_blocking=True), y_truth.cuda(non_blocking=True)

      optimizer.zero_grad()
      y_hat =  model(x)
      loss = objective(y_hat, y_truth)

      #assert loss - myobjective(y_hat, y_truth) < 1e-6, f"myloss {myobjective(y_hat, y_truth)} != loss {loss}"

      loss.backward()

      losses.append(loss.item())
      accuracy = 0

      #loop.set_description("batch:{} loss:{:.4f} val_loss:?".format(batch, loss.item()))
      loop.update(1)

      optimizer.step()

      if batch %100 == 0:
        val = np.mean( [objective(model(x.cuda()), y.cuda()).item() for x,y in validation_loader])
        validations.append((len(losses),val))


      loop.set_description("batch:{} loss:{:.4f} val_loss:{:.4f}".format(batch, loss.item(), validations[-1][1]))


  loop.close()
  plt.plot(losses,label="train")
  plt.plot(*zip(*validations),label="val")
  return losses, validations