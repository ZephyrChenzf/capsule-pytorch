from torchvision import datasets,transforms
import torch
from torch.autograd.variable import Variable
from torch.utils.data import DataLoader
import math
from torch.optim import lr_scheduler
import torch.nn.functional as F
from torch import nn,optim
import os
os.environ['CUDA_VISIBLE_DEVICES']='0'


def squash(x):#b,32*6*6,8
    lengths2 = x.pow(2).sum(dim=2)#b,32*6*6
    lengths = lengths2.sqrt()#b,32*6*6
    x = x * (lengths2 / (1 + lengths2) / lengths).view(x.size(0), x.size(1), 1)
    return x


class AgreementRouting(nn.Module):
    def __init__(self, input_caps, output_caps, n_iterations):
        super(AgreementRouting, self).__init__()
        self.n_iterations = n_iterations#3
        self.b = nn.Parameter(torch.zeros((input_caps, output_caps)))#32*6*6,10

    def forward(self, u_predict):#b,32*6*6,10,16
        batch_size, input_caps, output_caps, output_dim = u_predict.size()

        c = F.softmax(self.b,dim=1)
        s = (c.unsqueeze(2) * u_predict).sum(dim=1)#b,10,16
        v = squash(s)#b,10,16

        if self.n_iterations > 0:
            b_batch = self.b.expand((batch_size, input_caps, output_caps))#b,32*6*6,10
            for r in range(self.n_iterations):
                v = v.unsqueeze(1)#b,1,10,16
                b_batch = b_batch + (u_predict * v).sum(-1)#b,32*6*6,10

                c = F.softmax(b_batch.view(-1, output_caps),dim=1).view(-1, input_caps, output_caps, 1)#b,32*6*6,10,1
                s = (c * u_predict).sum(dim=1)#b,10,16
                v = squash(s)#b,10,16

        return v#b,10,16


class CapsLayer(nn.Module):
    def __init__(self, input_caps, input_dim, output_caps, output_dim, routing_module):
        super(CapsLayer, self).__init__()
        self.input_dim = input_dim#8
        self.input_caps = input_caps#32*6*6
        self.output_dim = output_dim#16
        self.output_caps = output_caps#10
        self.weights = nn.Parameter(torch.Tensor(input_caps, input_dim, output_caps * output_dim))#32*6*6,8,10*16
        self.routing_module = routing_module
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.input_caps)
        self.weights.data.uniform_(-stdv, stdv)

    def forward(self, caps_output):
        caps_output = caps_output.unsqueeze(2)#b,32*6*6,1,8
        u_predict = caps_output.matmul(self.weights)#b,32*6*6,1,10*16
        u_predict = u_predict.view(u_predict.size(0), self.input_caps, self.output_caps, self.output_dim)#b,32*6*6,10,16
        v = self.routing_module(u_predict)
        return v


class PrimaryCapsLayer(nn.Module):
    def __init__(self, input_channels, output_caps, output_dim, kernel_size, stride):
        super(PrimaryCapsLayer, self).__init__()
        self.conv = nn.Conv2d(input_channels, output_caps * output_dim, kernel_size=kernel_size, stride=stride)
        self.input_channels = input_channels
        self.output_caps = output_caps
        self.output_dim = output_dim

    def forward(self, input):
        out = self.conv(input)#b,32*8,6,6
        N, C, H, W = out.size()
        out = out.view(N, self.output_caps, self.output_dim, H, W)#b,32,8,6,6

        # will output N x OUT_CAPS x OUT_DIM
        out = out.permute(0, 1, 3, 4, 2).contiguous()#b,32,6,6,8
        out = out.view(out.size(0), -1, out.size(4))#b,32*6*6,8
        out = squash(out)#b,32*6*6,8
        return out


class CapsNet(nn.Module):
    def __init__(self, routing_iterations, n_classes=10):
        super(CapsNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 256, kernel_size=9, stride=1)
        self.primaryCaps = PrimaryCapsLayer(256, 32, 8, kernel_size=9, stride=2)  # outputs 6*6
        self.num_primaryCaps = 32 * 6 * 6
        routing_module = AgreementRouting(self.num_primaryCaps, n_classes, routing_iterations)
        self.digitCaps = CapsLayer(self.num_primaryCaps, 8, n_classes, 16, routing_module)

    def forward(self, input):#b,1,28,28
        x = self.conv1(input)#b,256，20，20
        x = F.relu(x)
        x = self.primaryCaps(x)#b,32*6*6,8
        x = self.digitCaps(x)#b,10,16
        probs = x.pow(2).sum(dim=2).sqrt()#b,10
        return x, probs


class ReconstructionNet(nn.Module):
    def __init__(self, n_dim=16, n_classes=10):
        super(ReconstructionNet, self).__init__()
        self.fc1 = nn.Linear(n_dim * n_classes, 512)
        self.fc2 = nn.Linear(512, 1024)
        self.fc3 = nn.Linear(1024, 784)
        self.n_dim = n_dim
        self.n_classes = n_classes

    def forward(self, x, target):
        mask = Variable(torch.zeros((x.size()[0], self.n_classes)), requires_grad=False)#b,10
        if next(self.parameters()).is_cuda:
            mask = mask.cuda()
        mask.scatter_(1, target.view(-1, 1), 1.)
        mask = mask.unsqueeze(2)#b,10,1
        x = x * mask
        x = x.view(-1, self.n_dim * self.n_classes)#b,16*10
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.sigmoid(self.fc3(x))#b,784
        return x


class CapsNetWithReconstruction(nn.Module):
    def __init__(self, capsnet, reconstruction_net):
        super(CapsNetWithReconstruction, self).__init__()
        self.capsnet = capsnet
        self.reconstruction_net = reconstruction_net

    def forward(self, x, target):
        x, probs = self.capsnet(x)#b,10,16  #b,10
        reconstruction = self.reconstruction_net(x, target)
        return reconstruction, probs


class MarginLoss(nn.Module):
    def __init__(self, m_pos, m_neg, lambda_):
        super(MarginLoss, self).__init__()
        self.m_pos = m_pos#0.9
        self.m_neg = m_neg#0.1
        self.lambda_ = lambda_#0.5

    def forward(self, lengths, targets, size_average=True):#b,10   #b
        t = torch.zeros(lengths.size()).long()#b,10
        if targets.is_cuda:
            t = t.cuda()
        t = t.scatter_(1, targets.data.view(-1, 1),1)#b,10
        targets = Variable(t)#b,10
        losses = targets.float() * F.relu(self.m_pos - lengths).pow(2) + \
                 self.lambda_ * (1. - targets.float()) * F.relu(lengths - self.m_neg).pow(2)
        return losses.mean() if size_average else losses.sum()
if __name__=='__main__':
    batch_size=64
    epochs=20
    learning_rate=1e-3
    routing_iterations=3
    with_reconstruction=True
    use_cuda=torch.cuda.is_available()


    if use_cuda==True:
        torch.cuda.manual_seed(20180601)
    else:
        torch.manual_seed(20180601)

    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST('./data', train=True, download=True,
                       transform=transforms.Compose([
                           transforms.Pad(2), transforms.RandomCrop(28),
                           transforms.ToTensor()
                       ])),
        batch_size=batch_size, shuffle=True)

    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST('./data', train=False, transform=transforms.Compose([
            transforms.ToTensor()
        ])),
        batch_size=64, shuffle=False)

    model = CapsNet(routing_iterations)

    if with_reconstruction:
        reconstruction_model = ReconstructionNet(16, 10)
        reconstruction_alpha = 0.0005
        model = CapsNetWithReconstruction(model, reconstruction_model)
    if use_cuda:
        model.cuda()

    optimizer=optim.Adam(model.parameters(),lr=learning_rate)
    scheduler=lr_scheduler.ReduceLROnPlateau(optimizer,verbose=True,patience=10,min_lr=1e-6)
    loss_fn = MarginLoss(0.9, 0.1, 0.5)

    for epoch in range(1,epochs+1):
        best_acc=0
        best_model=None
        model.train()
        for i, (data, target) in enumerate(train_loader):
            if use_cuda:
                data, target = data.cuda(), target.cuda()#b,1,28,28  #b
            data, target = Variable(data), Variable(target, requires_grad=False)
            #forward
            if with_reconstruction:
                output, probs = model(data, target)
                reconstruction_loss = F.mse_loss(output, data.view(-1, 784))
                margin_loss = loss_fn(probs, target)
                loss = reconstruction_alpha * reconstruction_loss + margin_loss
            else:
                output, probs = model(data)#b,10,16  #b,10
                loss = loss_fn(probs, target)
            #backword
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if i % 100 == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, i * len(data), len(train_loader.dataset),
                           100. * i / len(train_loader), loss.item()))

        model.eval()
        test_loss = 0
        correct = 0
        for data, target in test_loader:
            if use_cuda:
                data, target = data.cuda(), target.cuda()
            with torch.no_grad():
                data=Variable(data)
                target=Variable(target)

            # if with_reconstruction:
            #     output, probs = model(data, target)
            #     reconstruction_loss = F.mse_loss(output, data.view(-1, 784)).item()
            #     test_loss += loss_fn(probs, target).item()
            #     test_loss += reconstruction_alpha * reconstruction_loss
            # else:
            #     output, probs = model(data)
            #     test_loss += loss_fn(probs, target).item()

            output, probs = model.capsnet(data)
            test_loss += loss_fn(probs, target).item()

            pred = probs.data.max(1, keepdim=True)[1]  # get the index of the max probability
            correct += pred.eq(target.data.view_as(pred)).cpu().sum().item()

        test_loss /= len(test_loader)
        print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.3f}%)\n'.format(
            test_loss, correct, len(test_loader.dataset),
            100. * correct / len(test_loader.dataset)))
        scheduler.step(test_loss)
        if 100. * correct / len(test_loader.dataset) > best_acc:
            best_acc=100. * correct / len(test_loader.dataset)
            best_model=model
            torch.save(best_model,'capsule_model')
            print('model change')





