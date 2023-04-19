import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
import numpy as np
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sn
import pandas as pd

from utils import progress_bar
import torchvision
import torchvision.transforms as transforms

import os
import argparse

from models import *
from torch.utils.tensorboard import SummaryWriter
import datetime
from cosine_annealing_warmup import CosineAnnealingWarmupRestarts


parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
parser.add_argument('--save-dir', default='save_temp', type=str)
parser.add_argument('--seed', default=0, type=int)
parser.add_argument('--ls', default=0, type=float)
parser.add_argument('--device', default=0, type=int)
parser.add_argument('--alpha', default=1., type=float)

args = parser.parse_args()
torch.cuda.set_device(args.device)
torch.manual_seed(args.seed)
device = 'cuda'
best_acc = 0
start_epoch = 0
today = datetime.datetime.now()
writer = SummaryWriter(args.save_dir+"/"+today.strftime("%Y%m%d%H%M%S"))

# Data
print('==> Preparing data..')
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    # transforms.ColorJitter(),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)), 
])

trainset = torchvision.datasets.CIFAR10(
    root='./data', train=True, download=True, transform=transform_train)
trainloader = torch.utils.data.DataLoader(
    trainset, batch_size=128, shuffle=True, num_workers=8)

testset = torchvision.datasets.CIFAR10(
    root='./data', train=False, download=True, transform=transform_test)
testloader = torch.utils.data.DataLoader(
    testset, batch_size=100, shuffle=False, num_workers=8)

classes = ('plane', 'car', 'bird', 'cat', 'deer',
           'dog', 'frog', 'horse', 'ship', 'truck')
y_pred = []
y_true = []

print('==> Building model..')

net = ResNet50()

net = net.to(device)
if device == 'cuda':
    cudnn.benchmark = True

criterion = nn.CrossEntropyLoss(label_smoothing=args.ls)
optimizer = optim.SGD(net.parameters(), lr=args.lr,
                      momentum=0.9, weight_decay=5e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)
# scheduler = CosineAnnealingWarmupRestarts(optimizer,
#                                           first_cycle_steps=200,
#                                           cycle_mult=1.0,
#                                           max_lr=0.1,
#                                           min_lr=0.001,
#                                           warmup_steps=5,
#                                           gamma=1.0)

def mixup_data(x, y, alpha=1.0):
    '''Returns mixed inputs, pairs of targets, and lambda'''
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size()[0]
    index = torch.randperm(batch_size).cuda()

    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam

def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)

def train(epoch):
    print('\nEpoch: %d' % epoch)
    net.train()
    train_loss = 0
    reg_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.cuda(), targets.cuda()

        inputs, targets_a, targets_b, lam = mixup_data(inputs, targets, args.alpha)
        inputs, targets_a, targets_b = map(Variable, (inputs, targets_a, targets_b))
        outputs = net(inputs)
        loss = mixup_criterion(criterion, outputs, targets_a, targets_b, lam)
        train_loss += loss.data
        _, predicted = torch.max(outputs.data, 1)
        total += targets.size(0)
        correct += (lam * predicted.eq(targets_a.data).cpu().sum().float()
                    + (1 - lam) * predicted.eq(targets_b.data).cpu().sum().float())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        progress_bar(batch_idx, len(trainloader),
                     'Loss: %.3f | Reg: %.5f | Acc: %.3f%% (%d/%d)'
                     % (train_loss/(batch_idx+1), reg_loss/(batch_idx+1),
                        100.*correct/total, correct, total))

    writer.add_scalar("Loss/Train", train_loss/(batch_idx+1), epoch)
    writer.add_scalar("Accuracy/Train", 100.*correct/total, epoch)


def test(epoch):
    global best_acc
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            if epoch + 1 == start_epoch + 200:
                y_pred.extend(predicted.data.cpu().numpy())
                y_true.extend(targets.data.cpu().numpy())

    print('Loss: %.3f | Acc: %.3f%% (%d/%d)' % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))
    writer.add_scalar("Loss/Test", test_loss/(batch_idx+1), epoch)
    writer.add_scalar("Accuracy/Test", 100.*correct/total, epoch)

    # Save checkpoint.
    acc = 100.*correct/total
    if acc > best_acc:
        best_acc = acc


for epoch in range(start_epoch, start_epoch+200):
    train(epoch)
    test(epoch)
    scheduler.step()

print("Best Acc =", best_acc)

cf_matrix = confusion_matrix(y_true, y_pred)
df_cm = pd.DataFrame(cf_matrix / np.sum(cf_matrix, axis=1)[:, None], index = [i for i in classes],
                     columns = [i for i in classes])
plt.figure(figsize = (12,7))
sn.heatmap(df_cm, annot=True)
plt.savefig('output.png')
