from __future__ import division
import argparse
import os
import shutil
import time

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import numpy as np

import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.autograd import Variable

from models.alexnet import AlexNet, AlexClassifier
from models.vgg import VGGClassifier, vgg16_bn
from models.resnet import ResClassifier, ResNet18

from models.discriminator import *



parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--epochs', default=300, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch_size', default=512, type=int,
                    metavar='N', help='batch_size (default: 32')
parser.add_argument('--lr', '--learning_rate', default=0.001, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--weight-decay', '--wd', default=5e-4, type=float,
                    metavar='W', help='weight decay (default: 5e-4)')
parser.add_argument('--print_freq', '-p', default=2, type=int,
                    metavar='N', help='print frequency (default: 20)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--pretrained', dest='pretrained', action='store_true',
                    help='use pre-trained model')
parser.add_argument('--half', dest='half', action='store_true',
                    help='use half-precision(16-bit) ')
parser.add_argument('--cuda', dest='cuda', action='store_true',
                    help='use gpu ')
parser.add_argument('--save_dir', dest='save_dir',
                    help='The directory used to save the trained models',
                    default='checkpoints', type=str)


opt = parser.parse_args()

best_prec1 = 0

normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
# device = 'cuda' if opt.cuda else 'cpu'
device = 'cuda'
print(opt.cuda)
# % update


# define loss function (criterion) and pptimizer

def train(opt, trainloader, valloader):
    # network
    
    G1 = ResNet18().to(device)
    G2 = AlexNet().to(device)

    D1 = Discriminator().to(device)
    D2 = Discriminator().to(device)

    G1_classifier = ResClassifier().to(device)
    G2_classifier = AlexClassifier().to(device)

    criterion = nn.CrossEntropyLoss().to(device)
    # optimizer
    #optimizer_G1 = torch.optim.SGD(G1.parameters(), lr=opt.lr, momentum=opt.momentum)
    #optimizer_G2 = torch.optim.SGD(G2.parameters(), lr=opt.lr, momentum=opt.momentum)
    #optimizer_D1 = torch.optim.SGD(D1.parameters(), lr=opt.lr, momentum=opt.momentum)
    #optimizer_D2 = torch.optim.SGD(D2.parameters(), lr=opt.lr, momentum=opt.momentum)
    #optimizer_G1c = torch.optim.SGD(G1_classifier.parameters(), lr=opt.lr, momentum=opt.momentum)
    #optimizer_G2c = torch.optim.SGD(G2_classifier.parameters(), lr=opt.lr, momentum=opt.momentum)


    optimizer_G1 = torch.optim.Adam(G1.parameters(), lr=opt.lr)
    optimizer_G2 = torch.optim.Adam(G2.parameters(), lr=opt.lr)
    optimizer_D1 = torch.optim.Adam(D1.parameters(), lr=opt.lr)
    optimizer_D2 = torch.optim.Adam(D2.parameters(), lr=opt.lr)
    optimizer_G1c = torch.optim.Adam(G1_classifier.parameters(), lr=opt.lr)
    optimizer_G2c = torch.optim.Adam(G2_classifier.parameters(), lr=opt.lr)

    scheduler_G1 = torch.optim.lr_scheduler.MultiStepLR(optimizer_G1, milestones=[50, 100, 150, 200, 250], gamma=0.5)
    scheduler_G2 = torch.optim.lr_scheduler.MultiStepLR(optimizer_G2, milestones=[50, 100, 150, 200, 250], gamma=0.5)
    scheduler_D1 = torch.optim.lr_scheduler.MultiStepLR(optimizer_D1, milestones=[50, 100, 150, 200, 250], gamma=0.5)
    scheduler_D2 = torch.optim.lr_scheduler.MultiStepLR(optimizer_D2, milestones=[50, 100, 150, 200, 250], gamma=0.5)
    scheduler_G1c = torch.optim.lr_scheduler.MultiStepLR(optimizer_G1c, milestones=[50, 100, 150, 200, 250], gamma=0.5)
    scheduler_G2c = torch.optim.lr_scheduler.MultiStepLR(optimizer_G2c, milestones=[50, 100, 150, 200, 250], gamma=0.5)

    if opt.pretrained:
        # Load checkpoint.
        print('==> Resuming from checkpoint..')
        assert os.path.isdir(opt.resume), 'Error: no checkpoint directory found!'
        G1.load_state_dict(torch.load(os.path.join(opt.resume, 'netG1.pth')))
        G2.load_state_dict(torch.load(os.path.join(opt.resume, 'netG2.pth')))
        D1.load_state_dict(torch.load(os.path.join(opt.resume, 'netD1.pth')))
        D2.load_state_dict(torch.load(os.path.join(opt.resume, 'netD2.pth')))
        G1_classifier.load_state_dict(torch.load(os.path.join(opt.resume, 'netG1_classifier.pth')))
        G2_classifier.load_state_dict(torch.load(os.path.join(opt.resume, 'netG2_classifier.pth')))

    for epoch in range(opt.start_epoch, opt.start_epoch + opt.epochs):
        scheduler_G1.step()
        scheduler_G2.step()
        scheduler_D1.step()
        scheduler_D2.step()
        scheduler_G1c.step()
        scheduler_G2c.step()

        for batch_idx, (inputs, targets) in enumerate(trainloader):
            image = Variable(inputs).to(device)
            label = Variable(targets).to(device)
            
            # Update D network
            optimizer_D1.zero_grad()
            optimizer_D2.zero_grad()

            for p in D1.parameters():
                p.requires_grad = True  # to ensure computation
            for p in D2.parameters():
                p.requires_grad = True  # to ensure computation
            for p in G1.parameters():
                p.requires_grad = False  # to avoid computation
            for p in G2.parameters():
                p.requires_grad = False  # to avoid computation

            loss_D1, loss_D2 = update_D(image, G1, G2, D1, D2, criterion)
            optimizer_D1.step()
            optimizer_D2.step()

            # Update G network
            optimizer_G1.zero_grad()
            optimizer_G2.zero_grad()
            optimizer_G1c.step()
            optimizer_G2c.step()

            for p in D1.parameters():
                p.requires_grad = False  # to avoid computation
            for p in D2.parameters():
                p.requires_grad = False  # to avoid computation
            for p in G1.parameters():
                p.requires_grad = True  # to ensure computation
            for p in G2.parameters():
                p.requires_grad = True  # to ensure computation

            loss_G1, loss_G2, loss_G1_cls, loss_G2_cls = update_G(image, label, G1, G2, D1, D2, criterion, G1_classifier, G2_classifier)
            optimizer_G1.step()
            optimizer_G2.step()
            optimizer_G1c.step()
            optimizer_G2c.step()

            if batch_idx % opt.print_freq == 0:
                print('[%d/%d][%d/%d]   loss_D1: %f  loss_D2: %f  loss_G1: %f  loss_G2 %f'
                      % (epoch, opt.epochs, batch_idx, len(trainloader), loss_D1.item(), loss_D2.item(), loss_G1.item(), loss_G2.item()))
                print("loss_G1_cls: {0}, loss_G2_cls: {1}".format(loss_G1_cls, loss_G2_cls))
                print()
        
        #torch.save(G1.state_dict(), '{0}/netG1.pth'.format(opt.save_dir))
        #torch.save(G2.state_dict(), '{0}/netG2.pth'.format(opt.save_dir))
        #torch.save(D1.state_dict(), '{0}/netD1.pth'.format(opt.save_dir))
        #torch.save(D2.state_dict(), '{0}/netD2.pth'.format(opt.save_dir))
        #torch.save(G1_classifier.state_dict(), '{0}/netG1_classifier.pth'.format(opt.save_dir))
        #torch.save(G2_classifier.state_dict(), '{0}/netG2_classifier.pth'.format(opt.save_dir))

        total = 0
        correct1 = 0
        correct2 = 0
        count = 0
        for batch_idx, (inputs, targets) in enumerate(valloader):
            count = count + 1
            image = Variable(inputs).to(device)
            label = Variable(targets).to(device)
            feature1 = G1(image)
            feature2 = G2(image)
            pred1 = G1_classifier(feature1)
            pred2 = G2_classifier(feature2)
            _, predicted1 = torch.max(pred1.data, 1)
            _, predicted2 = torch.max(pred2.data, 1)
            total += label.size(0)
            correct1 += (predicted1 == label).sum()
            correct2 += (predicted2 == label).sum()
        print('accuracy1：%.3f%%' % (100*np.double(correct1) / np.double(total)))
        print('accuracy2：%.3f%%' % (100*np.double(correct2) / np.double(total)))





        torch.save(G1.state_dict(), '/home/shared/codes/man/checkpoints/netG1.pth')
        torch.save(G2.state_dict(), '/home/shared/codes/man/checkpoints/netG2.pth')
        torch.save(D1.state_dict(), '/home/shared/codes/man/checkpoints/netD1.pth')
        torch.save(D2.state_dict(), '/home/shared/codes/man/checkpoints/netD2.pth')
        torch.save(G1_classifier.state_dict(), '/home/shared/codes/man/checkpoints/netG1_classifier.pth')
        torch.save(G2_classifier.state_dict(), '/home/shared/codes/man/checkpoints/netG2_classifier.pth')


# update D1, D2
def update_D(image, G1, G2, D1, D2, criterion):
    feature1 = G1(image)
    feature2 = G2(image)

    output11 = D1(feature1)  # to avoid compute G1 & G2
    output12 = D1(feature2)

    output22 = D2(feature2)
    output21 = D2(feature1)
    loss_D1_1 = criterion(output11, torch.ones(image.size(0)).long().to(device))
    loss_D1_2 = criterion(output12, torch.zeros(image.size(0)).long().to(device))
    loss_D1 = 0.1*loss_D1_1 + 0.1*loss_D1_2
    loss_D2_1 = criterion(output21, torch.zeros(image.size(0)).long().to(device))
    loss_D2_2 = criterion(output22, torch.ones(image.size(0)).long().to(device))
    loss_D2 = 0.1*loss_D2_1 + 0.1*loss_D2_2

    loss_D1.backward()
    loss_D2.backward()
    return loss_D1, loss_D2

# update G1, G2 and C1, C2
def update_G(image, label, G1, G2, D1, D2, criterion, G1_classifier, G2_classifier):
    feature1 = G1(image)
    feature2 = G2(image)

    output11 = D1(feature1)
    output12 = D1(feature2)

    output22 = D2(feature2)
    output21 = D2(feature1)

    loss_G1_real = criterion(output11, torch.ones(image.size(0)).long().to(device))
    loss_G1_fake = criterion(output12, torch.ones(image.size(0)).long().to(device))
    loss_G1_cls = criterion(G1_classifier(feature1), label)

    loss_G1 = 0.1*loss_G1_real + 0.1*loss_G1_fake + loss_G1_cls
    loss_G1.backward(retain_graph=True)

    loss_G2_real = criterion(output22, torch.ones(image.size(0)).long().to(device))
    loss_G2_fake = criterion(output21, torch.ones(image.size(0)).long().to(device))
    loss_G2_cls = criterion(G2_classifier(feature2), label)

    loss_G2 = 0.1*loss_G2_real + 0.1*loss_G2_fake + loss_G2_cls
    loss_G2.backward()

    return loss_G1, loss_G2, loss_G1_cls, loss_G2_cls

def main(opt):
    train_loader = torch.utils.data.DataLoader(datasets.CIFAR10(root='./data', train=True,  transform=transforms.Compose([
                 transforms.RandomHorizontalFlip(),
                 transforms.RandomCrop(32, 4),
                 transforms.ToTensor(),
                 normalize,
             ]), download=True),
             batch_size=opt.batch_size, shuffle=True,
             num_workers=opt.workers, pin_memory=True)
        
    val_loader = torch.utils.data.DataLoader(
        datasets.CIFAR10(root='./data', train=False, download=True,transform=transforms.Compose([    
            transforms.ToTensor(),
                normalize,
            ])),
            batch_size=opt.batch_size, shuffle=False,
            num_workers=opt.workers, pin_memory=True)

    train(opt, train_loader, val_loader)


if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = "7"
    main(opt)
