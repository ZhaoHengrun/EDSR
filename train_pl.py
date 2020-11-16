import argparse
import os
import torch
import random
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import models
from model import Net
from data import get_training_set
import visdom
import numpy as np
import copy

torch.cuda.set_device(0)  # use the chosen gpu
vis = visdom.Visdom(env='EDSR_total_3_3')

# Training settings
parser = argparse.ArgumentParser(description="PyTorch EDSR")
parser.add_argument("--batchSize", type=int, default=16, help="training batch size")  # default 16
parser.add_argument("--nEpochs", type=int, default=500, help="number of epochs to train for")
parser.add_argument("--lr", type=float, default=1e-4, help="Learning Rate. Default=1e-4")
parser.add_argument("--step", type=int, default=30,
                    help="Sets the learning rate to the initial LR decayed by momentum every n epochs, Default: n=10")
parser.add_argument("--cuda", default=True, action="store_true", help="use cuda?")
parser.add_argument("--resume", default='', type=str,
                    help="path to latest checkpoint (default: none)")
parser.add_argument("--start-epoch", default=1, type=int, help="manual epoch number (useful on restarts)")
parser.add_argument("--threads", type=int, default=1, help="number of threads for data loader to use")
parser.add_argument("--momentum", default=0.9, type=float, help="momentum")
parser.add_argument("--weight-decay", "--wd", default=0, type=float, help="weight decay, Default: 0")

min_avr_loss = 99999999
save_flag = 0
epoch_avr_loss = 0
n_iter = 0


class FeatureExtractor(nn.Module):
    # relu2_2:8 relu3_3:15 relu4_3:22 relu5_1:25 relu5_3:29
    def __init__(self, cnn, feature_layer=15):
        super(FeatureExtractor, self).__init__()
        self.features = nn.Sequential(*list(cnn.features.children())[:(feature_layer + 1)])

    def forward(self, x):
        return self.features(x)


def create_loss_model(vgg, end_layer, use_maxpool=True, use_cuda=False):
    """
        [1] uses the output of vgg16 relu2_2 layer as a loss function (layer8 on PyTorch default vgg16 model).
        This function expects a vgg16 model from PyTorch and will return a custom version up until layer = end_layer
        that will be used as our loss function.
    """

    vgg = copy.deepcopy(vgg)

    model = nn.Sequential()

    if use_cuda:
        model.cuda()

    i = 0
    for layer in list(vgg):

        if i > end_layer:
            break

        if isinstance(layer, nn.Conv2d):
            name = "conv_" + str(i)
            model.add_module(name, layer)

        if isinstance(layer, nn.ReLU):
            name = "relu_" + str(i)
            model.add_module(name, layer)

        if isinstance(layer, nn.MaxPool2d):
            name = "pool_" + str(i)
            if use_maxpool:
                model.add_module(name, layer)
            else:
                avgpool = nn.AvgPool2d(kernel_size=layer.kernel_size, stride=layer.stride, padding=layer.padding)
                model.add_module(name, avgpool)
        i += 1
    return model


def main():
    global opt, model
    opt = parser.parse_args()
    print(opt)

    cuda = opt.cuda
    if cuda and not torch.cuda.is_available():
        raise Exception("No GPU found, please run without --cuda")

    opt.seed = random.randint(1, 10000)
    print("Random Seed: ", opt.seed)
    torch.manual_seed(opt.seed)
    if cuda:
        torch.cuda.manual_seed(opt.seed)

    cudnn.benchmark = True

    print("===> Loading datasets")
    # train_set = DatasetFromHdf5("path_to_dataset.h5")
    train_set = get_training_set()
    training_data_loader = DataLoader(dataset=train_set, batch_size=opt.batchSize, shuffle=True)

    print("===> Building model")
    model = Net()

    criterion = nn.L1Loss()

    if cuda:
        model = model.cuda()
        criterion = criterion.cuda()

    # optionally resume from a checkpoint
    if opt.resume:
        if os.path.isfile(opt.resume):
            print("=> loading checkpoint '{}'".format(opt.resume))
            checkpoint = torch.load(opt.resume)
            opt.start_epoch = 374
            model.load_state_dict(checkpoint.state_dict())
        else:
            print("=> no checkpoint found at '{}'".format(opt.resume))

    print("===> Setting Optimizer")
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=opt.lr,
                           weight_decay=opt.weight_decay, betas=(0.9, 0.999), eps=1e-08)
    # optimizer = optim.Adam(model.parameters(), lr=opt.lr,
    #                        weight_decay=opt.weight_decay, betas=(0.9, 0.999), eps=1e-08)

    print("===> Training")
    for epoch in range(opt.start_epoch, opt.nEpochs + 1):
        train(training_data_loader, optimizer, model, criterion, epoch)
        save_checkpoint(model, epoch)


def adjust_learning_rate(optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 10"""
    lr = opt.lr * (0.1 ** (epoch // opt.step))
    print('lr{}  iter:'.format(lr, n_iter))
    return lr


def train(training_data_loader, optimizer, model, criterion, epoch):
    global min_avr_loss
    global save_flag
    global epoch_avr_loss
    global n_iter

    L2_loss = nn.MSELoss()
    vgg16 = models.vgg16(pretrained=True).features
    if opt.cuda:
        vgg16.cuda()
        L2_loss.cuda()
    # get_feature = create_loss_model(vgg16, 8, use_maxpool=True, use_cuda=True)
    get_feature = FeatureExtractor(models.vgg16(pretrained=True))
    if opt.cuda:
        get_feature.cuda()
    print(get_feature)

    avr_loss = 0
    avr_vgg_loss = 0
    avr_total_loss = 0

    lr = adjust_learning_rate(optimizer, epoch - 1)

    for param_group in optimizer.param_groups:
        param_group["lr"] = lr

    print("Epoch={}, lr={}".format(epoch, optimizer.param_groups[0]["lr"]))
    model.train()

    for iteration, batch in enumerate(training_data_loader, 1):
        n_iter = iteration
        input, target = Variable(batch[0], requires_grad=False), Variable(batch[1], requires_grad=False)
        # input, target = batch[0].cuda(), batch[1].cuda()

        if opt.cuda:
            input = input.cuda()
            target = target.cuda()

        out = model(input)

        out_feature = get_feature(out)
        real_feature = get_feature(target)
        vgg_loss = L2_loss(out_feature, real_feature)

        loss = criterion(out, target)
        total_loss = loss + 0.025 * vgg_loss
        optimizer.zero_grad()
        # loss.backward()
        total_loss.backward()
        optimizer.step()

        avr_total_loss += total_loss.item()
        avr_loss += loss.item()
        avr_vgg_loss += vgg_loss.item()

        # if iteration % 100 == 0:
        print("===> Epoch[{}]({}/{}): total_loss{:.8f} L1_Loss: {:.8f} vgg_Loss: {:.8f}".format(
            epoch, iteration,
            len(training_data_loader),
            total_loss.item(),
            loss.item(),
            vgg_loss.item(),
        ))
    avr_total_loss = avr_total_loss / len(training_data_loader)
    avr_loss = avr_loss / len(training_data_loader)
    avr_vgg_loss = avr_vgg_loss / len(training_data_loader)

    vis.line(Y=np.array([avr_total_loss]), X=np.array([epoch]),
             win='total_loss',
             opts=dict(title='total_loss'),
             update='append'
             )
    vis.line(Y=np.array([avr_loss]), X=np.array([epoch]),
             win='L1_loss',
             opts=dict(title='L1_loss'),
             update='append'
             )
    vis.line(Y=np.array([avr_vgg_loss]), X=np.array([epoch]),
             win='vgg_loss',
             opts=dict(title='vgg_loss'),
             update='append'
             )
    epoch_avr_loss = avr_loss
    if epoch_avr_loss < min_avr_loss:
        min_avr_loss = epoch_avr_loss
        print('|||||||||||||||||||||min_batch_loss is {:.10f}|||||||||||||||||||||'.format(min_avr_loss))
        save_flag = True
    else:
        save_flag = False


def save_checkpoint(model, epoch):
    global min_avr_loss
    global save_flag

    model_folder = "checkpoints/3_3/"
    model_out_path = model_folder + "model_epoch_{}.pth".format(epoch)
    # state = {"epoch": epoch, "model": model}
    torch.save(model, model_out_path)
    if not os.path.exists(model_folder):
        os.makedirs(model_folder)

    if save_flag is True:
        torch.save(model, '{}epoch_{}_min_batch_loss_{}.pth'.format(model_folder, epoch, min_avr_loss))
        print('min_loss model saved')

    print("Checkpoint saved to {}".format(model_out_path))


if __name__ == "__main__":
    main()
# ssh -L 8097:127.0.0.1:8097 zhr@192.168.1.102
