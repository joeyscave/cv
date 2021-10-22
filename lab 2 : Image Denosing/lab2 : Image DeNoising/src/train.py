import os
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.utils as utils
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from models import DnCNN
from dataset import prepare_data, Dataset
from utils import *

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# 参数管理
parser = argparse.ArgumentParser(description="DnCNN")
parser.add_argument("--preprocess", type=bool, default=False, help='run prepare_data or not')
parser.add_argument("--batchSize", type=int, default=64, help="Training batch size")
parser.add_argument("--num_of_layers", type=int, default=17, help="Number of total layers")
parser.add_argument("--epochs", type=int, default=50, help="Number of training epochs")
parser.add_argument("--milestone", type=int, default=30, help="When to decay learning rate; should be less than epochs")
parser.add_argument("--lr", type=float, default=1e-3, help="Initial learning rate")
parser.add_argument("--outf", type=str, default="train_logs", help='path of log files')
parser.add_argument("--logdir", type=str, default="test_logs", help='path of log files')
parser.add_argument("--noiseL", type=float, default=25, help='noise level')
parser.add_argument("--val_noiseL", type=float, default=25, help='noise level used on validation set')
opt = parser.parse_args()


def main():
    # Load dataset
    print('Loading dataset ...\n')
    dataset_train = Dataset(train=True)
    dataset_val = Dataset(train=False)
    loader_train = DataLoader(dataset=dataset_train, num_workers=4, batch_size=opt.batchSize, shuffle=True)
    print("# of training samples: %d\n" % int(len(dataset_train)))
    # Build model
    net = DnCNN(channels=1, num_of_layers=opt.num_of_layers)    # 实例化网络，通道数为1（针对灰度图像）
    net.apply(weights_init_kaiming)  # 权重初始化
    criterion = nn.MSELoss(reduction='sum')  # loss标准为L2均方和
    # Move to GPU
    device_ids = [0]
    model = nn.DataParallel(net, device_ids=device_ids).cuda()
    criterion.cuda()
    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=opt.lr)   # 使用Adam优化算法
    # training
    writer = SummaryWriter(opt.outf)    # 记录训练logs
    step = 0
    for epoch in range(opt.epochs):
        if epoch < opt.milestone:
            current_lr = opt.lr
        else:   # 当周期数超过milestone，衰减学习率，防止过拟合
            current_lr = opt.lr / 10.
        # set learning rate
        for param_group in optimizer.param_groups:
            param_group["lr"] = current_lr
        print('learning rate %f' % current_lr)
        # train
        for i, data in enumerate(loader_train, 0):
            # training step
            model.train()  # 训练模式，确保BN参数在训练过程中更新
            model.zero_grad()  # 初始化模型梯度
            optimizer.zero_grad()  # 初始化优化器梯度
            img_train = data
            noise = torch.FloatTensor(img_train.size()).normal_(mean=0, std=opt.noiseL / 255.)  # 生成加性白噪声
            imgn_train = img_train + noise  # 生成噪声图像
            img_train, imgn_train = Variable(img_train.cuda()), Variable(imgn_train.cuda())  # 激活GPU计算
            noise = Variable(noise.cuda())
            out_train = model(imgn_train)   # 应用模型
            loss = criterion(out_train, noise) / (imgn_train.size()[0] * 2)  # 计算loss
            loss.backward()     # 反向传播
            optimizer.step()    # 更新网络参数
            # results
            model.eval()  # 测试模式，确保BN参数在训练过程中不变
            out_train = torch.clamp(imgn_train - model(imgn_train), 0., 1.)  # 归一化
            psnr_train = batch_PSNR(out_train, img_train, 1.)   # 计算信噪比
            print("[epoch %d][%d/%d] loss: %.4f PSNR_train: %.4f" %
                  (epoch + 1, i + 1, len(loader_train), loss.item(), psnr_train))
            if step % 10 == 0:
                # Log the scalar values
                writer.add_scalar('loss', loss.item(), step)
                writer.add_scalar('PSNR on training data', psnr_train, step)
            step += 1
        ## the end of each epoch
        model.eval()
        # validate
        psnr_val = 0    # 平均信噪比
        for k in range(len(dataset_val)):
            img_val = torch.unsqueeze(dataset_val[k], 0)  # 增加维度
            noise = torch.FloatTensor(img_val.size()).normal_(mean=0, std=opt.val_noiseL / 255.)
            imgn_val = img_val + noise  # 生成加性白噪声图像
            with torch.no_grad():   # 节省显存
                img_val, imgn_val = Variable(img_val.cuda()), Variable(imgn_val.cuda())
                out_val = torch.clamp(imgn_val - model(imgn_val), 0., 1.)
                psnr_val += batch_PSNR(out_val, img_val, 1.)
        psnr_val /= len(dataset_val)  # 计算平均信噪比
        print("\n[epoch %d] PSNR_val: %.4f" % (epoch + 1, psnr_val))
        writer.add_scalar('PSNR on validation data', psnr_val, epoch + 1)
        # log the images
        out_train = torch.clamp(imgn_train - model(imgn_train), 0., 1.)
        Img = utils.make_grid(img_train.data, nrow=8, normalize=True, scale_each=True)  # 原图像网格
        Imgn = utils.make_grid(imgn_train.data, nrow=8, normalize=True, scale_each=True)    # 噪声图像网格
        Irecon = utils.make_grid(out_train.data, nrow=8, normalize=True, scale_each=True)   # 降噪图像网格
        writer.add_image('clean image', Img, epoch)
        writer.add_image('noisy image', Imgn, epoch)
        writer.add_image('reconstructed image', Irecon, epoch)
        # save model
        torch.save(model.state_dict(), os.path.join(opt.outf, 'net.pth'))
        torch.save(model.state_dict(), os.path.join(opt.logdir, 'net.pth'))


if __name__ == "__main__":
    if opt.preprocess:  # 如需要，进行数据集预处理
        prepare_data(data_path='data', patch_size=40, stride=10, aug_times=1)
    main()
