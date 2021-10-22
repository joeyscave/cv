import cv2
import os
import argparse
import glob
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
from models import DnCNN
from utils import *
from torch.utils.tensorboard import SummaryWriter
import torchvision.utils as utils

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# 参数管理
parser = argparse.ArgumentParser(description="DnCNN_Test")
parser.add_argument("--num_of_layers", type=int, default=17, help="Number of total layers")
parser.add_argument("--logdir", type=str, default="test_logs", help='path of log files')
parser.add_argument("--test_data", type=str, default='test', help='test on Test data')
parser.add_argument("--test_noiseL", type=float, default=25, help='noise level used on test set')
opt = parser.parse_args()

# 归一化
def normalize(data):
    return data / 255.


def main():
    writer = SummaryWriter(opt.logdir)  # 记录logs
    # Build model
    print('Loading model ...\n')
    net = DnCNN(channels=1, num_of_layers=opt.num_of_layers)    # 实例化网络
    device_ids = [0]
    model = nn.DataParallel(net, device_ids=device_ids).cuda()
    model.load_state_dict(torch.load(os.path.join(opt.logdir, 'net.pth')))  # 载入模型
    model.eval()    # 测试模式，不更新BN参数
    # load data info
    print('Loading data info ...\n')
    files_source = glob.glob(os.path.join('data', opt.test_data, '*.png'))
    files_source.sort()
    # process data
    psnr_test = 0   # 平均PSNR
    ssim_test = 0   # 平均SSIM
    step = 0
    for f in files_source:
        # image
        Img = cv2.imread(f)
        Img = normalize(np.float32(Img[:, :, 0]))
        Img = np.expand_dims(Img, 0)
        Img = np.expand_dims(Img, 1)
        ISource = torch.Tensor(Img)
        # noise
        noise = torch.FloatTensor(ISource.size()).normal_(mean=0, std=opt.test_noiseL / 255.)
        # noisy image
        INoisy = ISource + noise    # 生成加性白噪声图像
        ISource, INoisy = Variable(ISource.cuda()), Variable(INoisy.cuda())  # 激活GPU计算
        with torch.no_grad():  # 节省显存
            Out = torch.clamp(INoisy - model(INoisy), 0., 1.)
            psnr = batch_PSNR(Out, ISource, 1.)
            ssim = batch_SSIM(Out, ISource, 1.)
            psnr_test += psnr
            ssim_test += ssim
        print("%s PSNR: %f  SSIM: %f" % (f, psnr,ssim))
        # 写入logs
        writer.add_scalar('PSNR on testing data', psnr, step + 1)
        writer.add_scalar('SSIM on testing data', ssim, step + 1)
        writer.add_image('test : clean image', ISource[0], step + 1)
        writer.add_image('test : noisy image', INoisy[0], step + 1)
        writer.add_image('test : reconstructed image', Out[0], step + 1)
        step += 1

    psnr_test /= len(files_source)
    ssim_test /= len(files_source)
    print("\nPSNR on test data %f" % psnr_test)
    print("\nSSIM on test data %f" % ssim_test)
    writer.close()


if __name__ == "__main__":
    main()
