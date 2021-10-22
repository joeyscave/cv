import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt

# 读入待处理图片并输出至屏幕
img = cv.imread('Raw.jpg',0)
cv.imshow('Before Equalize',img)
cv.waitKey(0)


def plot_Hist_cdf(img):
    """
    @brief 以bin=256计算灰度图的直方图和归一化累积分布函数并绘制在同一张图中,显示结果
    @:param img 此函数接收一张灰度图作为参数
    @:return cdf 返回灰度图的累积分布函数
    """
    # 计算直方图
    hist, bins = np.histogram(img.flatten(), 256, [0, 256])
    # 计算累积分布函数
    cdf = hist.cumsum()
    # 计算归一化累积分布函数，方便和直方图一起显示
    cdf_normalized = cdf * float(hist.max()) / cdf.max()
    # 绘制直方图和归一化累积分布函数
    plt.plot(cdf_normalized, color='b')
    plt.hist(img.flatten(), 256, [0, 256], color='r')
    plt.xlim([0, 256])
    plt.legend(('cdf', 'histogram'), loc='upper left')
    # 显示结果
    plt.show()
    # 返回累积分布函数
    return cdf

# 掩盖累积分布函数前面的0值部分
cdf_m = np.ma.masked_equal(plot_Hist_cdf(img),0)
# 计算均衡化映射表
cdf_m = (cdf_m - cdf_m.min())*255/(cdf_m.max()-cdf_m.min())
# 为被掩盖部分重新赋0值，并对映射表整体取整
cdf = np.ma.filled(cdf_m,0).astype('uint8')

# 映射得到均衡化后灰度图
imgE = cdf[img]

# 显示均衡化后直方图、归一化累积分布函数
plot_Hist_cdf(imgE)
# 显示均衡化后灰度图
cv.imshow('After Equalize',imgE)
cv.waitKey(0)
# 保存均衡化后灰度图
cv.imwrite('Equa.jpg',imgE)