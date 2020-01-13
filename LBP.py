import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
from skimage import feature as skft
import os
from PIL import Image
import warnings
import pandas as pd

import cv2
warnings.filterwarnings("ignore")

radius = 1
n_point = radius * 8

def texture_detect(image_path,root_path):
    for dirpath, dirnames, filenames in os.walk(image_path):
        for filename in filenames:
            fullpath = os.path.join(dirpath, filename)
            index = filename.rfind('.')
            filename = filename[:index]

            print(filename)

            image = cv2.imread(fullpath, cv2.IMREAD_GRAYSCALE)

            # 使用LBP方法提取图像的纹理特征.
            lbp = skft.local_binary_pattern(image, n_point, radius, 'default')
            # 统计图像的直方图
            max_bins = int(lbp.max() + 1)
            # hist size:256
            image_hist, _ = np.histogram(lbp, normed=True, bins=max_bins, range=(0, max_bins))
            print(image_hist)

            # n, bins, patches = plt.hist(image_hist)
            # plt.show()
            # image_hist=np.reshape(image_hist,(1,image_hist.shape[0]))
            # data1 = pd.DataFrame(image_hist)
            # data1.to_csv(root_path + '/'+'LBP.csv', mode='a', header=False, index=False)






image_path='./tumor'
root_path='./tumor_feature'
texture_detect(image_path,root_path)