import cv2
import numpy as np
import math
import matplotlib.pyplot as plt
import os
import pandas as pd

#计算每个像素的梯度
#计算图像横坐标和纵坐标方向的梯度，并据此计算每个像素位置的梯度方向值；
# 求导操作不仅能够捕获轮廓，人影和一些纹理信息，还能进一步弱化光照的影响。
# 在求出输入图像中像素点（x,y）处的水平方向梯度、垂直方向梯度和像素值，从而求出梯度幅值和方向。
def global_gradient(img):
    gradient_values_x = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=5)
    # print('gradient_values_x:',gradient_values_x.shape)
    # print(len(np.unique(gradient_values_x)))
    gradient_values_y = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=5)
    # print('gradient_values_y',gradient_values_y.shape)
    # print(len(np.unique(gradient_values_y)))
    gradient_magnitude = cv2.addWeighted(gradient_values_x, 0.5, gradient_values_y, 0.5, 0)
    gradient_angle = cv2.phase(gradient_values_x, gradient_values_y, angleInDegrees=True)
    # print('gradient_magnitude, gradient_angle:', gradient_magnitude.shape, gradient_angle.shape)

    return gradient_magnitude, gradient_angle




#为每个细胞单元构建梯度方向直方图
#我们将图像分成若干个“单元格cell”，默认我们将cell设为8*8个像素。
# 假设我们采用8个bin的直方图来统计这6*6个像素的梯度信息。也就是将cell的梯度方向360度分成8个方向块，
# 例如：如果这个像素的梯度方向是0-22.5度，直方图第1个bin的计数就加一，
# 这样，对cell内每个像素用梯度方向在直方图中进行加权投影（映射到固定的角度范围），就可以得到这个cell的梯度方向直方图了，
# 就是该cell对应的8维特征向量而梯度大小作为投影的权值。
# magnitude  幅度  angle 方向
def cell_gradient(bin_size,cell_magnitude, cell_angle):
    angle_unit = 360 / bin_size
    # print('angle_unit:',angle_unit)
    orientation_centers = [0] * bin_size
    # print('orientation_centers:',orientation_centers)
    for k in range(cell_magnitude.shape[0]):
        for l in range(cell_magnitude.shape[1]):
            gradient_strength = cell_magnitude[k][l]
            gradient_angle = cell_angle[k][l]
            min_angle = int(gradient_angle / angle_unit)%8
            max_angle = (min_angle + 1) % bin_size
            mod = gradient_angle % angle_unit
            orientation_centers[min_angle] += (gradient_strength * (1 - (mod / angle_unit)))
            orientation_centers[max_angle] += (gradient_strength * (mod / angle_unit))
    return orientation_centers
#可视化Cell梯度直方图
def render_gradient(img,cell_size,bin_size,cell_gradient):

    cell_width = cell_size / 2
    angle_unit = 360 / bin_size
    max_mag = np.array(cell_gradient).max()
    for x in range(cell_gradient.shape[0]):
        for y in range(cell_gradient.shape[1]):
            cell_grad = cell_gradient[x][y]
            cell_grad /= max_mag
            angle = 0
            angle_gap = angle_unit
            for magnitude in cell_grad:
                angle_radian = math.radians(angle)
                x1 = int(x * cell_size + magnitude * cell_width * math.cos(angle_radian))
                y1 = int(y * cell_size + magnitude * cell_width * math.sin(angle_radian))
                x2 = int(x * cell_size - magnitude * cell_width * math.cos(angle_radian))
                y2 = int(y * cell_size - magnitude * cell_width * math.sin(angle_radian))
                cv2.line(img, (y1, x1), (y2, x2), int(255 * math.sqrt(magnitude)))
                angle += angle_gap

    return img



def extract(img,cell_size,bin_size):
    height, width = img.shape
    # 计算每个像素的梯度
    gradient_magnitude, gradient_angle = global_gradient(img)
    print('gradient_magnitude, gradient_angle:',gradient_magnitude.shape, gradient_angle.shape)

    # 为每个细胞单元构建梯度方向直方图
    gradient_magnitude = abs(gradient_magnitude)

    cell_gradient_vector = np.zeros((int(height / cell_size), int(width / cell_size), bin_size))
    # (10,10,8)
    print('cell_gradient_vector:',cell_gradient_vector.shape)
    for i in range(cell_gradient_vector.shape[0]):

        for j in range(cell_gradient_vector.shape[1]):

            cell_magnitude = gradient_magnitude[i * cell_size:(i + 1) * cell_size,
                             j * cell_size:(j + 1) * cell_size]
            cell_angle = gradient_angle[i * cell_size:(i + 1) * cell_size,
                         j * cell_size:(j + 1) * cell_size]
            cell_gradient_vector[i][j] = cell_gradient(bin_size,cell_magnitude, cell_angle)



    #可视化Cell梯度直方图
    hog_image=render_gradient(np.zeros([height, width]),cell_size,bin_size,cell_gradient_vector)
    print('hog_image',hog_image.shape)
    # plt.imshow(hog_image, cmap=plt.cm.gray)
    # plt.show()


    # 统计Block的梯度信息
    #把细胞单元组合成大的块(block），块内归一化梯度直方图 由于局部光照的变化以及前景-背景对比度的变化，使得梯度强度的变化范围非常大。
    # 这就需要对梯度强度做归一化。归一化能够进一步地对光照、阴影和边缘进行压缩。把各个细胞单元组合成大的、空间上连通的区间（blocks）。
    # 这样，一个block内所有cell的特征向量串联起来便得到该block的HOG特征。
    # 这些区间是互有重叠的，本次实验采用的是矩阵形区间，
    # 它可以有三个参数来表征：每个区间中细胞单元的数目、每个细胞单元中像素点的数目、每个细胞的直方图通道数目。
    # 2*2细胞／区间、8*8像素／细胞、8个直方图通道,步长为1。则一块的特征数为2*2*8

    hog_vector = []

    for i in range(cell_gradient_vector.shape[0] - 1):
        for j in range(cell_gradient_vector.shape[1] - 1):

            block_vector = []
            block_vector.extend(cell_gradient_vector[i][j])
            block_vector.extend(cell_gradient_vector[i][j + 1])
            block_vector.extend(cell_gradient_vector[i + 1][j])
            block_vector.extend(cell_gradient_vector[i + 1][j + 1])
            mag = lambda vector: math.sqrt(sum(i ** 2 for i in vector))
            magnitude = mag(block_vector)
            if magnitude != 0:
                normalize = lambda block_vector, magnitude: [element / magnitude for element in block_vector]
                block_vector = normalize(block_vector, magnitude)
            hog_vector.append(block_vector)
    print('hog_vector:',np.array(hog_vector).shape)

    return hog_vector, hog_image
def texture_detect(image_path,root_path):
    for dirpath, dirnames, filenames in os.walk(image_path):
        for filename in filenames:
            fullpath = os.path.join(dirpath, filename)
            index = filename.rfind('.')
            filename = filename[:index]
            print(filename)
            crop_size = (40, 40)
            img = cv2.imread(fullpath,cv2.IMREAD_GRAYSCALE)
            img_new = cv2.resize(img, crop_size, interpolation=cv2.INTER_CUBIC)
            # cv2.namedWindow('input_image', cv2.WINDOW_AUTOSIZE)
            # cv2.imshow('input_image', img_new)
            # cv2.waitKey(0)

            # extract(img_new,cell_size=8,bin_size=9)
            vector, image = extract(img_new, cell_size=8, bin_size=9)
            hog_vector=np.reshape(vector,(1,np.array(vector).shape[0]*np.array(vector).shape[1]))
            print(hog_vector.shape)
            data1 = pd.DataFrame(hog_vector)
            data1.to_csv(root_path + '/' + 'hog.csv', mode='a', header=False, index=False)

            # print(np.array(vector).shape)


image_path='./tumor'
root_path='./tumor_feature'
texture_detect(image_path,root_path)