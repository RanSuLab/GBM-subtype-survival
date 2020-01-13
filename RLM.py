import numpy as np
import itertools
import cv2
import warnings
import pandas as pd
import os

warnings.filterwarnings("ignore")

def length(l):
    if hasattr(l, '__len__'):
        return np.size(l)
    else:
        i = 0
        for _ in l:
            i += 1
        return i


def getGrayLevelRumatrix(path,theta):
    '''
    计算给定图像的灰度游程矩阵
    参数：
    array: 输入，需要计算的图像
    theta: 输入，计算灰度游程矩阵时采用的角度，list类型，可包含字段:['deg0', 'deg45', 'deg90', 'deg135']
    glrlm: 输出，灰度游程矩阵的计算结果
    '''
    P = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    x, y = P.shape

    min_pixels = np.min(P)  # 图像中最小的像素值
    run_length = max(x, y)  # 像素的最大游行长度
    num_level = np.max(P) - np.min(P) + 1  # 图像的灰度级数
    # num_level=16

    deg0 = [val.tolist() for sublist in np.vsplit(P, x) for val in sublist]  # 0度矩阵统计


    deg90 = [val.tolist() for sublist in np.split(np.transpose(P), y) for val in sublist]  # 90度矩阵统计

    diags = [P[::-1, :].diagonal(i) for i in range(-P.shape[0] + 1, P.shape[1])]  # 45度矩阵统计
    deg45 = [n.tolist() for n in diags]

    Pt = np.rot90(P, 3)  # 135度矩阵统计
    diags = [Pt[::-1, :].diagonal(i) for i in range(-Pt.shape[0] + 1, Pt.shape[1])]
    deg135 = [n.tolist() for n in diags]


    glrlm = np.zeros((num_level, run_length, len(theta)))  # 按照统计矩阵记录所有的数据， 第三维度表示计算角度


    for angle in theta:
        for splitvec in range(0, len(eval(angle))):
            flattened = eval(angle)[splitvec]
            answer = []
            for key, iter in itertools.groupby(flattened):  # 计算单个矩阵的像素统计信息
                answer.append((key, length(iter)))
            for ansIndex in range(0, len(answer)):
                glrlm[int(answer[ansIndex][0] - min_pixels), int(answer[ansIndex][1] - 1), theta.index(
                    angle)] += 1  # 每次将统计像素值减去最小值就可以填入GLRLM矩阵中
    return glrlm



def apply_over_degree(function, x1, x2):
    rows, cols, nums = x1.shape
    result = np.ndarray((rows, cols, nums))
    for i in range(nums):
        # print(x1[:, :, i])
        result[:, :, i] = function(x1[:, :, i], x2)
        # print(result[:, :, i])
    result[result == np.inf] = 0
    result[np.isnan(result)] = 0
    return result


def calcuteIJ(rlmatrix):
    gray_level, run_length, _ = rlmatrix.shape
    I, J = np.ogrid[0:gray_level, 0:run_length]
    return I, J + 1


def calcuteS(rlmatrix):
    return np.apply_over_axes(np.sum, rlmatrix, axes=(0, 1))[0, 0]


#1. SRE
def getShortRunEmphasis(rlmatrix):
    I, J = calcuteIJ(rlmatrix)
    numerator = np.apply_over_axes(np.sum, apply_over_degree(np.divide, rlmatrix, (J * J)), axes=(0, 1))[0, 0]
    S = calcuteS(rlmatrix)
    return numerator / S
#2. LRE
def getLongRunEmphasis(rlmatrix):
    I, J = calcuteIJ(rlmatrix)
    numerator = np.apply_over_axes(np.sum,  apply_over_degree(np.multiply, rlmatrix, (J * J)), axes=(0, 1))[
        0, 0]
    S = calcuteS(rlmatrix)
    return numerator / S
#3. GLN
def getGrayLevelNonUniformity(rlmatrix):
    G = np.apply_over_axes(np.sum, rlmatrix, axes=1)
    numerator = np.apply_over_axes(np.sum, (G * G), axes=(0, 1))[0, 0]
    S = calcuteS(rlmatrix)
    return numerator / S

# 4. RLN
def getRunLengthNonUniformity(rlmatrix):
    R = np.apply_over_axes(np.sum, rlmatrix, axes=0)
    numerator = np.apply_over_axes(np.sum, (R * R), axes=(0, 1))[0, 0]
    S = calcuteS(rlmatrix)
    return numerator / S


# 5. RP
def getRunPercentage(rlmatrix):
    gray_level, run_length, _ = rlmatrix.shape
    num_voxels = gray_level * run_length
    return calcuteS(rlmatrix) / num_voxels


# 6. LGLRE
def getLowGrayLevelRunEmphasis(rlmatrix):
    I, J = calcuteIJ(rlmatrix)
    numerator = np.apply_over_axes(np.sum, apply_over_degree(np.divide, rlmatrix, (I * I)), axes=(0, 1))[0, 0]
    S = calcuteS(rlmatrix)
    return numerator / S


# 7. HGLRE
def getHighGrayLevelRunEmphais(rlmatrix):
    I, J = calcuteIJ(rlmatrix)
    numerator = np.apply_over_axes(np.sum, apply_over_degree(np.multiply, rlmatrix, (I * I)), axes=(0, 1))[
        0, 0]
    S = calcuteS(rlmatrix)
    return numerator / S


# 8. SRLGLE
def getShortRunLowGrayLevelEmphasis(rlmatrix):
    I, J = calcuteIJ(rlmatrix)
    numerator = \
        np.apply_over_axes(np.sum, apply_over_degree(np.divide, rlmatrix, (I * I * J * J)), axes=(0, 1))[0, 0]
    S = calcuteS(rlmatrix)
    return numerator / S


# 9. SRHGLE
def getShortRunHighGrayLevelEmphasis(rlmatrix):
    I, J = calcuteIJ(rlmatrix)
    temp = apply_over_degree(np.multiply, rlmatrix, (I * I))
    numerator = np.apply_over_axes(np.sum, apply_over_degree(np.divide, temp, (J * J)), axes=(0, 1))[0, 0]
    S = calcuteS(rlmatrix)
    return numerator / S


# 10. LRLGLE
def getLongRunLow(rlmatrix):
    I, J = calcuteIJ(rlmatrix)
    temp = apply_over_degree(np.multiply, rlmatrix, (J * J))
    numerator = np.apply_over_axes(np.sum, apply_over_degree(np.divide, temp, (I * I)), axes=(0, 1))[0, 0]
    S = calcuteS(rlmatrix)
    return numerator / S

# 11. LRHGLE
def getLongRunHighGrayLevelEmphais(rlmatrix):
    I, J = calcuteIJ(rlmatrix)
    numerator = \
        np.apply_over_axes(np.sum, apply_over_degree(np.multiply, rlmatrix, (I * I * J * J)), axes=(0, 1))[0, 0]
    S = calcuteS(rlmatrix)
    return numerator / S


def get_feature(image_path,csv_path):
    theta=['deg0', 'deg45', 'deg90', 'deg135']
    for dirpath, dirnames, filenames in os.walk(image_path):
        for filename in filenames:
            fullpath = os.path.join(dirpath, filename)
            index = filename.rfind('.')
            filename = filename[:index]
            print(filename)
            rlmatrix = getGrayLevelRumatrix(fullpath, theta)
            # print(rlmatrix.shape)
            SRE = getShortRunEmphasis(rlmatrix)
            SRE=np.mean(SRE)
            # print(SRE)
            LRE = getLongRunEmphasis(rlmatrix)
            LRE = np.mean(LRE)
            # print(LRE)
            GLN = getGrayLevelNonUniformity(rlmatrix)
            GLN = np.mean(GLN)
            # print(GLN)
            RLN = getRunLengthNonUniformity(rlmatrix)
            RLN = np.mean(RLN)
            # print(RLN)
            RP = getRunPercentage(rlmatrix)
            RP = np.mean(RP)
            # print(RP)
            LGLRE = getLowGrayLevelRunEmphasis(rlmatrix)
            LGLRE = np.mean(LGLRE)
            # print(LGLRE)
            HGLRE = getHighGrayLevelRunEmphais(rlmatrix)
            HGLRE = np.mean(HGLRE)
            # print(HGLRE)
            SRLGLE = getShortRunLowGrayLevelEmphasis(rlmatrix)
            SRLGLE = np.mean(SRLGLE)
            # print(SRLGLE)
            SRHGLE = getShortRunHighGrayLevelEmphasis(rlmatrix)
            SRHGLE = np.mean(SRHGLE)
            # print(SRHGLE)
            LRLGLE = getLongRunLow(rlmatrix)
            LRLGLE = np.mean(LRLGLE)
            # print(LRLGLE)
            LRHGLE = getLongRunHighGrayLevelEmphais(rlmatrix)
            LRHGLE = np.mean(LRHGLE)
            # print(LRHGLE)
            list_1 = [SRE, LRE, GLN, RLN, RP, LGLRE, HGLRE, SRLGLE, SRHGLE, LRLGLE, LRHGLE]

            list = np.reshape(list_1, (1, 11))
            df_to_save = pd.DataFrame(list)
            df_to_save.to_csv(csv_path +'/'+ 'RLM.csv', mode='a+', header=False)






image_path='./tumor'
csv_path='./tumor_feature'
get_feature(image_path,csv_path)







