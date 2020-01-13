from sklearn.decomposition import PCA  #导入主成分分析库
import numpy as np
import os
import pandas as pd
from fishervector import FisherVectorGMM


#pca
def pca(data_path,save_path):
    for dirpath, dirnames, filenames in os.walk(data_path):
         filenames.sort(key=lambda x: int(x.split('.')[0]))
         for filename in filenames:
            fullpath = os.path.join(dirpath, filename)
            index = filename.rfind('.')
            filename = filename[:index]
            data = np.load(fullpath)
            # print(data.shape)
            data=np.reshape(data[:,:,:,0:32],(32*64,64*64))
            pca = PCA()
            pca.fit(data)
            # print("模型的各个特征向量:")
            # print(pca.components_)  # 返回模型的各个特征向量
            # print('每个成分各自方差百分比 :')
            # print(pca.explained_variance_ratio_)  # 返回各个成分各自的方差百分比

            pca = PCA(32)  # 保留主成分的个数
            pca.fit(data)
            low_d = pca.transform(data)  # 降低维度
            save_dir_1 = save_path + '/' + filename[:18]
            exists = os.path.exists(save_dir_1)
            if not exists:
               os.makedirs(save_dir_1)
            data1 = pd.DataFrame(low_d)
            data1.to_csv(save_dir_1 + '/' + filename+'csv', mode='a', header=False, index=False)



#fisher vector
def fisher_vector(sample_path,file_path, save_path):
   gmm = np.load(sample_path)
   gmm_reshape = np.reshape(gmm, (1, gmm.shape[0], gmm.shape[1]))
   print(gmm_reshape.shape)
   fv_gmm = FisherVectorGMM(n_kernels=64).fit(gmm_reshape)
   # fv_gmm = FisherVectorGMM().fit_by_bic(gmm_reshape, choices_n_kernels=[2,4,8,16,32,64])
   for dirpath, dirnames, filenames in os.walk(file_path):
      for filename in filenames:
         fullpath = os.path.join(dirpath, filename)
         image_data = np.load(fullpath)
         print(filename)
         image_data_reshape = np.reshape(image_data, (1, image_data.shape[0], image_data.shape[1]))
         print(image_data_reshape.shape)
         fv = fv_gmm.predict(image_data_reshape)
         fv_reshape = np.reshape(fv, (1, fv.shape[1] * fv.shape[2]))
         print(fv_reshape.shape)
         data1 = pd.DataFrame(fv_reshape)
         data1.to_csv(save_path + '/' + filename + '.csv', mode='a', header=False, index=False)
