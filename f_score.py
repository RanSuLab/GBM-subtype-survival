import pandas as pd
import numpy as np
import math

np.seterr(invalid='ignore')
def fscore_core(n1,n2,n3,n4,xb,xb1,xb2,xb3,xb4,xk1,xk2,xk3,xk4):
    '''
    np: number of positive features
    nn: number of negative features
    reference: http://link.springer.com/chapter/10.1007/978-3-540-35488-8_13
    '''

    def sig1 (i,n1,xb1,xk1):
        return sum([(xk1[i][k]-xb1[i])**2 for k in range(int(n1))])

    def sig2 (i,n2,xb2,xk2):
        return sum([(xk2[i][k]-xb2[i])**2 for k in range(int(n2))])

    def sig3 (i,n3,xb3,xk3):
        return sum([(xk3[i][k]-xb3[i])**2 for k in range(int(n3))])

    def sig4 (i,n4,xb4,xk4):
        return sum([(xk4[i][k]-xb4[i])**2 for k in range(int(n4))])

    n_feature = len(xb)
    # print('n_feature',n_feature)
    fscores = []
    mydict = {}
    for i in range(n_feature):
        fscore_numerator = (xb1[i]-xb[i])**2 + (xb2[i]-xb[i])**2+(xb3[i]-xb[i])**2+(xb4[i]-xb[i])**2
        fscore_denominator = (1/float(n1-1))*(sig1(i,n1,xb1,xk1))+(1/float(n2-1))*(sig2(i,n2,xb2,xk2))\
                             +(1/float(n3-1))*(sig3(i,n3,xb3,xk3))+(1/float(n4-1))*(sig4(i,n4,xb4,xk4))
        #判断f_score是否为nan
        if (math.isnan(fscore_numerator/fscore_denominator)) != True:
            mydict[fscore_numerator/fscore_denominator] = i + 1
            fscores.append(fscore_numerator/fscore_denominator)
    print('the number of fscore:',np.array(fscores).shape[0])
    print('fscores finish')
    with open("./f_scores.txt", "w") as f:
        f.write(str(fscores))
    # print(mydict)
    print('dict finish')
    a = sorted(fscores, reverse=True)
    print('sorted list finish')
    with open("./a.txt", "w") as f:
        f.write(str(a))
    f_list=[]
    for i in range(np.array(a).shape[0]):
        f_list.append(mydict[a[i]])
    # print(f_list)
    print('fscore select finish')
    with open("./f_list.txt","w") as f:
        f.write(str(f_list))
    return fscores

def fscore(feature,classindex):
    '''
    feature: a matrix whose row indicates instances, col indicates features
    classindex: 1 indicates positive and 0 indicates negative
    feature：一个矩阵，其行表示实例，col表示特征
  classindex：1表示正数，0表示负数
    '''
    n_instance = len(feature)
    print(n_instance)
    n_feature  = len(feature[0])
    print(n_feature)
    n1=0
    n2=0
    n3=0
    n4=0
    for i in range(n_instance):
        if (classindex[i]) == 1:
            n1=n1+1
            # print('n1:',n1)
        elif (classindex[i]) == 2:
            n2=n2+1
            # print('n2:',n2)
        elif (classindex[i]) == 3:
            n3=n3+1
            # print('n3:',n3)
        else:
            n4=n4+1
            # print('n4:',n4)
    print(n1,n2,n3,n4)
    xb = []
    xb1=[];xb2 = [];xb3 = [];xb4 = []
    xk1=[];xk2=[];xk3=[];xk4=[]
    # # xb：整个实例的每个特征的平均值列表
    # # xbp：正实例的每个特征的平均值列表
    # # xbn：负实例的每个特征的平均值列表
    # # xkp：每个特征的列表，它是每个正实例的列表
    # # xkn：每个特征的列表，它是每个负实例的列表
    for i in range(n_feature):
        xk1_i = [];xk2_i = [];xk3_i = [];xk4_i = []
        for k in range(n_instance):
            if (classindex[k]) == 1:
                xk1_i.append(feature[k][i])
            elif (classindex[k]) == 2:
                xk2_i.append(feature[k][i])
            elif (classindex[k]) == 3:
                xk3_i.append(feature[k][i])
            else:
                xk4_i.append(feature[k][i])
        xk1.append(xk1_i)

        xk2.append(xk2_i)

        xk3.append(xk3_i)

        xk4.append(xk4_i)

        sum_xk1_i = sum(xk1_i)
        sum_xk2_i = sum(xk2_i)
        sum_xk3_i = sum(xk3_i)
        sum_xk4_i = sum(xk4_i)
        xb1.append(sum_xk1_i/float(n1))

        xb2.append(sum_xk2_i / float(n2))

        xb3.append(sum_xk3_i / float(n3))

        xb4.append(sum_xk4_i / float(n4))

        xb.append((sum_xk1_i+sum_xk2_i+sum_xk3_i+sum_xk4_i)/float(n_instance))

    print('calculate finish')



    return fscore_core(n1,n2,n3,n4,xb,xb1,xb2,xb3,xb4,xk1,xk2,xk3,xk4)


# data=pd.read_csv('D:/feature_map/fv_with_label/fv_with_label.csv',header=None)
data=pd.read_csv('./feature.csv',header=None)
feature=np.array(data)[:,1:]
classindex=np.array(data)[:,0]
fscore(feature,classindex)

new_list = [] #f.list.txt
list=[]
for i in new_list:
    print(i)
    feature=np.array(data)[:,int(i)]
    list.append(feature)

data1 = pd.DataFrame(np.transpose(list))
data1.to_csv('./fscore.csv', mode='a', header=False, index=False)