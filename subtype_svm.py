from sklearn.model_selection import StratifiedKFold
import numpy as np
from scipy import interp
from sklearn.metrics import roc_curve,auc
import pandas as pd
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix
from sklearn import metrics
import warnings
import sklearn.exceptions
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import time
from collections import Counter
import warnings
from sklearn.preprocessing import label_binarize
warnings.filterwarnings("ignore")

warnings.filterwarnings("ignore", category=sklearn.exceptions.UndefinedMetricWarning)


def get_count_by_counter(l):
    t1 = time.time()
    count = Counter(l)   #类型： <class 'collections.Counter'>
    t2 = time.time()
    # print (t2-t1)
    count_dict = dict(count)   #类型： <type 'dict'>
    return count_dict

def cnf(true_value, prediction_value):
    cnf_matrix = confusion_matrix(true_value, prediction_value)

    # print(cnf_matrix)

    FP = cnf_matrix.sum(axis=0) - np.diag(cnf_matrix)
    FN = cnf_matrix.sum(axis=1) - np.diag(cnf_matrix)
    TP = np.diag(cnf_matrix)
    TN = cnf_matrix.sum() - (FP + FN + TP)
    FP = FP.astype(float)
    FN = FN.astype(float)
    TP = TP.astype(float)
    TN = TN.astype(float)

    ACC = accuracy_score(true_value, prediction_value)

    # Sensitivity, hit rate, recall, or true positive rate
    TPR = metrics.recall_score(true_value, prediction_value, average='macro')

    # Specificity or true negative rate
    TNR = np.mean(TN / (TN + FP))

    # Precision or positive predictive value
    F1_score = metrics.f1_score(true_value, prediction_value, average='macro')
    return ACC,TPR,TNR,F1_score,cnf_matrix




def auc_cal(XXX,y,g,c):


    n_class = len(np.unique(y))
    tprs = []
    aucs = []
    mean_fpr = np.linspace(0, 1, 100)
    skf = StratifiedKFold(n_splits=10)
    y = pd.Categorical(y).codes

    for train_index, validation_index in skf.split(XXX, y):
        # print("TRAIN:", train_index, "Validation:", validation_index)
        X_train, X_val = XXX[train_index], XXX[validation_index]
        Y_train, Y_val = y[train_index], y[validation_index]
        clf = SVC(kernel='rbf', gamma=g, C=c, probability=True)
        clf.fit(X_train, Y_train)
        y_one_hot = label_binarize(Y_val, np.arange(n_class))

        # 计算属于各个类别的概率，返回值的shape = [n_samples, n_classes]
        y_score = clf.predict_proba(X_val)

        # 2、手动计算micro类型的AUC
        # 首先将矩阵y_one_hot和y_score展开，然后计算假正例率FPR和真正例率TPR
        fpr, tpr, thresholds = metrics.roc_curve(y_one_hot.ravel(), y_score.ravel())
        tprs.append(interp(mean_fpr, fpr, tpr))
        tprs[-1][0] = 0.0
        # 计算auc
        roc_auc = auc(fpr, tpr)
        aucs.append(roc_auc)

    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0
    mean_auc = auc(mean_fpr, mean_tpr)

    return mean_auc


def selection(feature_path,label_path):
    feature = pd.read_csv(feature_path, header=None)
    labels = pd.read_csv(label_path, header=None)
    X = np.array(feature)
    y = np.array(labels)
    y = np.reshape(y, (71,))
    skf = StratifiedKFold(n_splits=10)

    mydict_ACC={}
    mydict_c={}
    mydict_gamma={}
    final_ACC=[]

    for k in range(1,X.shape[1]+1,1):
        print(k)
        c_value_list = [ 0.001, 0.01, 0.1, 1,  10,  100,1000]
        gamma_value_list = [1000, 100, 10, 1, 0.1, 0.01, 0.001]
        max_ACC = 0
        max_c_value = 0
        max_gamma_value = 0
        max_SPE = 0
        max_TPR = 0
        max_F1 = 0
        max_AUC=0
        for c_value in c_value_list:
            for gamma_value in gamma_value_list:
                ACC_list=[]
                SPE_list = []
                TPR_list=[]
                F1_score_list=[]
                AUC_list = []
                XX=X[:,0:k]
                for train_index, validation_index in skf.split(XX, y):
                    # print("TRAIN:", train_index, "Validation:", validation_index)
                    X_train, X_val = XX[train_index], XX[validation_index]
                    Y_train, Y_val = y[train_index], y[validation_index]


                    AUC_test = auc_cal(X_train, Y_train, gamma_value, c_value)

                    skf_1 = StratifiedKFold(n_splits=10)
                    ACC__list=[]
                    TPR_list=[]
                    SPE_list=[]
                    F1_score_list=[]
                    for t_index, v_index in skf_1.split(X_train, Y_train):
                        x_train, x_test = X_train[t_index], X_train[v_index]
                        y_train, y_test = Y_train[t_index], Y_train[v_index]
                        clf = SVC(kernel='rbf',gamma=gamma_value, C=c_value, probability=True)
                        clf.fit(x_train, y_train)
                        prediction_test = clf.predict(x_test)
                        ACC, TPR, SPE, F1_score, cnf_matrix = cnf(y_test, prediction_test)
                        ACC__list.append(ACC)
                        TPR_list.append(TPR)
                        SPE_list.append(SPE)
                        F1_score_list.append(F1_score)

                    ACC_test=np.mean(ACC__list)
                    TPR_test=np.mean(TPR_list)
                    SPE_test=np.mean(SPE_list)
                    F1_score_test=np.mean(F1_score_list)




                    ACC_list.append(ACC_test)
                    TPR_list.append(TPR_test)
                    SPE_list.append(SPE_test)
                    F1_score_list.append(F1_score_test)
                    AUC_list.append(AUC_test)


                mean_ACC=np.mean(ACC_list)
                mean_TPR=np.mean(TPR_list)
                mean_SPE=np.mean(SPE_list)
                mean_F1_score=np.mean(F1_score_list)
                mean_AUC=np.mean(AUC_list)
                if mean_ACC > max_ACC:
                    max_ACC = mean_ACC
                    max_c_value = c_value
                    max_gamma_value = gamma_value
                    max_SPE = mean_SPE
                    max_TPR = mean_TPR
                    max_F1 = mean_F1_score
                    max_AUC=mean_AUC

        print('feature number:', k, 'c:', max_c_value, 'gamma:', max_gamma_value, 'ACC:',max_ACC, 'TPR:',max_TPR,
              'SPE:', max_SPE,'F1_score:', max_F1,'AUC:',max_AUC)
        if max_ACC > 0.75:
            print('\033[1;31m')
            print('feature number:', k, 'c:', max_c_value, 'gamma:', max_gamma_value, 'ACC:', max_ACC, 'TPR:', max_TPR,
                  'SPE:', max_SPE, 'F1_score:', max_F1, 'AUC:', max_AUC)
            print('\033[0m')
        list_1 = [k, max_c_value, max_gamma_value, max_ACC, max_TPR, max_SPE, max_F1,max_AUC]

        df_to_save = pd.DataFrame(np.array(list_1).reshape(1, 8))
        df_to_save.to_csv('./subtype.csv', mode='a+', header=False)
        final_ACC.append(max_ACC)
        if max_ACC in mydict_ACC.keys():
            print("键已存在")
        else:
            mydict_ACC[max_ACC] = k
        if max_ACC in mydict_c.keys():
            print("键已存在")
        else:
            mydict_c[max_ACC] = max_c_value

        if max_ACC in mydict_gamma.keys():
            print("键已存在")
        else:
            mydict_gamma[max_ACC] = max_gamma_value

    print(mydict_ACC)
    print(mydict_ACC)
    print(mydict_gamma)

    final_k = mydict_ACC[max(final_ACC)]
    final_C = mydict_c[max(final_ACC)]
    final_gamma = mydict_gamma[max(final_ACC)]



    return final_k,final_C,final_gamma


def svm(feature_path,label_path):
    feature = pd.read_csv(feature_path, header=None)
    labels = pd.read_csv(label_path, header=None)
    X = np.array(feature)
    y = np.array(labels)
    y = np.reshape(y, (71,))
    class_number = get_count_by_counter(y)
    skf = StratifiedKFold(n_splits=10)
    k,c_value,g=selection(feature_path,label_path)




    XXX = X[:, 0:k]
    a = 0
    b = 0
    c = 0
    d = 0
    ACC = []
    SPE = []
    TPR = []
    F1_score = []

    auc_roc=auc_cal(XXX,y,g,c_value)
    for train_index, validation_index in skf.split(XXX, y):
        # print("TRAIN:", train_index, "Validation:", validation_index)
        X_train, X_val = XXX[train_index], XXX[validation_index]
        Y_train, Y_val = y[train_index], y[validation_index]
        clf = SVC(kernel='rbf', gamma=g, C=c_value, probability=True)
        clf.fit(X_train, Y_train)
        prediction_val = clf.predict(X_val)
        print(Y_val)
        print(prediction_val)
        print('----------------------------------------')
        ACC_val, TPR_val, SPE_val, F1_score_val, cnf_matrix = cnf(Y_val, prediction_val)
        # print(cnf_matrix)
        ACC.append(ACC_val)
        SPE.append(SPE_val)
        TPR.append(TPR_val)
        F1_score.append(F1_score_val)
        if cnf_matrix.shape[0] == 4:
            a = a + cnf_matrix[0][0]
            b = b + cnf_matrix[1][1]
            c = c + cnf_matrix[2][2]
            d = d + cnf_matrix[3][3]
        else:
            a = a + cnf_matrix[0][0]
            c = c + cnf_matrix[1][1]
            d = d + cnf_matrix[2][2]
    # print(a,b,c,d)
    a_ACC = round(a / class_number[1], 4)
    b_ACC = round(b / class_number[2], 4)
    c_ACC = round(c / class_number[3], 4)
    d_ACC = round(d / class_number[4], 4)

    print('feature number:', k, 'c:', c_value, 'gamma:', g, 'ACC:', np.mean(ACC), 'TPR:', np.mean(TPR),
          'SPE', np.mean(SPE), 'F1_score:', np.mean(F1_score),'AUC:',auc_roc)
    print('classical:',a_ACC,'neural:', b_ACC,'proneural:', c_ACC,'mesenchymal:', d_ACC)



feature_path = './fscore.csv'
label_path = './label.csv'
svm(feature_path, label_path)
