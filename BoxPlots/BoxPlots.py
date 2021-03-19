import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

data = 'Oklahoma'

logAUC = np.load('AUC/' + data + '/log_auc.npy')
logWeightAUC = np.load('AUC/' + data + '/logWeight_auc.npy')

svmLinAUC = np.load('AUC/' + data + '/SVMLin_auc.npy')
svmLinWeightAUC = np.load('AUC/' + data + '/SVMLinWeight_auc.npy')

svmRBFAUC = np.load('AUC/' + data + '/SVMRBF_auc.npy')
svmRBFWeightAUC = np.load('AUC/' + data + '/SVMRBFWeight_auc.npy')

DNN = np.load('AUC/' + data + '/DNN_auc.npy')
CSDNN = np.load('AUC/' + data + '/CSDNN_auc.npy')

AUCs = pd.DataFrame({"LR": logAUC,
          "WLR": logWeightAUC,
          "SVM-Lin": svmLinAUC,
          "WSVM-Lin": svmLinWeightAUC,
          "SVM-RBF": svmRBFAUC,
          "WSVM-RBF": svmRBFWeightAUC,
          "DNN": DNN,
          "CSDNN": CSDNN})


"""
all_arr = [logAUC,
           logWeightAUC,
           svmRBFAUC,
           svmRBFWeightAUC]
"""
g = sns.boxplot(data=AUCs, color='cornflowerblue')
g.set_title('AUC Ranges')
g.set_ylabel('AUC')
g.set_xticklabels(g.get_xticklabels(),rotation=30)
plt.show()
