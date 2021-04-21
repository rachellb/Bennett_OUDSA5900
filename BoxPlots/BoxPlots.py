import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

data = 'Texas'

"""
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

g = sns.boxplot(data=AUCs, color='cornflowerblue')
g.set_title('AUC Ranges')
g.set_ylabel('AUC')
g.set_xticklabels(g.get_xticklabels(),rotation=30)
plt.savefig('AUCOKBox.png', dpi=400, bbox_inches="tight")
plt.show()
"""

logAUC = np.load('Gmean/' + data + '/log_gmean.npy')
logWeightAUC = np.load('Gmean/' + data + '/logWeight_gmean.npy')

svmLinAUC = np.load('Gmean/' + data + '/SVMLin_gmean.npy')
svmLinWeightAUC = np.load('Gmean/' + data + '/SVMLinWeight_gmean.npy')

svmRBFAUC = np.load('Gmean/' + data + '/SVMRBF_gmean.npy')
svmRBFWeightAUC = np.load('Gmean/' + data + '/SVMRBFWeight_gmean.npy')

DNN = np.load('Gmean/' + data + '/DNN_gmean.npy')
CSDNN = np.load('Gmean/' + data + '/CSDNN_gmean.npy')

AUCs = pd.DataFrame({"LR": logAUC,
          "WLR": logWeightAUC,
          "SVM-Lin": svmLinAUC,
          "WSVM-Lin": svmLinWeightAUC,
          "SVM-RBF": svmRBFAUC,
          "WSVM-RBF": svmRBFWeightAUC,
          "DNN": DNN,
          "CSDNN": CSDNN})


g = sns.boxplot(data=AUCs, color='cornflowerblue')
g.set_title('Gmean Ranges')
g.set_ylabel('Gmean')
g.set_xticklabels(g.get_xticklabels(),rotation=30)
#plt.savefig('GmeanOKBox.png', dpi=400, bbox_inches="tight")
plt.show()
