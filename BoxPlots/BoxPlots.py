import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

data = 'Oklahoma'

logAUC = np.load('AUC/' + data + '/log_auc.npy')
logWeightAUC = np.load('AUC/' + data + '/logWeight_auc.npy')

#self.svmLinAUC = np.load('Predictions/' + self.data + '/svmLin_pred.npy')
#self.svmLinWeightAUC = np.load('Predictions/' + self.data + '/svmLinWeight_pred.npy')

svmRBFAUC = np.load('AUC/' + data + '/SVMRBF_auc.npy')
svmRBFWeightAUC = np.load('AUC/' + data + '/SVMRBFWeight_auc.npy')

"""
self.Scores = {"LR": self.logAUC,
          "WLR": self.logWeightAUC,
          #"SVM-Lin": self.svmLin,
          #"WSVM-Lin": self.svmLinWeight,
          "SVM-RBF": self.svmRBFAUC,
          "WSVM-RBF": self.svmRBFWeightAUC}
          #"DNN": self.nn,
          #"CSDNN": self.nnWeight}

"""

all_arr = [logAUC,
           logWeightAUC,
           svmRBFAUC,
           svmRBFWeightAUC]

g = sns.boxplot(data=all_arr)
g.set(xticks=["LR", "WLR", "SVMRBF", "WSVMRBF"])
plt.show()
