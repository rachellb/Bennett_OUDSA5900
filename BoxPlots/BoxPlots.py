import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt



data = 'Oklahoma'

logSP = np.load('Results/SP/' + data + '/LRUnweight_sp.npy')
logWeightSP = np.load('Results/SP/' + data + '/LRWeight_sp.npy')

logRE = np.load('Results/RE/' + data + '/LRUnweight_re.npy')
logWeightRE = np.load('Results/RE/' + data + '/LRWeight_re.npy')

logPR = np.load('Results/PR/' + data + '/LRUnweight_pr.npy')
logWeightPR = np.load('Results/PR/' + data + '/LRWeight_pr.npy')

SVMLinSP = np.load('Results/SP/' + data + '/SVMLinUnweight_sp.npy')
SVMLinWeightSP = np.load('Results/SP/' + data + '/SVMLinWeight_sp.npy')

SVMLinRE = np.load('Results/RE/' + data + '/SVMLinUnweight_re.npy')
SVMLinWeightRE = np.load('Results/RE/' + data + '/SVMLinWeight_re.npy')

SVMLinPR = np.load('Results/PR/' + data + '/SVMLinUnweight_pr.npy')
SVMLinWeightPR = np.load('Results/PR/' + data + '/SVMLinWeight_pr.npy')

SVMRBFSP = np.load('Results/SP/' + data + '/SVMRBFUnweight_sp.npy')
SVMRBFWeightSP = np.load('Results/SP/' + data + '/SVMRBFWeight_sp.npy')

SVMRBFRE = np.load('Results/RE/' + data + '/SVMRBFUnweight_re.npy')
SVMRBFWeightRE = np.load('Results/RE/' + data + '/SVMRBFWeight_re.npy')

SVMRBFPR = np.load('Results/PR/' + data + '/SVMRBFUnweight_pr.npy')
SVMRBFWeightPR = np.load('Results/PR/' + data + '/SVMRBFWeight_pr.npy')

SPs = pd.DataFrame({"LR": logSP,
                     "WLR": logWeightSP,
                    "SVM-Lin": SVMLinSP,
                    "SVM-Lin-Weight": SVMLinWeightSP,
                    "SVM-RBF":SVMRBFSP,
                    "SVM-RBF-Weight": SVMRBFWeightSP})

REs = pd.DataFrame({"LR": logRE,
                    "WLR": logWeightRE,
                    "SVM-Lin": SVMLinRE,
                    "SVM-Lin-Weight": SVMLinWeightRE,
                    "SVM-RBF": SVMRBFRE,
                    "SVM-RBF-Weight": SVMRBFWeightRE})

PRs = pd.DataFrame({"LR": logPR,
                    "WLR": logWeightPR,
                    "SVM-Lin": SVMLinPR,
                    "SVM-Lin-Weight": SVMLinWeightPR,
                    "SVM-RBF":SVMRBFPR,
                    "SVM-RBF-Weight": SVMRBFWeightPR})


g = sns.boxplot(data=REs, color='cornflowerblue')

"""
logAUC = np.load('AUC/' + data + '/log_auc.npy')
logWeightAUC = np.load('AUC/' + data + '/logWeight_auc.npy')

svmLinAUC = np.load('AUC/' + data + '/SVMLin_auc.npy')
svmLinWeightAUC = np.load('AUC/' + data + '/SVMLinWeight_auc.npy')

svmRBFAUC = np.load('AUC/' + data + '/SVMRBF_auc.npy')
svmRBFWeightAUC = np.load('AUC/' + data + '/SVMRBFWeight_auc.npy')

DNN = np.load('AUC/' + data + '/DNN_auc.npy')
CSDNN = np.load('AUC/' + data + '/CSDNN_auc.npy')

# focal
CSDNN = [0.6549152135848999, 0.6558568477630615, 0.6550260782241821, 0.6723521947860718, 0.6599668264389038,
         0.6689961552619934, 0.6598379611968994, 0.6743229031562805, 0.66634601354599, 0.6589312553405762,
         0.6644132733345032, 0.6537855863571167, 0.6480781435966492, 0.6586068868637085, 0.6595781445503235,
         0.6671280860900879, 0.6740486025810242, 0.6761791706085205, 0.6578359603881836, 0.667838990688324,
         0.6548725366592407, 0.6508557200431824, 0.6586705446243286, 0.6785315275192261, 0.6588424444198608,
         0.6597940325737, 0.6679134368896484, 0.6714135408401489, 0.6576091051101685, 0.6693313717842102,
         0.6758384704589844, 0.6595399379730225, 0.6531185507774353, 0.6498880982398987, 0.6651637554168701,
         0.6662071347236633, 0.6589258909225464, 0.6622872352600098, 0.6761903166770935, 0.6614277362823486,
         0.6649857759475708, 0.6648510098457336, 0.6643334627151489, 0.6514975428581238, 0.6569932699203491,
         0.6657391786575317, 0.6648776531219482, 0.6670934557914734, 0.6739480495452881, 0.6568173170089722]
DNN = [0.6554155349731445, 0.656138002872467, 0.6524182558059692, 0.6645488739013672, 0.659263551235199,
       0.6658438444137573, 0.6585919260978699, 0.6724326014518738, 0.6649152636528015, 0.6572024822235107,
       0.6647369861602783, 0.6529985070228577, 0.6469423770904541, 0.6582251787185669, 0.6593493819236755,
       0.6641704440116882, 0.6744422316551208, 0.6741927862167358, 0.6571906805038452, 0.6677724123001099,
       0.6545686721801758, 0.6497185230255127, 0.6580217480659485, 0.67836594581604, 0.6554616093635559,
       0.6570390462875366, 0.6682471036911011, 0.6693865060806274, 0.657170832157135, 0.6704819202423096,
       0.6725427508354187, 0.6578121185302734, 0.6532428860664368, 0.6464844942092896, 0.6624728441238403,
       0.6649773120880127, 0.6580352783203125, 0.6612235903739929, 0.674673855304718, 0.6587862968444824,
       0.662359356880188, 0.655402421951294, 0.656591534614563, 0.6518529057502747, 0.6554012298583984,
       0.6650369763374329, 0.6652970910072327, 0.666218101978302, 0.6730774641036987, 0.65073561668396]

AUCs = pd.DataFrame({"LR": logAUC,
                     "WLR": logWeightAUC,
                     "SVM-Lin": svmLinAUC,
                     "WSVM-Lin": svmLinWeightAUC,
                     "SVM-RBF": svmRBFAUC,
                     "WSVM-RBF": svmRBFWeightAUC,
                     "DNN": DNN,
                     "CSDNN": CSDNN})

g = sns.boxplot(data=AUCs, color='cornflowerblue')
g.set_title('AUC')
g.set_ylabel('AUC')
g.set_xticklabels(g.get_xticklabels(), rotation=30)
# plt.savefig('AUCOKBox.png', dpi=400, bbox_inches="tight")
plt.show()

logGmean = np.load('Gmean/' + data + '/log_gmean.npy')
logWeightGmean = np.load('Gmean/' + data + '/logWeight_gmean.npy')

svmLinGmean = np.load('Gmean/' + data + '/SVMLin_gmean.npy')
svmLinWeightGmean = np.load('Gmean/' + data + '/SVMLinWeight_gmean.npy')

svmRBFGmean = np.load('Gmean/' + data + '/SVMRBF_gmean.npy')
svmRBFWeightGmean = np.load('Gmean/' + data + '/SVMRBFWeight_gmean.npy')

DNN = np.load('Gmean/' + data + '/DNN_gmean.npy')
CSDNN = np.load('Gmean/' + data + '/CSDNN_gmean.npy')

DNN = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.04602873089491618, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

CSDNNWCE = [0.5758381147941702, 0.5639415646049434, 0.6088173094241943, 0.5382052173788944, 0.5798785370909045, 0.5975432538174166, 0.6078820473889947, 0.5596615206646991, 0.6082210607897186, 0.5667598886743921, 0.5848188553749586, 0.60122821802609, 0.5723890076192831, 0.5918084464004107, 0.6001292982111379, 0.5662544421450245, 0.5771686970426565, 0.6027363584202626, 0.571921914623171, 0.5859849449729468, 0.6063742438642385, 0.5938134810205329, 0.5958194407632955, 0.5748880574968201, 0.5866801741395175, 0.5784947270155382, 0.5928658960579303, 0.5465242712943233, 0.5594575424771426, 0.5361327303320307, 0.5818904892303284, 0.5854826835675049, 0.5827172906406596, 0.57845164445652, 0.5855689959691683, 0.5793611969074727, 0.5774901602067416, 0.5869017959249537, 0.5716047412832146, 0.5700587846704699, 0.5733254862031508, 0.5978815927075773, 0.6109075172128191, 0.5574543210650235, 0.5754140270489261, 0.5934284285772633, 0.584405576541061, 0.5523644506324937, 0.5738741743005819, 0.5700441341395744]
# focal
CSDNN = [0.5935320095395386, 0.5951370448955005, 0.6107994437256082, 0.579474218665994, 0.5877689536087779, 0.6008145589941973, 0.604265404592895, 0.5929613652649961, 0.6015495062899089, 0.5735895128698284, 0.5830569625029942, 0.6004303246873528, 0.5864254655070615, 0.5826674674367296, 0.6076726585182353, 0.587314543891738, 0.5833087010872373, 0.5946794193303745, 0.5978650700953282, 0.5936662884084147, 0.6083749831478281, 0.6187635728121248, 0.6004996215924587, 0.5914649565389795, 0.5925626178600281, 0.5800380829445391, 0.5946000554406455, 0.5730715843686045, 0.6080259604092141, 0.5925424832829105, 0.5939151037013977, 0.5958170403970671, 0.5838488819883075, 0.5915530326382822, 0.5898917206381369, 0.589829468071763, 0.5967365098619162, 0.6090430648136996, 0.5911495230671494, 0.5855354899415952, 0.5645516917700957, 0.6089598434748509, 0.6139793988477766, 0.6009259724229422, 0.5873511082213543, 0.6016993402746187, 0.5933497910286147, 0.5838182290866906, 0.5968757810015468, 0.5925800672734791]

gmean = pd.DataFrame({"LR": logGmean,
                      "WLR": logWeightGmean,
                      "SVM-Lin": svmLinGmean,
                      "WSVM-Lin": svmLinWeightGmean,
                      "SVM-RBF": svmRBFGmean,
                      "WSVM-RBF": svmRBFWeightGmean,
                      "DNN": DNN,
                      "CSDNN (WCE)": CSDNNWCE,
                      "CSDNN (Focal)": CSDNN})

g = sns.boxplot(data=gmean, color='cornflowerblue')
g.set_title('Oklahoma 2017-2018 PUDF', fontsize=14)
g.set_ylabel('G-mean', fontsize=14)
g.set_xticklabels(g.get_xticklabels(), rotation=30)
#plt.savefig('GmeanOkBox.png', dpi=400, bbox_inches="tight")
plt.show()
"""