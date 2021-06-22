import pandas as pd
from sklearn.model_selection import train_test_split
import os
from datetime import datetime

date = datetime.today().strftime('%m%d%y')  # For labelling purposes

class dataSplitter():

    def prepData(self, data):

        self.data = pd.read_csv(data)


    def splitData(self, testSize, valSize, dataset):

        self.split1 = 5
        self.split2 = 107
        X = self.data.drop(columns='Preeclampsia/Eclampsia')
        Y = self.data['Preeclampsia/Eclampsia']

        self.X_train, self.X_test, self.Y_train, self.Y_test = train_test_split(X, Y, stratify=Y, test_size=testSize,
                                                                                random_state=self.split1)

        self.X_train, self.X_val, self.Y_train, self.Y_val = train_test_split(self.X_train, self.Y_train,
                                                                              stratify=self.Y_train, test_size=valSize,
                                                                              random_state=self.split2)

        self.X_train.to_csv('Data/' + dataset +'/' + date +'_X_Train.csv', index=False)
        self.Y_train.to_csv('Data/' + dataset +'/' + date +'_Y_Train.csv', index=False)
        self.X_test.to_csv('Data/' + dataset +'/' + date +'_X_Test.csv', index=False)
        self.Y_test.to_csv('Data/' + dataset +'/' + date +'_Y_Test.csv', index=False)
        self.X_val.to_csv('Data/' + dataset +'/' + date +'_X_Val.csv', index=False)
        self.Y_val.to_csv('Data/' + dataset +'/' + date +'_Y_Val.csv', index=False)
        
if __name__ == "__main__":



    parent = os.path.dirname(os.getcwd())

    pathOK=os.path.join(parent, 'Data/Processed/Oklahoma/Complete/Full/Outliers/Chi2_Categorical_042021.csv')
    pathTX=os.path.join(parent, 'Data/Processed/Texas/Full/Outliers/Complete/Chi2_Categorical_041521.csv')

    splitter = dataSplitter()
    splitter.prepData(data=pathOK)
    splitter.splitData(testSize=0.10, valSize=0.10, dataset='Oklahoma')

