import os
import pandas as pd
import numpy as np

# This code all for cleaning the various datasets, each returns a pandas dataframe, missing values intact.


def cleanBC():
    # Get data
    parent = os.path.dirname(os.getcwd())
    dataPath = os.path.join(parent, 'Data/Breastcancer/breast-cancer-wisconsin.data')

    df = pd.read_csv(dataPath, header=None)
    df.columns = ['SampleCodeNumber','ClumpThickness','UniformityOfCellSize','UniformityOfCellShape',
                    'MarinalAdhesion', 'SingleEpithelialCellSize','BareNuclei','BlandChromatin',
                    'NormalNucleoli','Mitosis','Class']

    # Replace question marks with NaN for easier imputing
    df.replace(to_replace='?', value=np.NaN, inplace=True)

    df['Class'] = np.where(df['Class'] == 2, 0, df['Class'])  # Change to 0 if 2, otherwise leave as is
    df['Class'] = np.where(df['Class'] == 4, 1, df['Class'])  # Change to 1 if 4, otherwise leave as is

    df.rename(columns={"Class": "Label"})

    return df

def cleanPima():
    # Get data
    parent = os.path.dirname(os.getcwd())
    dataPath = os.path.join(parent, 'Data/Pima/diabetes.csv')

    df = pd.read_csv(dataPath)

    df.rename(columns={"Outcome ": "Label"})

    return df

def cleanSpect():
    # Get data
    parent = os.path.dirname(os.getcwd())
    trainPath = os.path.join(parent, 'Data/Spect/SPECT.train')
    testPath = os.path.join(parent, 'Data/Spect/SPECT.test')

    train = pd.read_csv(trainPath, header=None)
    test = pd.read_csv(testPath, header=None)

    frames = [train, test]
    df = pd.concat(frames)

    df.rename(columns={0: "Label"}, inplace=True)

    return df

def cleanHepatitis():
    # Get data
    parent = os.path.dirname(os.getcwd())
    dataPath = os.path.join(parent, 'Data/Hepatitis/hepatitis.data')

    df = pd.read_csv(dataPath, header=None)
    # What do spiders have to do with hepatitis?
    df.columns = ['Label', 'Age', 'Sex', 'Steroid', 'Antivirals', 'Fatigue',
                  'Malaise', 'Anorexia', 'LiverBig', 'LiverFirm', 'SpleenPalpable',
                  'Spiders', 'Ascites', 'Varices', 'Bilirubin', 'AlkPhosphate', 'Sgot',
                  'Albumin', 'Protime', 'Histology']

    df.replace(to_replace='?', value=np.NaN, inplace=True)
    df.replace(to_replace=2, value=1, inplace=True)  # Binary variables set yes=2 for some reason
    df.replace(to_replace=1, value=0, inplace=True)  # Binary variables set no=1 for some reason


    return df

def cleanHeartDisease():
    # Get data
    parent = os.path.dirname(os.getcwd())
    dataPath = os.path.join(parent, 'Data/HeartDisease/reprocessed.hungarian.data')
    df = pd.read_csv(dataPath, header=None, delim_whitespace=True)

    df.replace(to_replace=-9.0, value=np.NaN, inplace=True) # Missing coded as -9.0
    df['num'] = np.where(df['num'].isin([1, 2, 3, 4]), 1, df['num'])  # Collapse smaller classes into one class

    df.rename(columns={"num ": "Label"})

    return df

def cleanTransfusion():
    # Get data
    parent = os.path.dirname(os.getcwd())
    dataPath = os.path.join(parent, 'Data/Transfusion/transfusion.data')

    df = pd.read_csv(dataPath)

    df.rename(columns={"whether he/she donated blood in March 2007": "Label"})

    return df



