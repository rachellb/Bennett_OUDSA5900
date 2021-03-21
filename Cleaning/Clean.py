import os
import pandas as pd
import numpy as np

# This code all for cleaning the various datasets, each returns a pandas dataframe, missing values intact.
from sklearn.preprocessing import OrdinalEncoder

def cleanBC():
    # Get data
    parent = os.path.dirname(os.getcwd())
    dataPath = os.path.join(parent, 'Data/Breastcancer/breast-cancer-wisconsin.data')
    df = pd.read_csv(dataPath, header=None)
    df.columns = ['SampleCodeNumber', 'ClumpThickness', 'UniformityOfCellSize', 'UniformityOfCellShape',
                    'MarinalAdhesion', 'SingleEpithelialCellSize', 'BareNuclei', 'BlandChromatin',
                    'NormalNucleoli', 'Mitosis', 'Class']

    # Replace question marks with NaN for easier imputing
    df.replace(to_replace='?', value=np.NaN, inplace=True)

    df['Class'] = np.where(df['Class'] == 2, 0, df['Class'])  # Change to 0 if 2, otherwise leave as is
    df['Class'] = np.where(df['Class'] == 4, 1, df['Class'])  # Change to 1 if 4, otherwise leave as is

    # Drop the IDs
    df.drop(columns='SampleCodeNumber', inplace=True)

    df.rename(columns={"Class": "Label"}, inplace=True)

    return df

def cleanPima():
    # Get data
    parent = os.path.dirname(os.getcwd())
    dataPath = os.path.join(parent, 'Data/Pima/diabetes.csv')

    df = pd.read_csv(dataPath)

    # Since several patients have value of 0 for variables and I assume they're not dead, set to missing
    """
    for col in df.columns:
        df[col] = np.where(df[col] == 0, np.NaN, df[col])

    # Fix Outcome and Pregnancies, which I assume are correctly labeled
    df['Outcome'] = np.where(df['Outcome'].isnull(), 0, df['Outcome'])
    df['Pregnancies'] = np.where(df['Pregnancies'].isnull(), 0, df['Pregnancies'])
    """
    df.rename(columns={"Outcome": "Label"}, inplace=True)

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
    df.replace(to_replace=1, value=0, inplace=True)  # Binary variables set no=1 for some reason
    df.replace(to_replace=2, value=1, inplace=True)  # Binary variables set yes=2 for some reason


    return df

def cleanHeartDisease():
    # Get data
    parent = os.path.dirname(os.getcwd())
    dataPath = os.path.join(parent, 'Data/HeartDisease/reprocessed.hungarian.data')
    df = pd.read_csv(dataPath, header=None, delim_whitespace=True)
    df.columns = ['Age', 'Sex', 'CP', 'Trestbps', 'chol', 'fbs', 'restecg',
                  'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal', 'num']

    df.replace(to_replace=-9.0, value=np.NaN, inplace=True) # Missing coded as -9.0
    df['num'] = np.where(df['num'].isin([1, 2, 3, 4]), 1, df['num'])  # Collapse smaller classes into one class

    df.rename(columns={"num": "Label"}, inplace=True)

    return df

def cleanTransfusion():
    # Get data
    parent = os.path.dirname(os.getcwd())
    dataPath = os.path.join(parent, 'Data/Transfusion/transfusion.data')

    df = pd.read_csv(dataPath)

    df.rename(columns={"whether he/she donated blood in March 2007": "Label"}, inplace=True)

    return df

def cleanDataTX(age):
    parent = os.path.dirname(os.getcwd())
    dataPathq1 = os.path.join(parent, r"Data/Texas_PUDF/PUDF_base1_1q2013_tab.txt")
    dataPathq2 = os.path.join(parent, r"Data/Texas_PUDF/PUDF_base1_2q2013_tab.txt")
    dataPathq3 = os.path.join(parent, r"Data/Texas_PUDF/PUDF_base1_3q2013_tab.txt")
    dataPathq4 = os.path.join(parent, r"Data/Texas_PUDF/PUDF_base1_4q2013_tab.txt")


    quarter1 = pd.read_csv('file://' + dataPathq1, delimiter="\t", usecols=['RECORD_ID',
                                                                                                 'DISCHARGE',
                                                                                                 'SOURCE_OF_ADMISSION',
                                                                                                 'PAT_STATUS',
                                                                                                 'PAT_STATE',
                                                                                                 'COUNTY',
                                                                                                 'SEX_CODE',
                                                                                                 'RACE',
                                                                                                 'ETHNICITY',
                                                                                                 'PAT_AGE',
                                                                                                 'FIRST_PAYMENT_SRC',
                                                                                                 'SECONDARY_PAYMENT_SRC',
                                                                                                 'LENGTH_OF_STAY',
                                                                                                 'ADMITTING_DIAGNOSIS',
                                                                                                 'PRINC_DIAG_CODE',
                                                                                                 'OTH_DIAG_CODE_1',
                                                                                                 'OTH_DIAG_CODE_2',
                                                                                                 'OTH_DIAG_CODE_3',
                                                                                                 'OTH_DIAG_CODE_4',
                                                                                                 'OTH_DIAG_CODE_5',
                                                                                                 'OTH_DIAG_CODE_6',
                                                                                                 'OTH_DIAG_CODE_7',
                                                                                                 'OTH_DIAG_CODE_8',
                                                                                                 'OTH_DIAG_CODE_9',
                                                                                                 'OTH_DIAG_CODE_10',
                                                                                                 'OTH_DIAG_CODE_11',
                                                                                                 'OTH_DIAG_CODE_12',
                                                                                                 'OTH_DIAG_CODE_13',
                                                                                                 'OTH_DIAG_CODE_14',
                                                                                                 'OTH_DIAG_CODE_15',
                                                                                                 'OTH_DIAG_CODE_16',
                                                                                                 'OTH_DIAG_CODE_17',
                                                                                                 'OTH_DIAG_CODE_18',
                                                                                                 'OTH_DIAG_CODE_19',
                                                                                                 'OTH_DIAG_CODE_20',
                                                                                                 'OTH_DIAG_CODE_21',
                                                                                                 'OTH_DIAG_CODE_22',
                                                                                                 'OTH_DIAG_CODE_23',
                                                                                                 'OTH_DIAG_CODE_24', ],
                           dtype={'RECORD_ID': object,
                                  'DISCHARGE': object,
                                  'PAT_STATUS': object,
                                  'PAT_STATE': str,
                                  'COUNTY': str,
                                  'SOURCE_OF_ADMISSION': object,
                                  'SEX_CODE': object,
                                  'RACE': object,
                                  'ETHNICITY': object,
                                  'PAT_AGE': object,
                                  'FIRST_PAYMENT_SRC': object,
                                  'SECONDARY_PAYMENT_SRC': object,
                                  'LENGTH_OF_STAY': float,
                                  'ADMITTING_DIAGNOSIS': str,
                                  'PRINC_DIAG_CODE': str,
                                  'OTH_DIAG_CODE_1': str,
                                  'OTH_DIAG_CODE_2': str,
                                  'OTH_DIAG_CODE_3': str,
                                  'OTH_DIAG_CODE_4': str,
                                  'OTH_DIAG_CODE_5': str,
                                  'OTH_DIAG_CODE_6': str,
                                  'OTH_DIAG_CODE_7': str,
                                  'OTH_DIAG_CODE_8': str,
                                  'OTH_DIAG_CODE_9': str,
                                  'OTH_DIAG_CODE_10': str,
                                  'OTH_DIAG_CODE_11': str,
                                  'OTH_DIAG_CODE_12': str,
                                  'OTH_DIAG_CODE_13': str,
                                  'OTH_DIAG_CODE_14': str,
                                  'OTH_DIAG_CODE_15': str,
                                  'OTH_DIAG_CODE_16': str,
                                  'OTH_DIAG_CODE_17': str,
                                  'OTH_DIAG_CODE_18': str,
                                  'OTH_DIAG_CODE_19': str,
                                  'OTH_DIAG_CODE_20': str,
                                  'OTH_DIAG_CODE_21': str,
                                  'OTH_DIAG_CODE_22': str,
                                  'OTH_DIAG_CODE_23': str,
                                  'OTH_DIAG_CODE_24': str})

    quarter2 = pd.read_csv('file://' + dataPathq2, delimiter="\t", usecols=['RECORD_ID',
                                                                                                 'DISCHARGE',
                                                                                                 'SOURCE_OF_ADMISSION',
                                                                                                 'PAT_STATUS',
                                                                                                 'PAT_STATE',
                                                                                                 'COUNTY',
                                                                                                 'SEX_CODE',
                                                                                                 'RACE',
                                                                                                 'ETHNICITY',
                                                                                                 'PAT_AGE',
                                                                                                 'FIRST_PAYMENT_SRC',
                                                                                                 'SECONDARY_PAYMENT_SRC',
                                                                                                 'LENGTH_OF_STAY',
                                                                                                 'ADMITTING_DIAGNOSIS',
                                                                                                 'PRINC_DIAG_CODE',
                                                                                                 'OTH_DIAG_CODE_1',
                                                                                                 'OTH_DIAG_CODE_2',
                                                                                                 'OTH_DIAG_CODE_3',
                                                                                                 'OTH_DIAG_CODE_4',
                                                                                                 'OTH_DIAG_CODE_5',
                                                                                                 'OTH_DIAG_CODE_6',
                                                                                                 'OTH_DIAG_CODE_7',
                                                                                                 'OTH_DIAG_CODE_8',
                                                                                                 'OTH_DIAG_CODE_9',
                                                                                                 'OTH_DIAG_CODE_10',
                                                                                                 'OTH_DIAG_CODE_11',
                                                                                                 'OTH_DIAG_CODE_12',
                                                                                                 'OTH_DIAG_CODE_13',
                                                                                                 'OTH_DIAG_CODE_14',
                                                                                                 'OTH_DIAG_CODE_15',
                                                                                                 'OTH_DIAG_CODE_16',
                                                                                                 'OTH_DIAG_CODE_17',
                                                                                                 'OTH_DIAG_CODE_18',
                                                                                                 'OTH_DIAG_CODE_19',
                                                                                                 'OTH_DIAG_CODE_20',
                                                                                                 'OTH_DIAG_CODE_21',
                                                                                                 'OTH_DIAG_CODE_22',
                                                                                                 'OTH_DIAG_CODE_23',
                                                                                                 'OTH_DIAG_CODE_24', ],
                           dtype={'RECORD_ID': object,
                                  'DISCHARGE': object,
                                  'SOURCE_OF_ADMISSION': object,
                                  'PAT_STATUS': object,
                                  'PAT_STATE': str,
                                  'COUNTY': str,
                                  'SEX_CODE': object,
                                  'RACE': object,
                                  'ETHNICITY': object,
                                  'PAT_AGE': object,
                                  'FIRST_PAYMENT_SRC': object,
                                  'SECONDARY_PAYMENT_SRC': object,
                                  'LENGTH_OF_STAY': float,
                                  'ADMITTING_DIAGNOSIS': str,
                                  'PRINC_DIAG_CODE': str,
                                  'OTH_DIAG_CODE_1': str,
                                  'OTH_DIAG_CODE_2': str,
                                  'OTH_DIAG_CODE_3': str,
                                  'OTH_DIAG_CODE_4': str,
                                  'OTH_DIAG_CODE_5': str,
                                  'OTH_DIAG_CODE_6': str,
                                  'OTH_DIAG_CODE_7': str,
                                  'OTH_DIAG_CODE_8': str,
                                  'OTH_DIAG_CODE_9': str,
                                  'OTH_DIAG_CODE_10': str,
                                  'OTH_DIAG_CODE_11': str,
                                  'OTH_DIAG_CODE_12': str,
                                  'OTH_DIAG_CODE_13': str,
                                  'OTH_DIAG_CODE_14': str,
                                  'OTH_DIAG_CODE_15': str,
                                  'OTH_DIAG_CODE_16': str,
                                  'OTH_DIAG_CODE_17': str,
                                  'OTH_DIAG_CODE_18': str,
                                  'OTH_DIAG_CODE_19': str,
                                  'OTH_DIAG_CODE_20': str,
                                  'OTH_DIAG_CODE_21': str,
                                  'OTH_DIAG_CODE_22': str,
                                  'OTH_DIAG_CODE_23': str,
                                  'OTH_DIAG_CODE_24': str})

    quarter3 = pd.read_csv('file://' + dataPathq3, delimiter="\t", usecols=['RECORD_ID',
                                                                                                 'DISCHARGE',
                                                                                                 'SOURCE_OF_ADMISSION',
                                                                                                 'PAT_STATUS',
                                                                                                 'PAT_STATE',
                                                                                                 'COUNTY',
                                                                                                 'SEX_CODE',
                                                                                                 'RACE',
                                                                                                 'ETHNICITY',
                                                                                                 'PAT_AGE',
                                                                                                 'FIRST_PAYMENT_SRC',
                                                                                                 'SECONDARY_PAYMENT_SRC',
                                                                                                 'LENGTH_OF_STAY',
                                                                                                 'ADMITTING_DIAGNOSIS',
                                                                                                 'PRINC_DIAG_CODE',
                                                                                                 'OTH_DIAG_CODE_1',
                                                                                                 'OTH_DIAG_CODE_2',
                                                                                                 'OTH_DIAG_CODE_3',
                                                                                                 'OTH_DIAG_CODE_4',
                                                                                                 'OTH_DIAG_CODE_5',
                                                                                                 'OTH_DIAG_CODE_6',
                                                                                                 'OTH_DIAG_CODE_7',
                                                                                                 'OTH_DIAG_CODE_8',
                                                                                                 'OTH_DIAG_CODE_9',
                                                                                                 'OTH_DIAG_CODE_10',
                                                                                                 'OTH_DIAG_CODE_11',
                                                                                                 'OTH_DIAG_CODE_12',
                                                                                                 'OTH_DIAG_CODE_13',
                                                                                                 'OTH_DIAG_CODE_14',
                                                                                                 'OTH_DIAG_CODE_15',
                                                                                                 'OTH_DIAG_CODE_16',
                                                                                                 'OTH_DIAG_CODE_17',
                                                                                                 'OTH_DIAG_CODE_18',
                                                                                                 'OTH_DIAG_CODE_19',
                                                                                                 'OTH_DIAG_CODE_20',
                                                                                                 'OTH_DIAG_CODE_21',
                                                                                                 'OTH_DIAG_CODE_22',
                                                                                                 'OTH_DIAG_CODE_23',
                                                                                                 'OTH_DIAG_CODE_24', ],
                           dtype={'RECORD_ID': object,
                                  'DISCHARGE': object,
                                  'SOURCE_OF_ADMISSION': object,
                                  'PAT_STATUS': object,
                                  'PAT_STATE': str,
                                  'COUNTY': str,
                                  'SEX_CODE': object,
                                  'RACE': object,
                                  'ETHNICITY': object,
                                  'PAT_AGE': object,
                                  'FIRST_PAYMENT_SRC': object,
                                  'SECONDARY_PAYMENT_SRC': object,
                                  'LENGTH_OF_STAY': float,
                                  'ADMITTING_DIAGNOSIS': str,
                                  'PRINC_DIAG_CODE': str,
                                  'OTH_DIAG_CODE_1': str,
                                  'OTH_DIAG_CODE_2': str,
                                  'OTH_DIAG_CODE_3': str,
                                  'OTH_DIAG_CODE_4': str,
                                  'OTH_DIAG_CODE_5': str,
                                  'OTH_DIAG_CODE_6': str,
                                  'OTH_DIAG_CODE_7': str,
                                  'OTH_DIAG_CODE_8': str,
                                  'OTH_DIAG_CODE_9': str,
                                  'OTH_DIAG_CODE_10': str,
                                  'OTH_DIAG_CODE_11': str,
                                  'OTH_DIAG_CODE_12': str,
                                  'OTH_DIAG_CODE_13': str,
                                  'OTH_DIAG_CODE_14': str,
                                  'OTH_DIAG_CODE_15': str,
                                  'OTH_DIAG_CODE_16': str,
                                  'OTH_DIAG_CODE_17': str,
                                  'OTH_DIAG_CODE_18': str,
                                  'OTH_DIAG_CODE_19': str,
                                  'OTH_DIAG_CODE_20': str,
                                  'OTH_DIAG_CODE_21': str,
                                  'OTH_DIAG_CODE_22': str,
                                  'OTH_DIAG_CODE_23': str,
                                  'OTH_DIAG_CODE_24': str})

    quarter4 = pd.read_csv('file://' + dataPathq4, delimiter="\t", usecols=['RECORD_ID',
                                                                            'DISCHARGE',
                                                                            'SOURCE_OF_ADMISSION',
                                                                            'PAT_STATUS',
                                                                            'PAT_STATE',
                                                                            'COUNTY',
                                                                            'SEX_CODE',
                                                                            'RACE',
                                                                            'ETHNICITY',
                                                                            'PAT_AGE',
                                                                            'FIRST_PAYMENT_SRC',
                                                                            'SECONDARY_PAYMENT_SRC',
                                                                            'LENGTH_OF_STAY',
                                                                            'ADMITTING_DIAGNOSIS',
                                                                            'PRINC_DIAG_CODE',
                                                                            'OTH_DIAG_CODE_1',
                                                                            'OTH_DIAG_CODE_2',
                                                                            'OTH_DIAG_CODE_3',
                                                                            'OTH_DIAG_CODE_4',
                                                                            'OTH_DIAG_CODE_5',
                                                                            'OTH_DIAG_CODE_6',
                                                                            'OTH_DIAG_CODE_7',
                                                                            'OTH_DIAG_CODE_8',
                                                                            'OTH_DIAG_CODE_9',
                                                                            'OTH_DIAG_CODE_10',
                                                                            'OTH_DIAG_CODE_11',
                                                                            'OTH_DIAG_CODE_12',
                                                                            'OTH_DIAG_CODE_13',
                                                                            'OTH_DIAG_CODE_14',
                                                                            'OTH_DIAG_CODE_15',
                                                                            'OTH_DIAG_CODE_16',
                                                                            'OTH_DIAG_CODE_17',
                                                                            'OTH_DIAG_CODE_18',
                                                                            'OTH_DIAG_CODE_19',
                                                                            'OTH_DIAG_CODE_20',
                                                                            'OTH_DIAG_CODE_21',
                                                                            'OTH_DIAG_CODE_22',
                                                                            'OTH_DIAG_CODE_23',
                                                                            'OTH_DIAG_CODE_24', ],
                           dtype={'RECORD_ID': object,
                                  'DISCHARGE': object,
                                  'SOURCE_OF_ADMISSION': object,
                                  'PAT_STATUS': object,
                                  'PAT_STATE': str,
                                  'COUNTY': str,
                                  'SEX_CODE': object,
                                  'RACE': object,
                                  'ETHNICITY': object,
                                  'PAT_AGE': object,
                                  'FIRST_PAYMENT_SRC': object,
                                  'SECONDARY_PAYMENT_SRC': object,
                                  'LENGTH_OF_STAY': float,
                                  'ADMITTING_DIAGNOSIS': str,
                                  'PRINC_DIAG_CODE': str,
                                  'OTH_DIAG_CODE_1': str,
                                  'OTH_DIAG_CODE_2': str,
                                  'OTH_DIAG_CODE_3': str,
                                  'OTH_DIAG_CODE_4': str,
                                  'OTH_DIAG_CODE_5': str,
                                  'OTH_DIAG_CODE_6': str,
                                  'OTH_DIAG_CODE_7': str,
                                  'OTH_DIAG_CODE_8': str,
                                  'OTH_DIAG_CODE_9': str,
                                  'OTH_DIAG_CODE_10': str,
                                  'OTH_DIAG_CODE_11': str,
                                  'OTH_DIAG_CODE_12': str,
                                  'OTH_DIAG_CODE_13': str,
                                  'OTH_DIAG_CODE_14': str,
                                  'OTH_DIAG_CODE_15': str,
                                  'OTH_DIAG_CODE_16': str,
                                  'OTH_DIAG_CODE_17': str,
                                  'OTH_DIAG_CODE_18': str,
                                  'OTH_DIAG_CODE_19': str,
                                  'OTH_DIAG_CODE_20': str,
                                  'OTH_DIAG_CODE_21': str,
                                  'OTH_DIAG_CODE_22': str,
                                  'OTH_DIAG_CODE_23': str,
                                  'OTH_DIAG_CODE_24': str})

    # Combining all the quarters into one dataframe
    frames = [quarter1, quarter2, quarter3, quarter4]
    year2013 = pd.concat(frames)

    # Insurance Codes
    medicare = ['16', 'MA', 'MB']
    medicaid = ['MC']
    sc = ['09', 'ZZ']
    other = ['10', '11', 'AM', 'CI', 'LI',
             'LM', '12', '13', '14', '15',
             'BL', 'CH', 'HM', 'OF', 'WC',
             'DS', 'VA', 'TV']

    # County Information ----------------------------------------------------------------------------

    # fips code of each county in Border Area
    on_border = ['043', '047', '061', '105',
                 '109', '127', '131', '137',
                 '141', '163', '215', '229',
                 '243', '247', '261', '271',
                 '283', '323', '311', '371',
                 '377', '385', '389', '427',
                 '435', '443', '463', '465',
                 '479', '489', '505', '507']

    year2013['On Border'] = 0
    year2013.loc[year2013['COUNTY'].isin(on_border), ['On Border']] = 1
    # If the value is missing in county, set new value to missing
    year2013['On Border'].loc[year2013['COUNTY'].isna()] = np.nan

    # Filter by hospital delivery
    # Checks if a delivery-type code (V27*) is in any of the icd9 columns
    year2013 = year2013.loc[(year2013['ADMITTING_DIAGNOSIS'].str.startswith('V27'))
                            | year2013['PRINC_DIAG_CODE'].str.startswith('V27')
                            | year2013['OTH_DIAG_CODE_1'].str.startswith('V27')
                            | year2013['OTH_DIAG_CODE_2'].str.startswith('V27')
                            | year2013['OTH_DIAG_CODE_3'].str.startswith('V27')
                            | year2013['OTH_DIAG_CODE_4'].str.startswith('V27')
                            | year2013['OTH_DIAG_CODE_5'].str.startswith('V27')
                            | year2013['OTH_DIAG_CODE_6'].str.startswith('V27')
                            | year2013['OTH_DIAG_CODE_7'].str.startswith('V27')
                            | year2013['OTH_DIAG_CODE_8'].str.startswith('V27')
                            | year2013['OTH_DIAG_CODE_9'].str.startswith('V27')
                            | year2013['OTH_DIAG_CODE_10'].str.startswith('V27')
                            | year2013['OTH_DIAG_CODE_11'].str.startswith('V27')
                            | year2013['OTH_DIAG_CODE_12'].str.startswith('V27')
                            | year2013['OTH_DIAG_CODE_13'].str.startswith('V27')
                            | year2013['OTH_DIAG_CODE_14'].str.startswith('V27')
                            | year2013['OTH_DIAG_CODE_15'].str.startswith('V27')
                            | year2013['OTH_DIAG_CODE_16'].str.startswith('V27')
                            | year2013['OTH_DIAG_CODE_17'].str.startswith('V27')
                            | year2013['OTH_DIAG_CODE_18'].str.startswith('V27')
                            | year2013['OTH_DIAG_CODE_19'].str.startswith('V27')
                            | year2013['OTH_DIAG_CODE_20'].str.startswith('V27')
                            | year2013['OTH_DIAG_CODE_21'].str.startswith('V27')
                            | year2013['OTH_DIAG_CODE_22'].str.startswith('V27')
                            | year2013['OTH_DIAG_CODE_23'].str.startswith('V27')
                            | year2013['OTH_DIAG_CODE_24'].str.startswith('V27')]

    # Drop year2012
    year2013 = year2013[year2013['DISCHARGE'] != '2012Q4']

    # Selecting appropriate age groups
    year2013 = year2013.query('PAT_AGE >= "04" & PAT_AGE <= "13"')

    if age == 'Ordinal':

        # Encode variables
        enc = OrdinalEncoder()
        year2013[["PAT_AGE"]] = enc.fit_transform(year2013[["PAT_AGE"]])

    elif age == 'Categorical':
        year2013 = age_encoderTX(year2013)

    # Drop Invalid gender rows
    year2013 = year2013[year2013['SEX_CODE'] != 'U']

    # Re-label Invalid gender rows
    year2013['SEX_CODE'] = year2013['SEX_CODE'].replace('M', 'F')

    # Replace all tick marks with nan, either impute or drop later
    year2013 = year2013.replace('`', np.NaN)

    # Replace sex code nan with F, since at this point we should have only females in the df
    year2013['SEX_CODE'] = year2013['SEX_CODE'].replace(np.NaN, 'F')

    # Changes payment sources from codes to corresponding categories
    year2013['FIRST_PAYMENT_SRC'] = year2013['FIRST_PAYMENT_SRC'].replace(medicare, "Medicare")
    year2013['FIRST_PAYMENT_SRC'] = year2013['FIRST_PAYMENT_SRC'].replace(medicaid, "Medicaid")
    year2013['FIRST_PAYMENT_SRC'] = year2013['FIRST_PAYMENT_SRC'].replace(sc,
                                                                          "Self-pay or Charity")
    year2013['FIRST_PAYMENT_SRC'] = year2013['FIRST_PAYMENT_SRC'].replace(other,
                                                                          "Other Insurance")

    year2013['SECONDARY_PAYMENT_SRC'] = year2013['SECONDARY_PAYMENT_SRC'].replace(medicare,
                                                                                  "Medicare")
    year2013['SECONDARY_PAYMENT_SRC'] = year2013['SECONDARY_PAYMENT_SRC'].replace(medicaid,
                                                                                  "Medicaid")
    year2013['SECONDARY_PAYMENT_SRC'] = year2013['SECONDARY_PAYMENT_SRC'].replace(sc,
                                                                                  "Self-pay or Charity")
    year2013['SECONDARY_PAYMENT_SRC'] = year2013['SECONDARY_PAYMENT_SRC'].replace(other,
                                                                                  "Other Insurance")

    year2013 = pd.get_dummies(year2013, dummy_na=False,
                              columns=['FIRST_PAYMENT_SRC', 'SECONDARY_PAYMENT_SRC'])

    # Create category columns
    year2013['Medicaid'] = 0
    year2013['Medicare'] = 0
    year2013['Self-pay or Charity'] = 0
    year2013['Other Insurance'] = 0

    year2013['Medicaid'] = np.where(year2013['FIRST_PAYMENT_SRC_Medicaid'] == 1, 1,
                                    year2013[
                                        'Medicaid'])  # Change to 1 if 1, otherwise leave as is
    year2013['Medicaid'] = np.where(year2013['SECONDARY_PAYMENT_SRC_Medicaid'] == 1, 1,
                                    year2013['Medicaid'])
    year2013['Medicare'] = np.where(year2013['FIRST_PAYMENT_SRC_Medicare'] == 1, 1,
                                    year2013['Medicare'])
    year2013['Medicare'] = np.where(year2013['SECONDARY_PAYMENT_SRC_Medicare'] == 1, 1,
                                    year2013['Medicare'])
    year2013['Self-pay or Charity'] = np.where(
        year2013['FIRST_PAYMENT_SRC_Self-pay or Charity'] == 1, 1,
        year2013['Self-pay or Charity'])
    year2013['Self-pay or Charity'] = np.where(
        year2013['SECONDARY_PAYMENT_SRC_Self-pay or Charity'] == 1, 1,
        year2013['Self-pay or Charity'])
    year2013['Other Insurance'] = np.where(year2013['FIRST_PAYMENT_SRC_Other Insurance'] == 1, 1,
                                           year2013['Other Insurance'])
    year2013['Other Insurance'] = np.where(year2013['SECONDARY_PAYMENT_SRC_Other Insurance'] == 1,
                                           1, year2013['Other Insurance'])

    # Drop columns with dummies
    year2013.drop(columns=['FIRST_PAYMENT_SRC_Medicaid',
                           'SECONDARY_PAYMENT_SRC_Medicaid',
                           'FIRST_PAYMENT_SRC_Medicare',
                           'SECONDARY_PAYMENT_SRC_Medicare',
                           'FIRST_PAYMENT_SRC_Self-pay or Charity',
                           'SECONDARY_PAYMENT_SRC_Self-pay or Charity',
                           'FIRST_PAYMENT_SRC_Other Insurance',
                           'SECONDARY_PAYMENT_SRC_Other Insurance']
                  , axis=1, inplace=True)

    # Rename Race columns
    """
    year2013['RACE'] = year2013['RACE'].replace('1', 'Native American')
    year2013['RACE'] = year2013['RACE'].replace('2', 'Asian or Pacific Islander')
    year2013['RACE'] = year2013['RACE'].replace('3', 'Black')
    year2013['RACE'] = year2013['RACE'].replace('4', 'White')
    year2013['RACE'] = year2013['RACE'].replace('5', 'Other Race')
    """

    # Columns for scanning ICD9 codes
    diagnosisColumns = ['ADMITTING_DIAGNOSIS',
                        'PRINC_DIAG_CODE',
                        'OTH_DIAG_CODE_1',
                        'OTH_DIAG_CODE_2',
                        'OTH_DIAG_CODE_3',
                        'OTH_DIAG_CODE_4',
                        'OTH_DIAG_CODE_5',
                        'OTH_DIAG_CODE_6',
                        'OTH_DIAG_CODE_7',
                        'OTH_DIAG_CODE_8',
                        'OTH_DIAG_CODE_9',
                        'OTH_DIAG_CODE_10',
                        'OTH_DIAG_CODE_11',
                        'OTH_DIAG_CODE_12',
                        'OTH_DIAG_CODE_13',
                        'OTH_DIAG_CODE_14',
                        'OTH_DIAG_CODE_15',
                        'OTH_DIAG_CODE_16',
                        'OTH_DIAG_CODE_17',
                        'OTH_DIAG_CODE_18',
                        'OTH_DIAG_CODE_19',
                        'OTH_DIAG_CODE_20',
                        'OTH_DIAG_CODE_21',
                        'OTH_DIAG_CODE_22',
                        'OTH_DIAG_CODE_23',
                        'OTH_DIAG_CODE_24']

    # Creating a dictionary to hold keys and values
    diseaseDictionary = {}

    diseaseDictionary['Obesity'] = ['V853', 'V854', '27800', '27801', '27803', '6491']
    # diseaseDictionary['Pregnancy resulting from assisted reproductive technology'] = ['V2385']
    diseaseDictionary['Cocaine dependence'] = ['3042', '3056']
    diseaseDictionary['Amphetamine dependence'] = ['3044', '3057']
    diseaseDictionary['Gestational diabetes mellitus'] = ['6488']
    diseaseDictionary['Pre-existing diabetes mellitus'] = ['250', '6480']
    diseaseDictionary['Anxiety'] = ['3000']
    diseaseDictionary['Anemia NOS'] = ['2859']
    diseaseDictionary['Iron deficiency anemia'] = ['280']
    diseaseDictionary['Other anemia'] = ['281']
    diseaseDictionary['Depression'] = ['311']
    diseaseDictionary['Primigravidas at the extremes of maternal age'] = ['6595', 'V2381', 'V2383']
    diseaseDictionary['Hemorrhagic disorders due to intrinsic circulating antibodies'] = ['2865']
    diseaseDictionary['Systemic lupus erythematosus'] = ['7100']
    diseaseDictionary['Lupus erythematosus'] = ['6954']
    diseaseDictionary['Autoimmune disease not elsewhere classified'] = ['27949']
    diseaseDictionary['Pure hypercholesterolemia'] = ['2720']
    diseaseDictionary['Unspecified vitamin D deficiency'] = ['2689']
    diseaseDictionary['Proteinuria'] = ['7910']
    diseaseDictionary['Tobacco use disorder'] = ['3051', '6490']
    diseaseDictionary['History of tobacco use'] = ['V1582']
    diseaseDictionary['Hypertension'] = ['401']
    diseaseDictionary['Hypertensive heart disease'] = ['402']
    diseaseDictionary['Chronic venous hypertension'] = ['4593']
    diseaseDictionary['Unspecified renal disease in pregnancy without mention of hypertension'] = ['6462']
    diseaseDictionary['Chronic kidney disease'] = ['585']
    diseaseDictionary['Hypertensive kidney disease'] = ['403']
    diseaseDictionary['Hypertensive heart and chronic kidney disease'] = ['404']
    diseaseDictionary['Renal failure not elsewhere classified'] = ['586']
    diseaseDictionary['Infections of genitourinary tract in pregnancy'] = ['6466']
    diseaseDictionary['UTI'] = ['5990']
    diseaseDictionary['Personal history of trophoblastic disease'] = ['V131']
    diseaseDictionary['Supervision of high-risk pregnancy with history of trophoblastic disease'] = ['V231']
    diseaseDictionary['Thrombophilia'] = ['28981']
    diseaseDictionary['History of premature delivery'] = ['V1321']
    diseaseDictionary['Hemorrhage in early pregnancy'] = ['640']
    diseaseDictionary[
        'Congenital abnormalities of the uterus including those complicating pregnancy, childbirth, or the puerperium'] = [
        '6540', '7522', '7523']
    # diseaseDictionary['Multiple Gestations'] = ['651']
    diseaseDictionary['Fetal Growth Restriction'] = ['764']
    diseaseDictionary['Asthma'] = ['493']
    diseaseDictionary['Obstructive Sleep Apnea'] = ['32723']
    diseaseDictionary['Other cardiovascular diseases complicating pregnancy and childbirth or the puerperium'] = [
        '6486']
    diseaseDictionary['Sickle cell disease'] = ['28260']
    diseaseDictionary['Thyroid Disease'] = ['240', '241', '242', '243', '244', '245', '246']
    diseaseDictionary['Inadequate Prenatal Care'] = ['V237']
    diseaseDictionary['Periodontal disease'] = ['523']
    diseaseDictionary['Preeclampsia/Eclampsia'] = ['6424', '6425', '6426', '6427']

    # Adds Disease column
    for disease in diseaseDictionary:
        year2013[disease] = 0  # This is how to add columns and default to 0

    # Filling out the diseases
    for disease in diseaseDictionary:
        for codes in diseaseDictionary[disease]:
            for col in diagnosisColumns:
                year2013.loc[year2013[col].str.startswith(codes, na=False), [disease]] = 1

    eclampExclude = ['64243', '64253', '64263', '64273']  # Exclude codes ending in 3
    for codes in eclampExclude:
        for col in diagnosisColumns:
            year2013.loc[
                year2013[col].str.startswith(codes, na=False), ['Preeclampsia/Eclampsia']] = 0

    # Drop columns with ICD-9 codes
    year2013.drop(columns=
                  ['ADMITTING_DIAGNOSIS', 'PRINC_DIAG_CODE', 'OTH_DIAG_CODE_1',
                   'OTH_DIAG_CODE_2', 'OTH_DIAG_CODE_3', 'OTH_DIAG_CODE_4',
                   'OTH_DIAG_CODE_5', 'OTH_DIAG_CODE_6', 'OTH_DIAG_CODE_7',
                   'OTH_DIAG_CODE_8', 'OTH_DIAG_CODE_9', 'OTH_DIAG_CODE_10',
                   'OTH_DIAG_CODE_11', 'OTH_DIAG_CODE_12', 'OTH_DIAG_CODE_13',
                   'OTH_DIAG_CODE_14', 'OTH_DIAG_CODE_15', 'OTH_DIAG_CODE_16',
                   'OTH_DIAG_CODE_17', 'OTH_DIAG_CODE_18', 'OTH_DIAG_CODE_19',
                   'OTH_DIAG_CODE_20', 'OTH_DIAG_CODE_21', 'OTH_DIAG_CODE_22',
                   'OTH_DIAG_CODE_23', 'OTH_DIAG_CODE_24'], axis=1, inplace=True)

    # Drop the columns that will not be used
    year2013.drop(
        columns=['LENGTH_OF_STAY', 'SOURCE_OF_ADMISSION', 'RECORD_ID', 'SEX_CODE', 'COUNTY', 'PAT_STATE',
                 'PAT_STATUS'], axis=1, inplace=True)

    # year2013 = (year2013.loc[(year2013['Pregnancy resulting from assisted reproductive technology'] == 0)])
    year2013 = (year2013.loc[(year2013['Multiple Gestations'] == 0)])

    # Setting dummies to true makes a column for each category that states whether or not it is missing (0 or 1).
    year2013 = pd.get_dummies(year2013, prefix_sep="__", dummy_na=True,
                              columns=['DISCHARGE', 'ETHNICITY'])

    # Propogates the missing values via the indicator columns
    year2013.loc[year2013["DISCHARGE__nan"] == 1, year2013.columns.str.startswith("DISCHARGE__")] = np.nan
    year2013.loc[year2013["ETHNICITY__nan"] == 1, year2013.columns.str.startswith("ETHNICITY__")] = np.nan

    # Drops the missingness indicator columns
    year2013 = year2013.drop(['DISCHARGE__nan'], axis=1)
    year2013 = year2013.drop(['ETHNICITY__nan'], axis=1)

    """
    year2013.rename(columns={'ETHNICITY__1': 'Hispanic', 'ETHNICITY__2': 'Non-Hispanic'},
                    inplace=True)
    """

    African_Am = year2013.loc[year2013['RACE'] == '3']
    African_Am.drop(columns=['RACE'], inplace=True)
    #African_Am.to_csv('Data/AfricanAmerican_' + date + '.csv', index=False)

    Native_Am = year2013.loc[year2013['RACE'] == '1']
    Native_Am.drop(columns=['RACE'], inplace=True)
    #Native_Am.to_csv('Data/NativeAmerican_' + date + '.csv', index=False)

    # One hot encoding race
    year2013 = pd.get_dummies(year2013, prefix_sep="__", dummy_na=True,
                              columns=['RACE'])
    year2013.loc[
        year2013["RACE__nan"] == 1, year2013.columns.str.startswith("RACE__")] = np.nan

    year2013 = year2013.drop(['RACE__nan'], axis=1)

    # Create new combined race and ethnicity columns
    year2013['White Hispanic'] = 0
    year2013['Black Hispanic'] = 0
    year2013['White Non-Hispanic'] = 0
    year2013['Black Non-Hispanic'] = 0
    year2013['Asian/Pacific Islander Hispanic'] = 0
    year2013['American Indian/Eskimo/Aleut Hispanic'] = 0
    year2013['Asian/Pacific Islander Non-Hispanic'] = 0
    year2013['American Indian/Eskimo/Aleut Non-Hispanic'] = 0
    year2013['Other Race Hispanic'] = 0
    year2013['Other Race Non-Hispanic'] = 0

    # Fill out columns with appropriate numbers
    year2013['White Hispanic'] = np.where(((year2013['RACE__4'] == 1) & (year2013['ETHNICITY__1'] == 1)), 1,
                                          year2013['White Hispanic'])
    year2013['Black Hispanic'] = np.where(((year2013['RACE__3'] == 1) & (year2013['ETHNICITY__1'] == 1)), 1,
                                          year2013['Black Hispanic'])
    year2013['Asian/Pacific Islander Hispanic'] = np.where(
        ((year2013['RACE__2'] == 1) & (year2013['ETHNICITY__1'] == 1)), 1,
        year2013['Asian/Pacific Islander Hispanic'])
    year2013['American Indian/Eskimo/Aleut Hispanic'] = np.where(
        ((year2013['RACE__1'] == 1) & (year2013['ETHNICITY__1'] == 1)), 1,
        year2013['American Indian/Eskimo/Aleut Hispanic'])
    year2013['Other Race Hispanic'] = np.where(((year2013['RACE__5'] == 1) & (year2013['ETHNICITY__1'] == 1)),
                                               1, year2013['Other Race Hispanic'])
    year2013['White Non-Hispanic'] = np.where(((year2013['RACE__4'] == 1) & (year2013['ETHNICITY__2'] == 1)), 1,
                                              year2013['White Non-Hispanic'])
    year2013['Black Non-Hispanic'] = np.where(((year2013['RACE__3'] == 1) & (year2013['ETHNICITY__2'] == 1)), 1,
                                              year2013['Black Non-Hispanic'])
    year2013['Asian/Pacific Islander Non-Hispanic'] = np.where(
        ((year2013['RACE__2'] == 1) & (year2013['ETHNICITY__2'] == 1)), 1,
        year2013['Asian/Pacific Islander Non-Hispanic'])
    year2013['American Indian/Eskimo/Aleut Non-Hispanic'] = np.where(
        ((year2013['RACE__1'] == 1) & (year2013['ETHNICITY__2'] == 1)), 1,
        year2013['American Indian/Eskimo/Aleut Non-Hispanic'])
    year2013['Other Race Non-Hispanic'] = np.where(
        ((year2013['RACE__5'] == 1) & (year2013['ETHNICITY__2'] == 1)), 1, year2013['Other Race Non-Hispanic'])

    # Drop original race and ethnicity columns
    year2013.drop(columns=['RACE__1', 'RACE__2', 'RACE__3', 'RACE__4',
                           'RACE__5', 'ETHNICITY__1', 'ETHNICITY__2'], axis=1, inplace=True)

    #year2013.to_csv('Data/year2013_' + date + '.csv', index=False)

    return year2013, African_Am, Native_Am

def cleanDataOK(dropMetro, age='Ordinal'):
        """
        ok2017 = pd.read_csv(
            r"file:///home/rachel/Documents/Preeclampsia_Research/Data/Oklahom_PUDF_2020.08.27/2017%20IP/pudf_cd.txt",
            sep=",")
        ok2018 = pd.read_csv(
            r"file:///home/rachel/Documents/Preeclampsia_Research/Data/Oklahom_PUDF_2020.08.27/2018%20IP/pudf_cdv2.txt",
            sep=",")
        """
        parent = os.path.dirname(os.getcwd())
        dataPath2017 = os.path.join(parent, r"Data/Oklahom_PUDF_2020.08.27/2017%20IP/pudf_cd.txt")
        dataPath2018 = os.path.join(parent, r"Data/Oklahom_PUDF_2020.08.27/2018%20IP/pudf_cdv2.txt")

        ok2017 = pd.read_csv('file://' + dataPath2017)
        ok2018 = pd.read_csv('file://' + dataPath2018)

        # Dropping unneeded columns
        ok2017.drop(columns=['pk_pudf', 'id_hups', 'cd_hospital_type', 'cd_admission_type_src', 'no_total_chgs',
                             'cd_drg_hci', 'cd_mdc', 'cd_ecode_cause_1',
                             'cd_ecode_cause_2', 'cd_ecode_cause_3'], inplace=True)
        ok2018.drop(columns=['pk_pudf', 'id_hups', 'cd_hospital_type', 'cd_admission_type_src', 'no_total_chgs',
                             'cd_drg_hci', 'cd_mdc', 'cd_ecode_cause_1',
                             'cd_ecode_cause_2', 'cd_ecode_cause_3'], inplace=True)

        ok2017.columns = ['State', 'Zip', 'County', 'Sex', 'Race', 'Marital_status', 'Age', 'admit_year',
                          'admit_month', 'admit_day',
                          'discharge_year', 'discharge_month', 'discharge_day', 'Length_of_stay', 'Status',
                          'Insurance', 'pdx', 'dx1', 'dx2', 'dx3',
                          'dx4', 'dx5', 'dx6', 'dx7', 'dx8', 'dx9', 'dx10', 'dx11', 'dx12', 'dx13', 'dx14', 'dx15',
                          'ppoa', 'poa1', 'poa2', 'poa3', 'poa4', 'poa5',
                          'poa6', 'poa7', 'poa8', 'poa9', 'poa10', 'poa11', 'poa12', 'poa13',
                          'poa14', 'poa15', 'ppx', 'px1', 'px2', 'px3', 'px4', 'px5', 'px6',
                          'px7', 'px8', 'px9', 'px10', 'px11', 'px12', 'px13', 'px14', 'px15']

        ok2018.columns = ['State', 'Zip', 'County', 'Sex', 'Race', 'Marital_status', 'Age', 'admit_year',
                          'admit_month', 'admit_day',
                          'discharge_year', 'discharge_month', 'discharge_day', 'Length_of_stay', 'Status',
                          'Insurance', 'pdx', 'dx1', 'dx2', 'dx3',
                          'dx4', 'dx5', 'dx6', 'dx7', 'dx8', 'dx9', 'dx10', 'dx11', 'dx12', 'dx13', 'dx14', 'dx15',
                          'ppoa', 'poa1', 'poa2', 'poa3', 'poa4', 'poa5',
                          'poa6', 'poa7', 'poa8', 'poa9', 'poa10', 'poa11', 'poa12', 'poa13',
                          'poa14', 'poa15', 'ppx', 'px1', 'px2', 'px3', 'px4', 'px5', 'px6',
                          'px7', 'px8', 'px9', 'px10', 'px11', 'px12', 'px13', 'px14', 'px15']

        ok2017 = (ok2017.loc[(ok2017['pdx'].str.startswith('Z37'))
                             | ok2017['dx1'].str.startswith('Z37')
                             | ok2017['dx2'].str.startswith('Z37')
                             | ok2017['dx3'].str.startswith('Z37')
                             | ok2017['dx4'].str.startswith('Z37')
                             | ok2017['dx5'].str.startswith('Z37')
                             | ok2017['dx6'].str.startswith('Z37')
                             | ok2017['dx8'].str.startswith('Z37')
                             | ok2017['dx9'].str.startswith('Z37')
                             | ok2017['dx10'].str.startswith('Z37')
                             | ok2017['dx11'].str.startswith('Z37')
                             | ok2017['dx12'].str.startswith('Z37')
                             | ok2017['dx13'].str.startswith('Z37')
                             | ok2017['dx14'].str.startswith('Z37')
                             | ok2017['dx15'].str.startswith('Z37')])

        ok2018 = (ok2018.loc[(ok2018['pdx'].str.startswith('Z37'))
                             | ok2018['dx1'].str.startswith('Z37')
                             | ok2018['dx2'].str.startswith('Z37')
                             | ok2018['dx3'].str.startswith('Z37')
                             | ok2018['dx4'].str.startswith('Z37')
                             | ok2018['dx5'].str.startswith('Z37')
                             | ok2018['dx6'].str.startswith('Z37')
                             | ok2018['dx8'].str.startswith('Z37')
                             | ok2018['dx9'].str.startswith('Z37')
                             | ok2018['dx10'].str.startswith('Z37')
                             | ok2018['dx11'].str.startswith('Z37')
                             | ok2018['dx12'].str.startswith('Z37')
                             | ok2018['dx13'].str.startswith('Z37')
                             | ok2018['dx14'].str.startswith('Z37')
                             | ok2018['dx15'].str.startswith('Z37')])

        # Fix missing values
        ok2017['State'] = np.where(ok2017['State'] == '99', np.NaN, ok2017['State'])
        ok2018['State'] = np.where(ok2018['State'] == '99', np.NaN, ok2018['State'])

        ok2017['Zip'] = np.where(ok2017['Zip'] == 99999.0, np.NaN, ok2017['Zip'])
        ok2018['Zip'] = np.where(ok2018['Zip'] == 99999.0, np.NaN, ok2018['Zip'])

        ok2017['Marital_status'] = np.where(ok2017['Marital_status'] == 'U', np.NaN, ok2017['Marital_status'])
        ok2018['Marital_status'] = np.where(ok2018['Marital_status'] == 'U', np.NaN, ok2018['Marital_status'])

        ok2017['Sex'] = np.where(ok2017['Sex'] == 'U', np.NaN, ok2017['Sex'])
        ok2018['Sex'] = np.where(ok2018['Sex'] == 'U', np.NaN, ok2018['Sex'])

        ok2017['Age'] = np.where(ok2017['Age'] == '99', np.NaN, ok2017['Age'])
        ok2018['Age'] = np.where(ok2018['Age'] == '99', np.NaN, ok2018['Age'])

        ok2017['Status'] = np.where(ok2017['Status'] == '99', np.NaN, ok2017['Status'])
        ok2018['Status'] = np.where(ok2018['Status'] == '99', np.NaN, ok2018['Status'])

        # Creating Insurance Binary Columns
        ok2017['Medicaid'] = 0
        ok2017['Medicare'] = 0
        ok2017['Self-pay'] = 0
        ok2017['Other Insurance'] = 0

        # Creating Insurance Binary Columns
        ok2018['Medicaid'] = 0
        ok2018['Medicare'] = 0
        ok2018['Self-pay'] = 0
        ok2018['Other Insurance'] = 0

        # Filling out appropriate Columns
        ok2017['Medicaid'] = np.where(ok2017['Insurance'] == 3, 1,
                                      ok2017['Medicaid'])  # Change to 1 if 1, otherwise leave as is
        ok2017['Medicare'] = np.where(ok2017['Insurance'] == 2, 1, ok2017['Medicare'])
        ok2017['Self-pay'] = np.where(ok2017['Insurance'] == 6, 1, ok2017['Self-pay'])
        ok2017['Other Insurance'] = np.where(ok2017['Insurance'].isin([1, 4, 5, 7]), int(1),
                                             ok2017['Other Insurance'])

        # For Missing Values, 9 is unknown in their dictionary
        ok2017['Medicaid'] = np.where(ok2017['Insurance'] == 9, np.NaN,
                                      ok2017['Medicaid'])  # Change to 1 if 1, otherwise leave as is
        ok2017['Medicare'] = np.where(ok2017['Insurance'] == 9, np.NaN, ok2017['Medicare'])
        ok2017['Self-pay'] = np.where(ok2017['Insurance'] == 9, np.NaN, ok2017['Self-pay'])
        ok2017['Other Insurance'] = np.where(ok2017['Insurance'] == 9, np.NaN, ok2017['Other Insurance'])

        # Filling out appropriate Columns
        ok2018['Medicaid'] = np.where(ok2018['Insurance'] == 3, 1,
                                      ok2018['Medicaid'])  # Change to 1 if 1, otherwise leave as is
        ok2018['Medicare'] = np.where(ok2018['Insurance'] == 2, 1, ok2018['Medicare'])
        ok2018['Self-pay'] = np.where(ok2018['Insurance'] == 6, 1, ok2018['Self-pay'])
        ok2018['Other Insurance'] = np.where(ok2018['Insurance'].isin([1, 4, 5, 7]), int(1),
                                             ok2018['Other Insurance'])

        # For Missing Values, 9 is unkown in their dictionary
        ok2018['Medicaid'] = np.where(ok2018['Insurance'] == 9, np.NaN,
                                      ok2018['Medicaid'])  # Change to 1 if 1, otherwise leave as is
        ok2018['Medicare'] = np.where(ok2018['Insurance'] == 9, np.NaN, ok2018['Medicare'])
        ok2018['Self-pay'] = np.where(ok2018['Insurance'] == 9, np.NaN, ok2018['Self-pay'])
        ok2018['Other Insurance'] = np.where(ok2018['Insurance'] == 9, np.NaN, ok2018['Other Insurance'])

        # Fixing incorrect values
        ok2018['Medicaid'] = np.where(ok2018['Insurance'].isin([11, 14]), np.NaN,
                                      ok2018['Medicaid'])  # Change to 1 if 1, otherwise leave as is
        ok2018['Medicare'] = np.where(ok2018['Insurance'].isin([11, 14]), np.NaN, ok2018['Medicare'])
        ok2018['Self-pay'] = np.where(ok2018['Insurance'].isin([11, 14]), np.NaN, ok2018['Self-pay'])
        ok2018['Other Insurance'] = np.where(ok2018['Insurance'].isin([11, 14]), np.NaN, ok2018['Other Insurance'])

        # Dropping Insurance column
        ok2017.drop(columns=['Insurance'], inplace=True)
        ok2018.drop(columns=['Insurance'], inplace=True)

        # Re-label Invalid gender rows
        ok2017['Sex'] = ok2017['Sex'].replace('M', 'F')
        ok2018['Sex'] = ok2018['Sex'].replace('M', 'F')

        # Selecting appropriate age groups
        ok2017 = ok2017.query('Age >= "01" & Age <= "50-54" | Age == "99"')
        ok2018 = ok2018.query('Age >= "01" & Age <= "50-54" | Age == "99"')

        if age == 'Ordinal':

            # Ordinal Encode Age
            enc = OrdinalEncoder()
            ok2017[["Age"]] = enc.fit_transform(ok2017[["Age"]])
            ok2018[["Age"]] = enc.fit_transform(ok2018[["Age"]])

        elif age == 'Categorical':
            ok2017 = age_encoderOK(ok2017)
            ok2018 = age_encoderOK(ok2018)

        # Re-label Race
        ok2017['Race'].replace('W', 'White', inplace=True)
        ok2017['Race'].replace('B', 'Black', inplace=True)
        ok2017['Race'].replace('I', 'Native American', inplace=True)
        ok2017['Race'].replace('O', 'Other/Unknown', inplace=True)

        ok2018['Race'].replace('W', 'White', inplace=True)
        ok2018['Race'].replace('B', 'Black', inplace=True)
        ok2018['Race'].replace('I', 'Native American', inplace=True)
        ok2018['Race'].replace('O', 'Other/Unknown', inplace=True)


        # Read in list of Counties and their designation
        urbanRural = pd.read_csv(
            r"file:///home/rachel/Documents/Preeclampsia_Research(old)/Data/County%20Metropolitan%20Classification.csv")
        urbanRural['county name'] = urbanRural['county name'].str.replace(' County', '')
        urbanRural['Metro status'] = urbanRural['Metro status'].replace('Metropolitan', 'Urban')
        urbanRural['Metro status'] = urbanRural['Metro status'].replace('Nonmetropolitan', 'Rural')
        urbanRural.drop(columns='value', inplace=True)

        # Match capitalization
        ok2017['County'] = ok2017.County.str.capitalize()
        ok2018['County'] = ok2018.County.str.capitalize()

        # Join to include whether county is urban or rural
        ok2017 = ok2017.merge(urbanRural, left_on=['County', 'State'], right_on=['county name', 'State'],
                              how='left')
        ok2018 = ok2018.merge(urbanRural, left_on=['County', 'State'], right_on=['county name', 'State'],
                              how='left')

        # Keeping admit month as proxy for whether when they developed preeclampsia
        ok2017.drop(columns=['admit_year', 'admit_day', 'discharge_month', 'discharge_year', 'discharge_day'],
                    inplace=True)
        ok2018.drop(columns=['admit_year', 'admit_day', 'discharge_month', 'discharge_year', 'discharge_day'],
                    inplace=True)

        # Re-label marriage status
        ok2017['Marital_status'] = ok2017['Marital_status'].replace('M', 'Married')
        ok2018['Marital_status'] = ok2018['Marital_status'].replace('M', 'Married')
        ok2017['Marital_status'] = ok2017['Marital_status'].replace('N', 'Unmarried')
        ok2018['Marital_status'] = ok2018['Marital_status'].replace('N', 'Unmarried')

        # A list of relevant columns
        diagnosisColumns = ['pdx', 'dx1', 'dx2', 'dx3',
                            'dx4', 'dx5', 'dx6', 'dx7',
                            'dx8', 'dx9', 'dx10', 'dx11',
                            'dx12', 'dx13', 'dx14', 'dx15']

        # Creating a dictionary to hold keys and values
        diseaseDictionary = {}

        diseaseDictionary['Obesity'] = ['E66', 'O9921', 'O9981', 'O9984', 'Z683', 'Z684', 'Z713', 'Z9884']
        diseaseDictionary['Pregnancy resulting from assisted reproductive technology'] = ['O0981']
        diseaseDictionary['Cocaine dependence'] = ['F14', 'T405']
        diseaseDictionary['Amphetamine dependence'] = ['F15', 'F19', 'P044', 'T4362']
        diseaseDictionary['Gestational diabetes mellitus'] = ['O244', 'P700']
        diseaseDictionary['Pre-existing diabetes mellitus'] = ['E10', 'E11', 'O240', 'O241', 'O243', 'O248', 'O249']
        diseaseDictionary['Anxiety'] = ['F064', 'F41']
        diseaseDictionary['Anemia NOS'] = ['D51']
        diseaseDictionary['Iron deficiency anemia'] = ['D50']
        diseaseDictionary['Other anemia'] = ['D64', 'D59', 'D489', 'D53', 'O990']
        diseaseDictionary['Depression'] = ['F32', 'F341', 'F33', 'F0631', 'Z139', 'Z1331', 'Z1332']
        diseaseDictionary['Primigravidas at the extremes of maternal age'] = ['O095', 'O096']
        diseaseDictionary['Hemorrhagic disorders due to intrinsic circulating antibodies'] = ['D683']
        diseaseDictionary['Systemic lupus erythematosus'] = ['M32']
        diseaseDictionary['Lupus erythematosus'] = ['L93', 'D6862']
        diseaseDictionary['Autoimmune disease not elsewhere classified'] = ['D89']
        diseaseDictionary['Pure hypercholesterolemia'] = ['E780']
        diseaseDictionary['Unspecified vitamin D deficiency'] = ['E55']
        diseaseDictionary['Proteinuria'] = ['D511', 'N06', 'O121', 'O122', 'R80']
        diseaseDictionary['Current Smoker'] = ['F172']
        diseaseDictionary['Hypertension'] = ['G932', 'I10', 'I14', 'I15', 'I272', 'I674', 'I973', 'O10', 'O13',
                                             'O16', 'R030']
        diseaseDictionary['Hypertensive heart disease'] = ['I11']
        diseaseDictionary['Chronic venous hypertension'] = ['I873']
        diseaseDictionary['Unspecified renal disease in pregnancy without mention of hypertension'] = ['O2683',
                                                                                                       'O9089']
        diseaseDictionary['Chronic kidney disease'] = ['D631', 'E0822', 'E0922', 'E0922', 'E1022', 'E1122', 'E1322',
                                                       'N18']
        diseaseDictionary['Hypertensive kidney disease'] = ['I12']
        diseaseDictionary['Hypertensive heart and chronic kidney disease'] = ['I13']
        diseaseDictionary['Renal failure not elsewhere classified'] = ['N19']
        diseaseDictionary['Infections of genitourinary tract in pregnancy'] = ['O23', 'O861', 'O862', 'O868']
        diseaseDictionary['UTI'] = ['O0338', 'O0388', 'O0488', 'O0788', 'O0883', 'N136', 'N390', 'N99521', 'N99531']
        diseaseDictionary['Personal history of trophoblastic disease'] = ['Z8759', 'O01']
        diseaseDictionary['Supervision of high-risk pregnancy with history of trophoblastic disease'] = ['O091']
        diseaseDictionary['Thrombophilia'] = ['D685', 'D686']
        diseaseDictionary['History of premature delivery'] = ['Z8751']
        diseaseDictionary['Hemorrhage in early pregnancy'] = ['O20']
        diseaseDictionary[
            'Congenital abnormalities of the uterus including those complicating pregnancy, childbirth, or the puerperium'] = [
            'O34', 'O340']
        diseaseDictionary['Multiple Gestations'] = ['O30']
        diseaseDictionary['Inadequate Prenatal Care'] = ['O093']
        diseaseDictionary['Periodontal disease'] = ['E08630', 'E09630', 'E10630', 'E11630', '13630', 'K05', 'K06',
                                                    'K08129']
        diseaseDictionary['Intrauterine Death'] = ['O364']
        diseaseDictionary['Preeclampsia/Eclampsia'] = ['O14', 'O15']

        # New Additions
        """
        diseaseDictionary['Edema'] = ['R609']
        diseaseDictionary['Hyperreflexia'] = ['R292']
        diseaseDictionary['Oliguria'] = ['R34']
        diseaseDictionary['Headache'] = ['R41']
        diseaseDictionary['Vomiting'] = ['R1110']
        """

        # Adds Disease column
        for disease in diseaseDictionary:
            ok2017[disease] = 0  # This is how to add columns and default to 0

            # Adds Disease column
        for disease in diseaseDictionary:
            ok2018[disease] = 0  # This is how to add columns and default to 0

        # Filling out the diseases
        for disease in diseaseDictionary:
            for codes in diseaseDictionary[disease]:
                for col in diagnosisColumns:
                    ok2017.loc[ok2017[col].str.startswith(codes, na=False), [disease]] = 1

        for disease in diseaseDictionary:
            for codes in diseaseDictionary[disease]:
                for col in diagnosisColumns:
                    ok2018.loc[ok2018[col].str.startswith(codes, na=False), [disease]] = 1

        ok2017.drop(columns=['State', 'Zip', 'Sex', 'County', 'Length_of_stay', 'Status', 'pdx',
                             'dx1', 'dx2', 'dx3', 'dx4', 'dx5', 'dx6', 'dx7',
                             'dx8', 'dx9', 'dx10', 'dx11', 'dx12', 'dx13', 'dx14',
                             'dx15', 'ppoa', 'poa1', 'poa2', 'poa3', 'poa4',
                             'poa5', 'poa6', 'poa7', 'poa8', 'poa9', 'poa10',
                             'poa11', 'poa12', 'poa13', 'poa14', 'poa15', 'ppx',
                             'px1', 'px2', 'px3', 'px4', 'px5', 'px6', 'px7',
                             'px8', 'px9', 'px10', 'px11', 'px12', 'px13', 'px14',
                             'px15', 'county name'], inplace=True)

        ok2018.drop(columns=['State', 'Zip', 'Sex', 'County', 'Length_of_stay', 'Status', 'pdx',
                             'dx1', 'dx2', 'dx3', 'dx4', 'dx5', 'dx6', 'dx7',
                             'dx8', 'dx9', 'dx10', 'dx11', 'dx12', 'dx13', 'dx14',
                             'dx15', 'ppoa', 'poa1', 'poa2', 'poa3', 'poa4',
                             'poa5', 'poa6', 'poa7', 'poa8', 'poa9', 'poa10',
                             'poa11', 'poa12', 'poa13', 'poa14', 'poa15', 'ppx',
                             'px1', 'px2', 'px3', 'px4', 'px5', 'px6', 'px7',
                             'px8', 'px9', 'px10', 'px11', 'px12', 'px13', 'px14',
                             'px15', 'county name'], inplace=True)

       # ok2017 = (ok2017.loc[(ok2017['Multiple Gestations'] == 0)])
       # ok2018 = (ok2018.loc[(ok2018['Multiple Gestations'] == 0)])

        ok2017 = (ok2017.loc[(ok2017['Pregnancy resulting from assisted reproductive technology'] == 0)])
        ok2018 = (ok2018.loc[(ok2018['Pregnancy resulting from assisted reproductive technology'] == 0)])

        # Setting dummies to true makes a column for each category that states whether or not it is missing (0 or 1).
        ok2017 = pd.get_dummies(ok2017, prefix_sep="__", dummy_na=True,
                                columns=['Race', 'Marital_status', 'Metro status'])

        # Propogates the missing values via the indicator columns
        ok2017.loc[ok2017["Race__nan"] == 1, ok2017.columns.str.startswith("Race__")] = np.nan
        ok2017.loc[ok2017["Marital_status__nan"] == 1, ok2017.columns.str.startswith("Marital_status__")] = np.nan
        ok2017.loc[ok2017["Metro status__nan"] == 1, ok2017.columns.str.startswith("Metro status__")] = np.nan

        # Drops the missingness indicator columns
        ok2017 = ok2017.drop(['Race__nan'], axis=1)
        ok2017 = ok2017.drop(['Marital_status__nan'], axis=1)
        ok2017 = ok2017.drop(['Metro status__nan'], axis=1)

        # Setting dummies to true makes a column for each category that states whether or not it is missing (0 or 1).
        ok2018 = pd.get_dummies(ok2018, prefix_sep="__", dummy_na=True,
                                columns=['Race', 'Marital_status', 'Metro status'])

        # Propogates the missing values via the indicator columns
        ok2018.loc[ok2018["Race__nan"] == 1, ok2018.columns.str.startswith("Race__")] = np.nan
        ok2018.loc[ok2018["Marital_status__nan"] == 1, ok2018.columns.str.startswith("Marital_status__")] = np.nan
        ok2018.loc[ok2018["Metro status__nan"] == 1, ok2018.columns.str.startswith("Metro status__")] = np.nan

        # Drops the missingness indicator columns
        ok2018 = ok2018.drop(['Race__nan'], axis=1)
        ok2018 = ok2018.drop(['Marital_status__nan'], axis=1)
        ok2018 = ok2018.drop(['Metro status__nan'], axis=1)

        ok2017.rename(columns={'Race__White': 'White',
                               'Race__Native American': 'Native American',
                               'Race__Black': 'Black',
                               'Race__Other/Unknown': 'Other/Unknown Race',
                               'Marital_status__Married': 'Married',
                               'Marital_status__Unmarried': 'Unmarried',
                               'Metro status__Rural': 'Rural',
                               'Metro status__Urban': 'Urban'}, inplace=True)

        ok2018.rename(columns={'Race__White': 'White',
                               'Race__Native American': 'Native American',
                               'Race__Black': 'Black',
                               'Race__Other/Unknown': 'Other/Unknown Race',
                               'Marital_status__Married': 'Married',
                               'Marital_status__Unmarried': 'Unmarried',
                               'Metro status__Rural': 'Rural',
                               'Metro status__Urban': 'Urban'}, inplace=True)

        if (dropMetro == True):
            ok2017.drop(columns=['Rural', 'Urban'], inplace=True)
            ok2018.drop(columns=['Rural', 'Urban'], inplace=True)

        # ok2017.to_csv('Data/Oklahoma_Clean/ok2017_Incomplete.csv', index=False)
        # ok2018.to_csv('Data/Oklahoma_Clean/ok2018_Incomplete.csv', index=False)

        return ok2017, ok2018

def age_encoderTX(data):
    age_map = {'04': 1, '05': 1, '06': 1,
               '07': 2, '08': 2, '09': 3,
               '10': 3, '11': 4, '12': 4, '13': 4}

    data['PAT_AGE'] = data['PAT_AGE'].map(age_map)

    data = pd.get_dummies(data, prefix_sep="__", dummy_na=True,
                          columns=['PAT_AGE'])

    data.loc[data["PAT_AGE__nan"] == 1, data.columns.str.startswith("PAT_AGE__")] = np.nan

    data = data.drop(['PAT_AGE__nan'], axis=1)

    data.rename(columns={'PAT_AGE__1.0': 'Ages 10-19',
                         'PAT_AGE__3.0': 'Ages 30-39',
                         'PAT_AGE__2.0': 'Ages 20-29',
                         'PAT_AGE__4.0': 'Ages 40+'}, inplace=True)

    return data

def age_encoderOK(data):

    age_map = {'10-14': 1, '15-19': 1, '20-24': 2,
               '25-29': 2, '30-34': 3, '35-39': 3,
               '40-44': 4, '45-49': 4, '50-54': 4}

    data['Age'] = data['Age'].map(age_map)

    data = pd.get_dummies(data, prefix_sep="__", dummy_na=True,
                            columns=['Age'])

    data.loc[data["Age__nan"] == 1, data.columns.str.startswith("Age__")] = np.nan

    data = data.drop(['Age__nan'], axis=1)

    data.rename(columns={'Age__1.0': 'Ages 10-19',
                         'Age__3.0': 'Ages 30-39',
                         'Age__2.0': 'Ages 20-29',
                         'Age__4.0': 'Ages 40+'}, inplace=True)

    return data
