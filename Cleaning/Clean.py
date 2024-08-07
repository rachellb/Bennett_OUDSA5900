import os
import pandas as pd
import numpy as np
from datetime import datetime

# This code all for cleaning the various datasets, each returns a pandas dataframe, missing values intact.
from sklearn.preprocessing import OrdinalEncoder

date = datetime.today().strftime('%m%d%y')  # For labelling purposes


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
    # Replacing the zero-values for Blood Pressure
    df1 = df.loc[df['Outcome'] == 1]
    df2 = df.loc[df['Outcome'] == 0]
    df1 = df1.replace({'BloodPressure': 0}, np.median(df1['BloodPressure']))
    df2 = df2.replace({'BloodPressure': 0}, np.median(df2['BloodPressure']))
    df1 = df1.replace({'BMI': 0}, np.median(df1['BMI']))
    df2 = df2.replace({'BMI': 0}, np.median(df2['BMI']))
    df1 = df1.replace({'Glucose': 0}, np.median(df1['Glucose']))
    df2 = df2.replace({'Glucose': 0}, np.median(df2['Glucose']))
    df1 = df1.replace({'SkinThickness': 0}, np.median(df1['SkinThickness']))
    df2 = df2.replace({'SkinThickness': 0}, np.median(df2['SkinThickness']))
    df1 = df1.replace({'Insulin': 0}, np.median(df1['Insulin']))
    df2 = df2.replace({'Insulin': 0}, np.median(df2['Insulin']))
    dataframe = [df1, df2]
    df = pd.concat(dataframe)

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

    df.replace(to_replace=-9.0, value=np.NaN, inplace=True)  # Missing coded as -9.0
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


def cleanDataMomi(weeks=14, system='linux'):

    # Prenatal Data
    parent = os.path.dirname(os.getcwd())
    if system == 'windows':
        dataPath = os.path.join(parent, r"Data\MOMI\Final_Prenatal_DeIdentified.xlsx")
        prenatal = pd.read_excel('file:\\' + dataPath)
    else:
        dataPath = os.path.join(parent, r"Data/MOMI/Final_Prenatal_DeIdentified.xlsx")
        prenatal = pd.read_excel('file://' + dataPath)

    print("Finished reading prenatal")

    # MOMI Data
    parent = os.path.dirname(os.getcwd())
    if system == 'windows':
        dataPath = os.path.join(parent, r"Data\MOMI\Final_MOMI_DeIdentified_Update_39Mar2021.xlsx")
        momi = pd.read_excel('file:\\' + dataPath)
    else:
        dataPath = os.path.join(parent, r"Data/MOMI/Final_MOMI_DeIdentified_Update_39Mar2021.xlsx")
        momi = pd.read_excel('file://' + dataPath)

    print("Finished reading momi")

    # Ultrasound Data
    parent = os.path.dirname(os.getcwd())
    if system == 'windows':
        dataPath = os.path.join(parent, r"Data\MOMI\Final_Ultrasound_DeIdentified.xlsx")
        ultrasound = pd.read_excel('file:\\' + dataPath)
    else:
        dataPath = os.path.join(parent, r"Data/MOMI/Final_Ultrasound_DeIdentified.xlsx")
        ultrasound = pd.read_excel('file://' + dataPath)

    print("Finished Reading all data")

    momi['MIDBV'] = np.where(momi['MIDBV'] == 99, np.NaN, momi['MIDBV'])
    momi['MIDCHLAM'] = np.where(momi['MIDCHLAM'] == 99, np.NaN, momi['MIDCHLAM'])
    momi['MIDCONDY'] = np.where(momi['MIDCONDY'] == 99, np.NaN, momi['MIDCONDY'])
    momi['MIDGC'] = np.where(momi['MIDGC'] == 99, np.NaN, momi['MIDGC'])
    momi['MIDHEPB'] = np.where(momi['MIDHEPB'] == 99, np.NaN, momi['MIDHEPB'])
    momi['MIDTRICH'] = np.where(momi['MIDTRICH'] == 99, np.NaN, momi['MIDTRICH'])
    momi['MIDGBS'] = np.where(momi['MIDGBS'] == 99, np.NaN, momi['MIDGBS'])
    momi['MHXPARA'] = np.where(momi['MHXPARA'] == 99, np.NaN, momi['MHXPARA'])
    momi['MHXABORT'] = np.where(momi['MHXABORT'] == 99, np.NaN, momi['MHXABORT'])
    momi['MHXGRAV'] = np.where(momi['MHXGRAV'] == 99, np.NaN, momi['MHXGRAV'])
    momi['MomEducation_State'] = np.where(momi['MomEducation_State'] == 'Unknown', np.NaN, momi['MomEducation_State'])
    momi['DadEducation_State'] = np.where(momi['DadEducation_State'] == 'Unknown', np.NaN, momi['DadEducation_State'])
    momi['Smoke_b'] = np.where(momi['Smoke_b'] == 'Unknown (unable to assess)', np.NaN, momi['Smoke_b'])
    momi['Race'] = np.where(momi['Race'].isin(['9', 'A']), np.NaN, momi['Race'])
    momi['Ethnicity'] = np.where(momi['Ethnicity'].isin(['UNSPECIFIED']), np.NaN, momi['Ethnicity'])
    momi['InfSex'] = np.where(momi['InfSex'] == 'U', np.NaN, momi['InfSex'])
    momi['InfSex'] = np.where(momi['InfSex'] == 'f', 'F', momi['InfSex'])

    # Dropping erroneous prenatal data. This data does not actually exist, is thousands of missing values
    prenatal.drop(prenatal[prenatal['DELWKSGT'].isnull()].index, inplace=True)
    prenatal.drop(prenatal[prenatal['PNV_Total_Number'].isnull()].index, inplace=True)

    """
    insuranceMap = {1: 'MedicalAssistance',
                    2: 'PrivateInsurance',
                    3: 'Self-pay'}

    momi['DFC'] = momi['DFC'].map(insuranceMap)

    outcomeMap = {1: 'SingleStillborn',
                  2: 'TwinsLiveborn',
                  3: 'TwinsOneLive',
                  4: 'TwinsStillborn',
                  5: 'MultsLiveborn',
                  6: 'OtherMultSomeLive',
                  7: 'OtherMultStillborn',
                  9: np.NaN,
                  10: 'SingleLiveborn'}

    momi['MMULGSTD'] = momi['MMULGSTD'].map(outcomeMap)

    neurMuscDiseaseMap = {0: 'None',
                          1: 'Multiple Sclerosis',
                          2: 'Cerebal Palsy',
                          3: 'Myotonic Dystrophy',
                          4: 'Myasthenia Gravis',
                          5: 'Other'}

    momi['MCNSMUSC'] = momi['MCNSMUSC'].map(neurMuscDiseaseMap)

    collagenVascMap = {0: 'None',
                       1: 'Rhematoid Arthritis',
                       2: 'Lupus',
                       8: 'Multiple Diagnostic Codes'}

    momi['MCOLVASC'] = momi['MCOLVASC'].map(collagenVascMap)

    struHeartMap = {0: 'None',
                    1: 'Congenital Heart Disease',
                    2: 'Rheumatic Heart Disease',
                    3: 'Myocarditis/Cardiomyopathy',
                    4: 'ValveDisorder',
                    5: 'ArtificialValves',
                    9: 'Other'}

    momi['MCVDANAT'] = momi['MCVDANAT'].map(struHeartMap)

    postpartMap = {0: 'None',
                   1: 'Endometritis',
                   2: 'UrinaryTractInfection',
                   3: 'Hemmorrage',
                   4: 'WoundInfection',
                   5: 'Disseminated',
                   6: 'Obstruction',
                   9: 'Other'}

    momi['MDELCOMP'] = momi['MDELCOMP'].map(postpartMap)

    diabetesMap = {0: 'None',
                   1: 'GestationalDiabetes',
                   2: 'TypeI',
                   3: 'TypeII',
                   4: 'UnspecifiedPriorDiabetes'}

    momi['MENDDIAB'] = momi['MENDDIAB'].map(diabetesMap)

    thyroidMap = {0: 'None',
                  1: 'Hyperthyroid',
                  2: 'Hypothyroid',
                  9: 'Other'}

    momi['MENDTHY'] = momi['MENDTHY'].map(thyroidMap)

    liverGallMap = {0: 'None',
                    1: 'HepA',
                    2: 'HepB',
                    3: 'HepC',
                    4: 'HepD',
                    5: 'HepE',
                    6: 'LiverTransplant',
                    7: 'Cholelithiasis',
                    8: 'Pancreatitis',
                    9: 'Other'}

    momi['MGILGBP'] = momi['MGILGBP'].map(liverGallMap)

    kidneyMap = {0: 'None',
                 1: 'Glomerulonephritis',
                 2: 'Pyelonephritis;',
                 3: 'LupusNephritis',
                 4: 'NephroticSyndrome',
                 5: 'Nephrolithiasis',
                 6: 'Transplant;',
                 7: 'RenalAbscess',
                 8: 'MultipleDiagnosticCodes',
                 9: 'Other'}

    momi['MGURENAL'] = momi['MGURENAL'].map(kidneyMap)

    anemiaMap = {0: 'None',
                 1: 'IronDeficiencyAnemia',
                 2: 'B12DeficiencyAnemia',
                 3: 'FolateDeficiencyAnemia',
                 9: 'UnspecifiedAnemia'}

    momi['MHEMANEM'] = momi['MHEMANEM'].map(anemiaMap)

    hemoGlob = {0: 'None',
                1: 'Hgb-SS',
                2: 'Hgb-SC',
                3: 'Hgb-Sthal',
                4: 'AlphaThalassemia',
                5: 'BetaThalassemia',
                6: 'SickleCellTrait',
                9: 'Other'}

    momi['MHEMHGB'] = momi['MHEMHGB'].map(hemoGlob)

    thromMap = {0: 'None',
                1: 'Gestational',
                2: 'DisseminatedIntravascularCoagulation',
                3: 'MultipleDiagnosticCodes',
                9: 'Other'}

    momi['MHEMPLT'] = momi['MHEMPLT'].map(thromMap)

    viralMap = {0: 'None',
                1: 'PrimaryCMV',
                2: 'ParovirusB19',
                3: 'Rubella',
                4: 'Toxoplasma',
                5: np.NaN,
                8: 'MultipleDiagnosticCodes',
                9: 'Other'}

    momi['MIDVIRPR'] = momi['MIDVIRPR'].map(viralMap)

    substanceMap = {0: 'None',
                    1: 'Stimulants',
                    2: 'Sedatives/Hypnotics/Anxiolytics',
                    3: 'Anti-depressants/OtherPsychoactives',
                    4: 'Hallucinogens',
                    6: 'Alcohol',
                    8: 'MultipleDiagnosticCodes',
                    9: 'Other'}

    momi['MTOXOTHR'] = momi['MTOXOTHR'].map(substanceMap)

    anoAnoMap = {0: 'None',
                 1: 'Anencephaly/Similar',
                 2: 'Encephalocele',
                 3: 'Microcephaly',
                 4: 'CongenitalHydrocephalus',
                 5: 'SpinaBifida',
                 8: 'MultipleDiagnosticCodes',
                 0: 'OtherCongenital'}

    momi['ICNSANAT'] = momi['ICNSANAT'].map(anoAnoMap)
    """

    # Ordinal Encoding Education
    education_map = {'8th grade or less': 1,
                     '9th-12th grade, no diploma': 2,
                     'High school graduate or GED completed': 3,
                     'Some college credit, no degree': 4,
                     'Associate degree': 5,
                     "Bachelor's degree": 6,
                     "Master's degree": 7,
                     'Doctorate or professional degree': 8,
                     'Doctorate or Professional degree': 8}

    momi['DadEducation_State'] = momi['DadEducation_State'].map(education_map)

    momi['MomEducation_State'] = momi['MomEducation_State'].map(education_map)

    # Renaming Race variables for easier comparison
    raceMap = {'B': 'AfricanAmerican', 'C': "Chinese", 'D': "Declined",
               'E': "OtherAsian", 'F': "Filipino", 'G': "Guam/Chamorro",
               'I': "Indian(Asian)", 'J': "Japanese", 'K': "Korean",
               'L': "AlaskanNative", 'N': "NativeAmerican", 'P': "OtherPacificIslander",
               'Q': "Hawaiian", 'S': "Samoan", 'V': "Vietnamese", 'W': "White", 'D': "Declined", 9: np.NaN}

    momi['Race'] = momi['Race'].map(raceMap)

    # Collapsing Race categories
    momi['RaceCollapsed'] = np.NaN

    AsianGroups = ['OtherAsian', 'Indian(Asian)', 'Chinese', 'Korean', 'Filipino', 'Japanese', 'Vietnamese']
    Polynesian = ['Hawaiian', 'Samoan', 'OtherPacificIslander', 'Guam/Chamorro']  # Unsure about Guam
    NativeGroups = ['NativeAmerican', 'AlaskanNative']

    # Asian
    momi['RaceCollapsed'] = np.where((momi['Race'].isin(AsianGroups)), 'Asian', momi['RaceCollapsed'])
    # Polynesian
    momi['RaceCollapsed'] = np.where((momi['Race'].isin(Polynesian)), 'Polynesian', momi['RaceCollapsed'])
    # Native
    momi['RaceCollapsed'] = np.where((momi['Race'].isin(NativeGroups)), 'Native', momi['RaceCollapsed'])
    # African
    momi['RaceCollapsed'] = np.where((momi['Race'] == 'AfricanAmerican'), 'African', momi['RaceCollapsed'])
    # White
    momi['RaceCollapsed'] = np.where((momi['Race'] == 'White'), 'White', momi['RaceCollapsed'])

    # Renaming Hypertensive variables for easier comparison
    hypMap = {0: 'None', 1: 'TransientHypertension',
              2: 'Preeclampsia mild', 3: 'PreeclampsiaSevere',
              5: 'Eclampsia', 6: 'ChronicHypwPre',
              8: 'MultipleDiagnosticCodes', 9: 'UnspecifiedHyp'}

    momi['MOBHTN'] = momi['MOBHTN'].map(hypMap)

    # Set mildpe to 0 if marked severe
    momi['Mild_PE'] = np.where(momi['MOBHTN'] == 'PreeclampsiaSevere', 0, momi['Mild_PE'])

    # Looking at any occurance of Preeclampsia/Eclampsia
    momi['Preeclampsia/Eclampsia'] = np.NaN
    momi['Preeclampsia/Eclampsia'] = np.where(
        (momi['Mild_PE'] == 1) | (momi['Severe_PE'] == 1) | (momi['SIPE'] == 1) | (momi['MOBHTN'] == 'Eclampsia'), 1, 0)

    # Renaming columns for easier analysis
    momi.rename(columns={"DMOMAGE": "MotherAge", "FatherAge_State": "FatherAge", "DFC": "Insurance",
                         "DELWKSGT": "GestAgeDelivery", "MHXGRAV": "TotalNumPregnancies",
                         "MHXPARA": "DeliveriesPriorAdmission",
                         "MHXABORT": "TotalAbortions", "PRIMIP": "Primagrivada", "DMOMHGT": "MaternalHeightMeters",
                         "MOBRPWT": "PrePregWeight", "MOBADMWT": "WeightAtAdmission",
                         "FOBLABHR": "HoursLaborToDelivery",
                         "FOBROMHR": "HoursMembraneReptureDelivery", "CSREPEAT": "RepeatCesarean",
                         "FDELTYPE": "DeliveryMethod",
                         "MMULGSTD": "OutcomeOfDelivery", "FOBDEATH": "FetalDeath",
                         "MCNSMUSC": "MaternalNeuromuscularDisease",
                         "MCOLVASC": "MCollagenVascularDisease", "MCVDANAT": "MStructuralHeartDiseas",
                         "MCVDHTN": "ChronicHypertension",
                         "MOBHTN": "PregRelatedHypertension", "MDELCOMP": "MPostPartumComplications",
                         "MDEPRESS": "Depression",
                         "MENDDIAB": "DiabetesMellitus", "MENDTHY": "ThyroidDisease",
                         "MGIHYPER": "HyperemesisGravidarum",
                         "MGILGBP": "MLiverGallPanc", "MGUINFER": "HistoryInfertility", "MGURENAL": "KidneyDisease",
                         "MHEARTOPER": "OperationsOnHeartandPericardium", "MHEMANEM": "MAnemiaWOHemoglobinopathy",
                         "MHEMHGB": "MHemoglobinopathy", "MHEMPLT": "Thrombocytopenia", "MHEMTRAN": "TransfusionOfPRBC",
                         "MIDBV": "BacterialVaginosis", "MIDCHLAM": "Chlamydia", "MIDCONDY": "Condylomata",
                         "MIDGBS": "GroupBStrep", "MIDGC": "GonococcalInfection", "MIDHEPB": "HepBInfection",
                         "MIDHSV": "Herpes", "MIDTB": "Tuberculosis", "MIDTRICH": "Trichomonas",
                         "MIDVIRPR": "ViralOrProtoInf",
                         "MINTERINJ": "ThoraxAbPelvInjuries", "MMORTECLAMP": "Eclampsia",
                         "MMORTHEARTFAIL": "HeartFailure",
                         "MMORTRENAL": "AcuteRenalFailure", "MMORTSICKLECELL": "SickleCell",
                         "MOBPRECS": "PreviousCesarean",
                         "MPULASTH": "Asthma", "MTOXCOC": "Cocaine", "MTOXNARC": "Opioid",
                         "MTOXOTHR": "OtherSubstanceAbuse",
                         "MTOXTHC": "Marijuana", "IDEMBWT": "InfantWeightGrams", "IGROWTH": "GestWeightCompare",
                         "ICNSANAT": "CNSAbnormality", "IIDSYPH": "CongenitalSyphilis", "IIDUTI": "UTI",
                         "Alcohol_a": 'Drinks/Week'}, inplace=True)

    # Dropping variables with more than 20% missing values
    momi = momi.loc[:, momi.isnull().mean() < .20]

    # Joining the momi data with the prenatal data - we want women who never had preeclampsia and first incidence of
    # preeclampsia, nothing else
    # Step 1, split systolic and diastolic
    new = prenatal["PNV_BP"].str.split("/", n=1, expand=True)
    prenatal["Systolic"] = new[0]
    prenatal["Diastolic"] = new[1]
    prenatal[["Systolic", "Diastolic"]] = prenatal[["Systolic", "Diastolic"]].apply(pd.to_numeric)

    # MAP = (Sys + (2*Dias))/3
    prenatal['MAP'] = np.NaN
    prenatal['MAP'] = (prenatal['Systolic'] + (2 * prenatal['Diastolic'])) / 3

    # Step 2, make indicator variable
    prenatal['High'] = np.where((prenatal['Systolic'] >= 130) | (prenatal['Diastolic'] >= 80), 1, 0)

    # Step 3, make a cumulative sum to count how many times this person has had spikes
    prenatal['Prev_highBP'] = prenatal.groupby(['MOMI_ID', 'Delivery_Number_Per_Mother'])['High'].cumsum().astype(int)

    # Drop all women under 14 weeks from prenatal data
    prenatal.drop(prenatal.loc[prenatal['PNV_GestAge'] > weeks].index, inplace=True)
    momi.sort_values('MOMI_ID', inplace=True)
    uniquePregMomi = momi.drop_duplicates(subset=['MOMI_ID', 'Delivery_Number_Per_Mother'], keep='last')

    prenatal.sort_values('PNV_GestAge', ascending=False, inplace=True)  # For preferenceing high bp
    uniquePregPrenatal = prenatal.drop_duplicates(subset=['MOMI_ID', 'Delivery_Number_Per_Mother'], keep='first')
    join = pd.merge(uniquePregMomi, uniquePregPrenatal, how='right')

    # Removes duplicates, keeping only instances with Preeclampsia
    join.sort_values('Preeclampsia/Eclampsia', ascending=False, inplace=True)
    join = join.drop_duplicates(subset=['MOMI_ID'], keep='first')

    # Dropping variables we won't be using
    join.drop(columns=['MOMI_ID', 'Delivery_Number_Per_Mother', 'InfantWeightGrams', 'Eclampsia',
                       'GestAgeDelivery', 'DeliveryMethod', 'FetalDeath', 'OutcomeOfDelivery', 'DeliveryMethod',
                       'PregRelatedHypertension', 'Mild_PE', 'Severe_PE', 'SIPE', 'High', 'PNV_BP', 'Has_Prenatal_Data',
                       'Has_Ultrasound_PlacLoc', 'NICULOS', 'InfantWeightGrams', 'GestWeightCompare',
                       'DELWKSGT', 'MMULGSTD', 'Systolic', 'Diastolic', 'Race',
                       'PNV_Total_Number', 'MPostPartumComplications', 'Thrombocytopenia', 'TransfusionOfPRBC',
                       'AcuteRenalFailure'], inplace=True)

    # Renaming Hypertensive variables for easier comparison
    hypMap = {'African': 0, 'Asian': 1,
              'Native': 2, 'Polynesian': 3,
              'White': 4}

    join['RaceCollapsed'] = join['RaceCollapsed'].map(hypMap)

    hypMap = {'F': 0, 'M': 1}
    join['InfSex'] = join['InfSex'].map(hypMap)

    joinAfrican = join[join['RaceCollapsed'] == 0]
    joinAfrican.drop(columns=['RaceCollapsed'], inplace=True)
    #joinAfrican.to_csv('momiEncodedAfrican_111921.csv', index=False)


    return join


def cleanDataTX(age):
    parent = os.path.dirname(os.getcwd())
    dataPathq1 = os.path.join(parent, r"Data/Texas_PUDF/PUDF_base1_1q2013_tab.txt")
    dataPathq2 = os.path.join(parent, r"Data/Texas_PUDF/PUDF_base1_2q2013_tab.txt")
    dataPathq3 = os.path.join(parent, r"Data/Texas_PUDF/PUDF_base1_3q2013_tab.txt")
    dataPathq4 = os.path.join(parent, r"Data/Texas_PUDF/PUDF_base1_4q2013_tab.txt")

    cols = ['RECORD_ID',
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
            'OTH_DIAG_CODE_24', ]
    dtype = {'RECORD_ID': object,
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
             'OTH_DIAG_CODE_24': str}

    quarter1 = pd.read_csv('file://' + dataPathq1, delimiter="\t", usecols=cols,
                           dtype=dtype)

    quarter2 = pd.read_csv('file://' + dataPathq2, delimiter="\t", usecols=cols,
                           dtype=dtype)

    quarter3 = pd.read_csv('file://' + dataPathq3, delimiter="\t", usecols=cols,
                           dtype=dtype)

    quarter4 = pd.read_csv('file://' + dataPathq4, delimiter="\t", usecols=cols,
                           dtype=dtype)


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
    year2013['FIRST_PAYMENT_SRC'] = year2013['FIRST_PAYMENT_SRC'].replace(sc, "Self-pay or Charity")
    year2013['FIRST_PAYMENT_SRC'] = year2013['FIRST_PAYMENT_SRC'].replace(other, "Other Insurance")

    year2013['SECONDARY_PAYMENT_SRC'] = year2013['SECONDARY_PAYMENT_SRC'].replace(medicare, "Medicare")
    year2013['SECONDARY_PAYMENT_SRC'] = year2013['SECONDARY_PAYMENT_SRC'].replace(medicaid, "Medicaid")
    year2013['SECONDARY_PAYMENT_SRC'] = year2013['SECONDARY_PAYMENT_SRC'].replace(sc, "Self-pay or Charity")
    year2013['SECONDARY_PAYMENT_SRC'] = year2013['SECONDARY_PAYMENT_SRC'].replace(other, "Other Insurance")

    # Setting dummies to true makes a column for each category that states whether or not it is missing (0 or 1).
    year2013 = pd.get_dummies(year2013, prefix_sep="__", dummy_na=True,
                              columns=['FIRST_PAYMENT_SRC', 'SECONDARY_PAYMENT_SRC'])

    # Propogates the missing values via the indicator columns
    year2013.loc[
        year2013["FIRST_PAYMENT_SRC__nan"] == 1, year2013.columns.str.startswith("FIRST_PAYMENT_SRC__")] = np.nan
    year2013.loc[year2013["SECONDARY_PAYMENT_SRC__nan"] == 1, year2013.columns.str.startswith(
        "SECONDARY_PAYMENT_SRC__")] = np.nan

    # Create category columns
    year2013['Medicaid'] = 0
    year2013['Medicare'] = 0
    year2013['Self-pay or Charity'] = 0
    year2013['Other Insurance'] = 0

    year2013['Medicaid'] = np.where(year2013['FIRST_PAYMENT_SRC__Medicaid'] == 1, 1,
                                    year2013[
                                        'Medicaid'])  # Change to 1 if 1, otherwise leave as is
    year2013['Medicaid'] = np.where(year2013['SECONDARY_PAYMENT_SRC__Medicaid'] == 1, 1,
                                    year2013['Medicaid'])
    year2013['Medicare'] = np.where(year2013['FIRST_PAYMENT_SRC__Medicare'] == 1, 1,
                                    year2013['Medicare'])
    year2013['Medicare'] = np.where(year2013['SECONDARY_PAYMENT_SRC__Medicare'] == 1, 1,
                                    year2013['Medicare'])
    year2013['Self-pay or Charity'] = np.where(year2013['FIRST_PAYMENT_SRC__Self-pay or Charity'] == 1, 1,
                                               year2013['Self-pay or Charity'])
    year2013['Self-pay or Charity'] = np.where(
        year2013['SECONDARY_PAYMENT_SRC__Self-pay or Charity'] == 1, 1,
        year2013['Self-pay or Charity'])
    year2013['Other Insurance'] = np.where(year2013['FIRST_PAYMENT_SRC__Other Insurance'] == 1, 1,
                                           year2013['Other Insurance'])
    year2013['Other Insurance'] = np.where(year2013['SECONDARY_PAYMENT_SRC__Other Insurance'] == 1,
                                           1, year2013['Other Insurance'])

    year2013['Medicaid'] = np.where(
        ((year2013['FIRST_PAYMENT_SRC__nan'].isnull()) & (year2013['SECONDARY_PAYMENT_SRC__nan'].isnull())), np.NaN,
        year2013['Medicaid'])
    year2013['Medicare'] = np.where(
        ((year2013['FIRST_PAYMENT_SRC__nan'].isnull()) & (year2013['SECONDARY_PAYMENT_SRC__nan'].isnull())), np.NaN,
        year2013['Medicare'])
    year2013['Self-pay or Charity'] = np.where(
        ((year2013['FIRST_PAYMENT_SRC__nan'].isnull()) & (year2013['SECONDARY_PAYMENT_SRC__nan'].isnull())), np.NaN,
        year2013['Self-pay or Charity'])
    year2013['Other Insurance'] = np.where(
        ((year2013['FIRST_PAYMENT_SRC__nan'].isnull()) & (year2013['SECONDARY_PAYMENT_SRC__nan'].isnull())), np.NaN,
        year2013['Other Insurance'])

    # Drop columns with dummies
    year2013.drop(columns=['FIRST_PAYMENT_SRC__Medicaid',
                           'SECONDARY_PAYMENT_SRC__Medicaid',
                           'FIRST_PAYMENT_SRC__Medicare',
                           'SECONDARY_PAYMENT_SRC__Medicare',
                           'FIRST_PAYMENT_SRC__Self-pay or Charity',
                           'SECONDARY_PAYMENT_SRC__Self-pay or Charity',
                           'FIRST_PAYMENT_SRC__Other Insurance',
                           'SECONDARY_PAYMENT_SRC__Other Insurance',
                           'FIRST_PAYMENT_SRC__nan',
                           'SECONDARY_PAYMENT_SRC__nan']
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
    diseaseDictionary['Pregnancy resulting from assisted reproductive technology'] = ['V2385']
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
    diseaseDictionary['Multiple Gestations'] = ['651']
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
        columns=['LENGTH_OF_STAY', 'SOURCE_OF_ADMISSION', 'RECORD_ID', 'PAT_STATE', 'SEX_CODE', 'COUNTY',
                 'PAT_STATUS'], axis=1, inplace=True)

    # year2013.to_csv('year2013_' + date + '.csv', index=False)

    # year2013 = (year2013.loc[(year2013['Pregnancy resulting from assisted reproductive technology'] == 0)])
    # year2013 = (year2013.loc[(year2013['Multiple Gestations'] == 0)])

    # Setting dummies to true makes a column for each category that states whether or not it is missing (0 or 1).
    year2013 = pd.get_dummies(year2013, prefix_sep="__", dummy_na=True,
                              columns=['DISCHARGE', 'ETHNICITY'])

    # Propogates the missing values via the indicator columns
    year2013.loc[year2013["DISCHARGE__nan"] == 1, year2013.columns.str.startswith("DISCHARGE__")] = np.nan
    year2013.loc[year2013["ETHNICITY__nan"] == 1, year2013.columns.str.startswith("ETHNICITY__")] = np.nan

    # Drops the missingness indicator columns
    year2013 = year2013.drop(['DISCHARGE__nan'], axis=1)
    year2013 = year2013.drop(['ETHNICITY__nan'], axis=1)

    year2013.rename(columns={'DISCHARGE__1': 'Discharge in Quarter 1',
                             'DISCHARGE__2': 'Discharge in Quarter 2',
                             'DISCHARGE__3': 'Discharge in Quarter 3',
                             'DISCHARGE__4': 'Discharge in Quarter 4'})
    """
    year2013.rename(columns={'ETHNICITY__1': 'Hispanic', 'ETHNICITY__2': 'Non-Hispanic'},
                    inplace=True)
    """
    African_Am = year2013.loc[year2013['RACE'] == '3']
    African_Am.drop(columns=['RACE'], inplace=True)
    # African_Am.to_csv('Data/AfricanAmerican_' + date + '.csv', index=False)

    Native_Am = year2013.loc[year2013['RACE'] == '1']
    Native_Am.drop(columns=['RACE'], inplace=True)
    # Native_Am.to_csv('Data/NativeAmerican_' + date + '.csv', index=False)

    # One hot encoding race
    year2013 = pd.get_dummies(year2013, prefix_sep="__", dummy_na=True,
                              columns=['RACE'])
    year2013.loc[
        year2013["RACE__nan"] == 1, year2013.columns.str.startswith("RACE__")] = np.nan

    year2013 = year2013.drop(['RACE__nan'], axis=1)
    """
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

    """


    return year2013


def cleanDataOK(dropMetro, age='Ordinal'):
    system = 'linux'

    if system == 'windows':
        parent = os.path.dirname(os.getcwd())
        dataPath2017 = os.path.join(parent, r'Data\Oklahom_PUDF_2020.08.27\2017_IP\pudf_cd.txt')
        dataPath2018 = os.path.join(parent, r'Data\Oklahom_PUDF_2020.08.27\2018_IP\pudf_cdv2.txt')

        ok2017 = pd.read_csv(dataPath2017)
        ok2018 = pd.read_csv(dataPath2018)

    else:
        parent = os.path.dirname(os.getcwd())
        dataPath2017 = os.path.join(parent, r"Data/Oklahom_PUDF_2020.08.27/2017_IP/pudf_cd.txt")
        dataPath2018 = os.path.join(parent, r"Data/Oklahom_PUDF_2020.08.27/2018_IP/pudf_cdv2.txt")

        ok2017 = pd.read_csv(dataPath2017)
        ok2018 = pd.read_csv(dataPath2018)

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

    ruralPath = '/home/rachel/PycharmProjects/Bennett_OUDSA5900/Data/County_Metropolitan_Classification.csv'

    urbanRural = pd.read_csv(ruralPath)
    urbanRural['county name'] = urbanRural['county name'].str.replace(' County', '')
    urbanRural['Metro status'] = urbanRural['Metro status'].replace('Metropolitan', 1)
    urbanRural['Metro status'] = urbanRural['Metro status'].replace('Nonmetropolitan', 0)
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
    ok2017['Marital_status'] = ok2017['Marital_status'].replace('M', 1)
    ok2018['Marital_status'] = ok2018['Marital_status'].replace('M', 1)
    ok2017['Marital_status'] = ok2017['Marital_status'].replace('N', 0)
    ok2018['Marital_status'] = ok2018['Marital_status'].replace('N', 0)

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
    diseaseDictionary['Periodontal disease'] = ['E08630', 'E09630', 'E10630', 'E11630', 'E13630', 'K05', 'K06',
                                                'K08129']
    diseaseDictionary['Other cardiovascular diseases complicating pregnancy and childbirth or the puerperium'] = [
        'O9943']
    diseaseDictionary['Obstructive Sleep Apnea'] = ['G4733']
    diseaseDictionary['Sickle cell disease'] = ['D57']
    diseaseDictionary['Thyroid Disease'] = ['E00', 'E01', 'E02', 'E03', 'E04', 'E05', 'E06', 'E07']
    # diseaseDictionary['Intrauterine Death'] = ['O364']
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

    data = ok2017.append(ok2018)

    African_Am = data.loc[data['Race'] == 'Black']
    African_Am.drop(columns=['Race'], inplace=True)

    Native_Am = data.loc[data['Race'] == 'Native American']
    Native_Am.drop(columns=['Race'], inplace=True)

    African_Am = data.loc[data['Race'] == 'Black']
    African_Am.drop(columns=['Race'], inplace=True)

    Native_Am = data.loc[data['Race'] == 'Native American']
    Native_Am.drop(columns=['Race'], inplace=True)

    White = data.loc[data['Race'] == 'White']
    White.drop(columns=['Race'], inplace=True)

    """
    # Setting dummies to true makes a column for each category that states whether or not it is missing (0 or 1).
    ok2017 = pd.get_dummies(ok2017, prefix_sep="__", dummy_na=True,
                            columns=['Race'])

    # Propogates the missing values via the indicator columns
    ok2017.loc[ok2017["Race__nan"] == 1, ok2017.columns.str.startswith("Race__")] = np.nan

    # Drops the missingness indicator columns
    ok2017 = ok2017.drop(['Race__nan'], axis=1)

    # Setting dummies to true makes a column for each category that states whether or not it is missing (0 or 1).
    ok2018 = pd.get_dummies(ok2018, prefix_sep="__", dummy_na=True,
                            columns=['Race'])

    # Propogates the missing values via the indicator columns
    ok2018.loc[ok2018["Race__nan"] == 1, ok2018.columns.str.startswith("Race__")] = np.nan

    # Drops the missingness indicator columns
    ok2018 = ok2018.drop(['Race__nan'], axis=1)

    ok2017.rename(columns={'Race__White': 'White',
                           'Race__Native American': 'Native American',
                           'Race__Black': 'Black',
                           'Race__Other/Unknown': 'Other/Unknown Race'}, inplace=True)

    ok2018.rename(columns={'Race__White': 'White',
                           'Race__Native American': 'Native American',
                           'Race__Black': 'Black',
                           'Race__Other/Unknown': 'Other/Unknown Race'}, inplace=True)

    if (dropMetro == True):
        African_Am.drop(columns=['Metro status'], inplace=True)
        Native_Am.drop(columns=['Metro status'], inplace=True)
        White.drop(columns=['Metro status'], inplace=True)
        ok2017.drop(columns=['Metro status'], inplace=True)
        ok2018.drop(columns=['Metro status'], inplace=True)

    # ok2017.to_csv('Data/Oklahoma_Clean/ok2017_Incomplete.csv', index=False)
    # ok2018.to_csv('Data/Oklahoma_Clean/ok2018_Incomplete.csv', index=False)
    """

    return ok2017.append(ok2018)


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


if __name__ == "__main__":

    """
    IMPORTANT: I commented out the sections that turn race into dummy variables in OkClean. Uncomment
    to set it to the setup used for regular analysis. 
    """

    data = cleanDataTX(dropMetro=True, age='Categorical')

    data.to_csv('../Data/Cleaned/Texas/TxCleaned_NoDummyRace_' + date + '.csv', index=False)

    #data = cleanDataOK(dropMetro=True, age='Categorical')

    #data.to_csv('../Data/Cleaned/Oklahoma/okCleaned_NoDummyRace_' + date + '.csv', index=False)
