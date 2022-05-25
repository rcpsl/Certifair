from aif360.datasets import AdultDataset, GermanDataset, StandardDataset, CompasDataset, BankDataset
from aif360.metrics import BinaryLabelDatasetMetric
from aif360.metrics import ClassificationMetric
from aif360.metrics.utils import compute_boolean_conditioning_vector
import numpy as np
import pandas as pd
from sklearn.preprocessing import  MaxAbsScaler
import tempeh.configurations as tc
import os
from urllib import request
import sys
sys.path.append(os.path.join(sys.path[0],"../"))
from property import AdultProperty, BankProperty, Compas2DProperty, CompasProperty, GermanProperty, LawProperty





class LawSchoolBarDataset(StandardDataset):
    def __init__(self, label_name='admit',
                 favorable_classes=[1.0],
                 protected_attribute_names=['race', 'gender'],
                 privileged_classes=[[1.0], [1.0]],
                 instance_weights_name=None,
                 categorical_features=['college', 'year'],
                 features_to_keep=[], features_to_drop=['enroll', 'asian', 'black', 'hispanic', 'white', 'missingrace', 'urm'],
                 na_values=['?'], custom_preprocessing=None,
                 metadata={}):
        # super().__init__('lawschool', split)

        data_file = 'datasets/law.dta'
        if not os.path.exists(data_file):
            request.urlretrieve(
                'http://www.seaphe.org/databases/FOIA/lawschs1_1.dta', data_file
            )

        df = pd.read_stata(data_file)
        super().__init__(df, label_name, favorable_classes, protected_attribute_names, privileged_classes,
                        categorical_features=categorical_features, features_to_drop=features_to_drop,custom_preprocessing=custom_preprocessing)

def load_preproc_data_law(verbose = True, normalize = True):
    def custom_preprocessing(dataset):
        dataset.replace(to_replace='', value=np.nan, inplace=True)
        dataset = dataset.drop('race',axis = 1)
        dataset = dataset.rename(columns={'white': 'race'})
        return dataset
    dataset = LawSchoolBarDataset(custom_preprocessing = custom_preprocessing)

    setattr(dataset, 'scale', np.ones(dataset.features.shape[1]))

    if normalize:
        min_max_scaler = MaxAbsScaler()
        dataset.features = min_max_scaler.fit_transform(dataset.features)
        dataset.scale = min_max_scaler.scale_
    if verbose:
        # print out some labels, names, etc.
        print(("#### Training Dataset shape"))
        print(dataset.features.shape)
        print(("#### Favorable and unfavorable labels"))
        print(dataset.favorable_label, dataset.unfavorable_label)
        print(("#### Protected attribute names"))
        print(dataset.protected_attribute_names)
        print(("#### Privileged and unprivileged protected attribute values"))
        print(dataset.privileged_protected_attributes, 
            dataset.unprivileged_protected_attributes)
        print(("#### Dataset feature names"))
        print(dataset.feature_names)
        print("Dataset Postiviity Rate", 100 * np.sum(dataset.labels == 1)/len(dataset.labels) )

    return dataset

def load_preproc_data_adult(protected_attributes=None, sub_samp=False, balance=False, normalize = True, verbose = True):
    def custom_preprocessing(df):
        """The custom pre-processing function is adapted from
            https://github.com/fair-preprocessing/nips2017/blob/master/Adult/code/Generate_df.ipynb
            If sub_samp != False, then return smaller version of dataset truncated to tiny_test data points.
        """

        #Preprocessing
        df = df.replace({'workclass': {'Without-pay': 'Other/Unknown', 'Never-worked': 'Other/Unknown'}})
        df = df.replace({'workclass': {'Federal-gov': 'Government', 'State-gov': 'Government',
                                     'Local-gov': 'Government'}})
        df = df.replace({'workclass': {'Self-emp-not-inc': 'Self-Employed', 'Self-emp-inc': 'Self-Employed'}})
        df = df.replace({'workclass': {'Never-worked': 'Self-Employed', 'Without-pay': 'Self-Employed'}})
        df = df.replace({'workclass': {'?': 'Other/Unknown'}})

        df = df.replace(
            {
                'occupation': {
                    'Adm-clerical': 'White-Collar', 'Craft-repair': 'Blue-Collar',
                    'Exec-managerial': 'White-Collar', 'Farming-fishing': 'Blue-Collar',
                    'Handlers-cleaners': 'Blue-Collar',
                    'Machine-op-inspct': 'Blue-Collar', 'Other-service': 'Service',
                    'Priv-house-serv': 'Service',
                    'Prof-specialty': 'Professional', 'Protective-serv': 'Service',
                    'Tech-support': 'Service',
                    'Transport-moving': 'Blue-Collar', 'Unknown': 'Other/Unknown',
                    'Armed-Forces': 'Other/Unknown', '?': 'Other/Unknown'
                }
            }
        )

        df = df.replace({'marital-status': {'Married-civ-spouse': 'Married', 'Married-AF-spouse': 'Married',
                                                            'Married-spouse-absent': 'Married', 'Never-married': 'Single'}})

        df = df.replace({'race': {'Black': 'Other', 'Asian-Pac-Islander': 'Other',
                                                'Amer-Indian-Eskimo': 'Other'}})

        # df = df[['age', 'workclass', 'education', 'marital-status', 'occupation',
        #                         'race', 'gender', 'hours-per-week', 'income']]


        df = df.replace({'education': {'Assoc-voc': 'Assoc', 'Assoc-acdm': 'Assoc',
                                                    '11th': 'School', '10th': 'School', '7th-8th': 'School',
                                                    '9th': 'School', '12th': 'School', '5th-6th': 'School',
                                                    '1st-4th': 'School', 'Preschool': 'School'}})

        

        # Group age by decade
        df['Age (decade)'] = df['age'].apply(lambda x: x//10*10)
        # df['Age (decade)'] = df['age'].apply(lambda x: np.floor(x/10.0)*10.0)

        def group_edu(x):
            if x <= 5:
                return '<6'
            elif x >= 13:
                return '>12'
            else:
                return x

        def age_cut(x):
            if x >= 70:
                return '>=70'
            else:
                return x

        def group_race(x):
            if x == "White":
                return 1.0
            else:
                return 0.0

        # Cluster education and age attributes.
        # Limit education range
        df['Education Years'] = df['education-num'].apply(lambda x: group_edu(x))
        df['Education Years'] = df['Education Years'].astype('category')

        # Limit age range
        df['Age (decade)'] = df['Age (decade)'].apply(lambda x: age_cut(x))

        # Rename income variable
        df['Income Binary'] = df['income-per-year']
        df['Income Binary'] = df['Income Binary'].replace(to_replace='>50K.', value='>50K', regex=True)
        df['Income Binary'] = df['Income Binary'].replace(to_replace='<=50K.', value='<=50K', regex=True)

        # Recode sex and race
        df['sex'] = df['sex'].replace({'Female': 0.0, 'Male': 1.0})
        df['race'] = df['race'].apply(lambda x: group_race(x))

        if sub_samp and not balance:
            df = df.sample(sub_samp)
        if sub_samp and balance:
            df_0 = df[df['Income Binary'] == '<=50K']
            df_1 = df[df['Income Binary'] == '>50K']
            df_0 = df_0.sample(int(sub_samp/2))
            df_1 = df_1.sample(int(sub_samp/2))
            df = pd.concat([df_0, df_1])

        df = df.rename(columns={'marital-status': 'marital_status', 'hours-per-week': 'hours_per_week',
                    'capital-loss': 'capital_loss', 'capital-gain': 'capital_gain','education-num': 'education_num'})
        return df

    # XD_features = ['Age (decade)', 'Education Years', 'sex', 'race']
    # XD_features = ['age', 'education-num','capital-gain', 'occupation','workclass', 'sex', 'race']
    XD_features = ['age', 'workclass', 'education', 'education_num', 'marital_status', 'occupation',
                                'race', 'sex', 'capital_gain', 'capital_loss', 'hours_per_week']
    # XD_features = ['age', 'workclass', 'education', 'marital_status', 'occupation',
    #                          'race', 'sex', 'hours_per_week']
    D_features = ['sex', 'race'] if protected_attributes is None else protected_attributes
    Y_features = ['Income Binary']
    X_features = list(set(XD_features)-set(D_features))
    # categorical_features = ['Age (decade)', 'Education Years']
    categorical_features = ['workclass', 'education', 'marital_status','occupation' ]


    # privileged classes
    all_privileged_classes = {"sex": [1.0],
                              "race": [1.0]}

    # protected attribute maps
    all_protected_attribute_maps = {"sex": {1.0: 'Male', 0.0: 'Female'},
                                    "race": {1.0: 'White', 0.0: 'Non-white'}}
    dataset = AdultDataset(
        label_name=Y_features[0],
        favorable_classes=['>50K', '>50K.'],
        protected_attribute_names=D_features,
        privileged_classes=[all_privileged_classes[x] for x in D_features],
        instance_weights_name=None,
        categorical_features=categorical_features,
        features_to_keep=X_features+Y_features+D_features,
        na_values=['?'],
        metadata={'label_maps': [{1.0: '>50K', 0.0: '<=50K'}],
                  'protected_attribute_maps': [all_protected_attribute_maps[x]
                                for x in D_features]},
        custom_preprocessing=custom_preprocessing)

    setattr(dataset, 'scale', np.ones(dataset.features.shape[1]))
    if normalize:
        min_max_scaler = MaxAbsScaler()
        dataset.features = min_max_scaler.fit_transform(dataset.features)
        dataset.scale = min_max_scaler.scale_

    if verbose:
        # print out some labels, names, etc.
        print(("#### Training Dataset shape"))
        print(dataset.features.shape)
        print(("#### Favorable and unfavorable labels"))
        print(dataset.favorable_label, dataset.unfavorable_label)
        print(("#### Protected attribute names"))
        print(dataset.protected_attribute_names)
        print(("#### Privileged and unprivileged protected attribute values"))
        print(dataset.privileged_protected_attributes, 
            dataset.unprivileged_protected_attributes)
        print(("#### Dataset feature names"))
        print(dataset.feature_names)
        print("Dataset Postiviity Rate", 100 * np.sum(dataset.labels == 1)/len(dataset.labels) )

    return dataset


def load_preproc_data_german(protected_attributes=None, verbose = True, normalize = True):
    """
    Load and pre-process german credit dataset.
    Args:
        protected_attributes(list or None): If None use all possible protected
            attributes, else subset the protected attributes to the list.

    Returns:
        GermanDataset: An instance of GermanDataset with required pre-processing.

    """
    def custom_preprocessing(df):
        """ Custom pre-processing for German Credit Data
        """

        def group_credit_hist(x):
            if x in ['A30', 'A31', 'A32']:
                return 'None/Paid'
            elif x == 'A33':
                return 'Delay'
            elif x == 'A34':
                return 'Other'
            else:
                return 'NA'

        def group_employ(x):
            if x == 'A71':
                return 'Unemployed'
            elif x in ['A72', 'A73']:
                return '1-4 years'
            elif x in ['A74', 'A75']:
                return '4+ years'
            else:
                return 'NA'

        def group_savings(x):
            if x in ['A61', 'A62']:
                return '<500'
            elif x in ['A63', 'A64']:
                return '500+'
            elif x == 'A65':
                return 'Unknown/None'
            else:
                return 'NA'

        def group_status(x):
            if x in ['A11', 'A12']:
                return '<200'
            elif x in ['A13']:
                return '200+'
            elif x == 'A14':
                return 'None'
            else:
                return 'NA'

        status_map = {'A91': 1.0, 'A93': 1.0, 'A94': 1.0,
                    'A92': 0.0, 'A95': 0.0}
        df['sex'] = df['personal_status'].replace(status_map)


        # group credit history, savings, and employment
        df['credit_history'] = df['credit_history'].apply(lambda x: group_credit_hist(x))
        df['savings'] = df['savings'].apply(lambda x: group_savings(x))
        df['employment'] = df['employment'].apply(lambda x: group_employ(x))
        # df['age'] = df['age'].apply(lambda x: np.float(x >= 26))
        df['status'] = df['status'].apply(lambda x: group_status(x))
        df['credit'] = df['credit'].apply(lambda x: int(x == 1.))
        return df

    # Feature partitions
    XD_features = ['status', 'month', 'credit_history', 'credit_amount', 'savings', 'employment', 'sex', 'age','investment_as_income_percentage' ]
    D_features = ['sex'] if protected_attributes is None else protected_attributes
    Y_features = ['credit']
    X_features = list(set(XD_features)-set(D_features))
    categorical_features = ['credit_history', 'savings', 'employment', 'status']
    # privileged classes
    all_privileged_classes = {"sex": [1.0]}

    # protected attribute maps
    all_protected_attribute_maps = {"sex": {1.0: 'Male', 0.0: 'Female'}}
    
    dataset =  GermanDataset(
        label_name=Y_features[0],
        favorable_classes=[1],
        protected_attribute_names=D_features,
        privileged_classes=[all_privileged_classes[x] for x in D_features],
        instance_weights_name=None,
        categorical_features=categorical_features,
        features_to_keep=X_features+Y_features+D_features,
        metadata={ 'label_maps': [{1.0: 'Good Credit', 0.0: 'Bad Credit'}],
                   'protected_attribute_maps': [all_protected_attribute_maps[x]
                                for x in D_features]
                    },
        custom_preprocessing=custom_preprocessing)
    f_names = dataset.feature_names
    cat_idx = []
    num_idx = []
    for idx,name in enumerate(f_names):
        if '=' in name or name in all_privileged_classes:
            cat_idx.append(idx)
        else:
            num_idx.append(idx)

    setattr(dataset,'cat_idx', cat_idx)
    setattr(dataset,'num_idx', num_idx)

    setattr(dataset, 'scale', np.ones(dataset.features.shape[1]))
    if normalize:
        min_max_scaler = MaxAbsScaler()
        dataset.features = min_max_scaler.fit_transform(dataset.features)
        dataset.scale = min_max_scaler.scale_

    if verbose:
        # print out some labels, names, etc.
        print(("#### Training Dataset shape"))
        print(dataset.features.shape)
        print(("#### Favorable and unfavorable labels"))
        print(dataset.favorable_label, dataset.unfavorable_label)
        print(("#### Protected attribute names"))
        print(dataset.protected_attribute_names)
        print(("#### Privileged and unprivileged protected attribute values"))
        print(dataset.privileged_protected_attributes, 
            dataset.unprivileged_protected_attributes)
        print(("#### Dataset feature names"))
        print(dataset.feature_names)
        print("Dataset Postiviity Rate", 100 * np.sum(dataset.labels == 1)/len(dataset.labels) )

    return dataset

def load_preproc_data_compas(verbose = True, normalize = True):
    def custom_preprocessing(df):
        df = df[(df.days_b_screening_arrest <= 30)
            & (df.days_b_screening_arrest >= -30)
            & (df.is_recid != -1)
            & (df.c_charge_degree != 'O')
            & (df.score_text != 'N/A')]

        df['in_custody'] = pd.to_datetime(df['in_custody'])
        df['out_custody'] = pd.to_datetime(df['out_custody'])
        df['custody_time'] = (df['out_custody'] - df['in_custody']).dt.total_seconds()
        df['c_jail_in'] = pd.to_datetime(df['c_jail_in'])
        df['c_jail_out'] = pd.to_datetime(df['c_jail_out'])
        df['jail_time'] = (df['c_jail_out'] - df['c_jail_in']).dt.total_seconds()
        df['c_charge_degree_bin'] = df['c_charge_degree'].apply(lambda x: int(x != 'F'))
    

        df = df[df['race'].isin(['African-American', 'Caucasian'])]

        df['two_year_recid'] = 1 - df['two_year_recid']
        
        return df
    
    f_to_keep = ['age', 'sex', 'race', 'priors_count', 'juv_fel_count', 'c_charge_degree_bin',
                'v_score_text','custody_time', 'jail_time']
    # cat_feats = ['c_charge_degree', 'v_score_text']
    cat_feats=['age_cat','v_score_text']
    # f_to_keep = ['age', 'jail_time','race']
    # f_to_keep += cat_feats
    protected_attribute_names=['race','sex']
    privileged_classes=[['Caucasian'],['Female']]
    dataset = CompasDataset(favorable_classes=[1],custom_preprocessing= custom_preprocessing,
                        categorical_features= cat_feats, features_to_keep=f_to_keep,
                        protected_attribute_names=protected_attribute_names,
                        privileged_classes=privileged_classes)

    setattr(dataset, 'scale', np.ones(dataset.features.shape[1]))
    if normalize:
        min_max_scaler = MaxAbsScaler()
        dataset.features = min_max_scaler.fit_transform(dataset.features)
        dataset.scale = min_max_scaler.scale_
    if verbose:
        # print out some labels, names, etc.
        print(("#### Training Dataset shape"))
        print(dataset.features.shape)
        print(("#### Favorable and unfavorable labels"))
        print(dataset.favorable_label, dataset.unfavorable_label)
        print(("#### Protected attribute names"))
        print(dataset.protected_attribute_names)
        print(("#### Privileged and unprivileged protected attribute values"))
        print(dataset.privileged_protected_attributes, 
            dataset.unprivileged_protected_attributes)
        print(("#### Dataset feature names"))
        print(dataset.feature_names)
        print("Dataset Postiviity Rate", 100 * np.sum(dataset.labels == 1)/len(dataset.labels) )
    return dataset

def load_preproc_data_compas2D(verbose = True, normalize = True):
    def custom_preprocessing(df):
        df = df[(df.days_b_screening_arrest <= 30)
            & (df.days_b_screening_arrest >= -30)
            & (df.is_recid != -1)
            & (df.c_charge_degree != 'O')
            & (df.score_text != 'N/A')]

        df['in_custody'] = pd.to_datetime(df['in_custody'])
        df['out_custody'] = pd.to_datetime(df['out_custody'])
        df['custody_time'] = (df['out_custody'] - df['in_custody']).dt.total_seconds()
        df['c_jail_in'] = pd.to_datetime(df['c_jail_in'])
        df['c_jail_out'] = pd.to_datetime(df['c_jail_out'])
        df['jail_time'] = (df['c_jail_out'] - df['c_jail_in']).dt.total_seconds()
        df['c_charge_degree_bin'] = df['c_charge_degree'].apply(lambda x: int(x != 'F'))
    

        df = df[df['race'].isin(['African-American', 'Caucasian'])]

        df['two_year_recid'] = 1 - df['two_year_recid']
        
        return df
    
    f_to_keep = ['age', 'sex', 'race', 'priors_count', 'juv_fel_count', 'c_charge_degree_bin',
                'v_score_text','custody_time', 'jail_time']
    # cat_feats = ['c_charge_degree', 'v_score_text']
    cat_feats=['age_cat','v_score_text']
    f_to_keep = ['age', 'jail_time','race']
    f_to_keep += cat_feats
    protected_attribute_names=['race']
    privileged_classes=[['Caucasian']]
    dataset = CompasDataset(favorable_classes=[1],custom_preprocessing= custom_preprocessing,
                        categorical_features= cat_feats, features_to_keep=f_to_keep,
                        protected_attribute_names=protected_attribute_names,
                        privileged_classes=privileged_classes)

    setattr(dataset, 'scale', np.ones(dataset.features.shape[1]))
    if normalize:
        min_max_scaler = MaxAbsScaler()
        dataset.features = min_max_scaler.fit_transform(dataset.features)
        dataset.scale = min_max_scaler.scale_
    if verbose:
        # print out some labels, names, etc.
        print(("#### Training Dataset shape"))
        print(dataset.features.shape)
        print(("#### Favorable and unfavorable labels"))
        print(dataset.favorable_label, dataset.unfavorable_label)
        print(("#### Protected attribute names"))
        print(dataset.protected_attribute_names)
        print(("#### Privileged and unprivileged protected attribute values"))
        print(dataset.privileged_protected_attributes, 
            dataset.unprivileged_protected_attributes)
        print(("#### Dataset feature names"))
        print(dataset.feature_names)
        print("Dataset Postiviity Rate", 100 * np.sum(dataset.labels == 1)/len(dataset.labels) )
    return dataset

def load_preproc_data_bank(verbose = True, normalize = True):
    def custom_preprocessing(df):
        df['default_bin'] = df['default'].apply(lambda x: int(x == 'yes'))
        df['housing_bin'] = df['housing'].apply(lambda x: int(x == 'yes'))
        df['loan_bin'] = df['loan'].apply(lambda x: int(x == 'yes'))
        return df

    f_to_keep = ['emp.var.rate', 'cons.price.idx','cons.conf.idx', 'euribor3m', 'nr.employed', 'default_bin'
                ,'housing_bin','loan_bin']
    categorical_features=[ 'marital', 'education','poutcome']
    f_to_keep += categorical_features
    dataset = BankDataset(features_to_keep=f_to_keep, categorical_features=categorical_features,custom_preprocessing=custom_preprocessing)
    
    setattr(dataset, 'scale', np.ones(dataset.features.shape[1]))

    if normalize:
        min_max_scaler = MaxAbsScaler()
        dataset.features = min_max_scaler.fit_transform(dataset.features)
        dataset.scale = min_max_scaler.scale_

    if verbose:
        # print out some labels, names, etc.
        print(("#### Training Dataset shape"))
        print(dataset.features.shape)
        print(("#### Favorable and unfavorable labels"))
        print(dataset.favorable_label, dataset.unfavorable_label)
        print(("#### Protected attribute names"))
        print(dataset.protected_attribute_names)
        print(("#### Privileged and unprivileged protected attribute values"))
        print(dataset.privileged_protected_attributes, 
            dataset.unprivileged_protected_attributes)
        print(("#### Dataset feature names"))
        print(dataset.feature_names)
        print("Dataset Postiviity Rate", 100 * np.sum(dataset.labels == 1)/len(dataset.labels) )
    return dataset
# dataset = load_preproc_data_compas()
# pass

class Datasets():

    loaders = {
        'adult': load_preproc_data_adult,
        'german': load_preproc_data_german,
        'law': load_preproc_data_law,
        'compas': load_preproc_data_compas,
        'compas2D': load_preproc_data_compas2D,
        'bank': load_preproc_data_bank
    }

    property_cls = {
        'adult': AdultProperty,
        'german': GermanProperty,
        'law': LawProperty,
        'compas': CompasProperty,
        'compas2D' : Compas2DProperty,
        'bank': BankProperty
    }