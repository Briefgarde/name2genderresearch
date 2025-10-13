
"""
This entire evaluator was taken almost directly from https://peerj.com/articles/cs-156/ with only minor modifications. 
The github of said paper is available here : https://github.com/GenderGapSTEM-PublicationAnalysis/name_gender_inference/tree/main
"""

import pandas as pd
from enum import Enum

class Gender(Enum):
    MALE = 'male'
    FEMALE = 'female'
    UNKNOWN = 'unknown'

class Evaluator():
    def __init__(self, dataframe:pd.DataFrame):
        self.dataframe = dataframe

    def get_all_assignement(self, df=None, col_true:str='correctGender', col_pred:str='predictedGender'):
        """
        This method goes through a dataframe and return all of the possible assignements, along the metrics presented in Santamaria. 
        The assignements is a confusion matrix like this : 
        | true \ predicted | male | female | unknown |
        |------------------|------|--------|---------|
        | male             | m_m  | m_f    | m_u     |
        | female           | f_m  | f_f    | f_u     |
        Thus, m_m mean the person is male, and the tool predicted male. 
        f_m mean the person is female, but the tool predicted male. 
        m_u mean the person is male, but the tool couldn't not predict a result.
        This method returns the number of assignement in each of those categories. 
        """
        if df is None:
            df = self.dataframe
        
        result = {}

        result['f_f'] = len(df[(df[col_true] == Gender.FEMALE.value) & (df[col_pred] == Gender.FEMALE.value)]) 
        result['f_m'] = len(df[(df[col_true] == Gender.FEMALE.value) & (df[col_pred] == Gender.MALE.value)])
        result['f_u'] = len(df[(df[col_true] == Gender.FEMALE.value) & (df[col_pred] == Gender.UNKNOWN.value)])
        result['m_f'] = len(df[(df[col_true] == Gender.MALE.value) & (df[col_pred] == Gender.FEMALE.value)])
        result['m_m'] = len(df[(df[col_true] == Gender.MALE.value) & (df[col_pred] == Gender.MALE.value)])
        result['m_u'] = len(df[(df[col_true] == Gender.MALE.value) & (df[col_pred] == Gender.UNKNOWN.value)])
        result['u_f'] = len(df[(df[col_true] == Gender.UNKNOWN.value) & (df[col_pred] == Gender.FEMALE.value)])
        result['u_m'] = len(df[(df[col_true] == Gender.UNKNOWN.value) & (df[col_pred] == Gender.MALE.value)])
        result['u_u'] = len(df[(df[col_true] == Gender.UNKNOWN.value) & (df[col_pred] == Gender.UNKNOWN.value)])
        
        return result
    
    def get_confusion_matrix(self, df:None, col_true:str='correctGender', col_pred:str='predictedGender'):
        if df is None:
            df = self.dataframe
        result = self.get_all_assignement(df, col_true, col_pred)
        return pd.DataFrame([[result['f_f'], result['f_m'], result['f_u']], [result['m_f'], result['m_m'], result['m_u']], [result['u_f'], result['u_m'], result['u_u']]], index=['f', 'm', 'u'],
                            columns=['f_pred', 'm_pred', 'u_pred'])
    

    def compute_error_without_unknown(self, conf_matrix):
        """Corresponds to 'errorCodedWithoutNA' from genderizeR-paper:
        https://journal.r-project.org/archive/2016/RJ-2016-002/RJ-2016-002.pdf"""
        error_without_unknown = (conf_matrix.loc['f', 'm_pred'] + conf_matrix.loc['m', 'f_pred']) / \
                                (conf_matrix.loc['f', 'm_pred'] + conf_matrix.loc['m', 'f_pred'] +
                                conf_matrix.loc['f', 'f_pred'] + conf_matrix.loc['m', 'm_pred'])

        return error_without_unknown

    def compute_error_with_unknown(self, conf_matrix):
        """
        Corresponds to 'errorCoded' from genderizeR-paper:
        https://journal.r-project.org/archive/2016/RJ-2016-002/RJ-2016-002.pdf"""
        true_f_and_m = conf_matrix.loc['f', :].sum() + conf_matrix.loc['m', :].sum()
        true_pred_f_and_m = conf_matrix.loc['f', 'f_pred'] + conf_matrix.loc['m', 'm_pred']
        error_with_unknown = (true_f_and_m - true_pred_f_and_m) / true_f_and_m

        return error_with_unknown

    def compute_error_unknown(self, conf_matrix):
        """Corresponds 'naCoded' from genderizeR-paper:
        https://journal.r-project.org/archive/2016/RJ-2016-002/RJ-2016-002.pdf"""
        true_f_and_m = conf_matrix.loc['f', :].sum() + conf_matrix.loc['m', :].sum()
        error_unknown = (conf_matrix.loc['f', 'u_pred'] + conf_matrix.loc['m', 'u_pred']) / true_f_and_m

        return error_unknown

    def compute_error_gender_bias(self, conf_matrix):
        """Corresponds 'errorGenderBias' from genderizeR-paper:
        https://journal.r-project.org/archive/2016/RJ-2016-002/RJ-2016-002.pdf"""
        error_gender_bias = (conf_matrix.loc['m', 'f_pred'] - conf_matrix.loc['f', 'm_pred']) / \
                            (conf_matrix.loc['f', 'f_pred'] + conf_matrix.loc['f', 'm_pred'] +
                            conf_matrix.loc['m', 'f_pred'] + conf_matrix.loc['m', 'm_pred'])

        return error_gender_bias
    
    def compute_weighted_error(self, conf_matrix, eps=0.2):
        """Compute weighted version of 'error_with_unknown', where terms related to classifying 'f' and 'm' as 'u'
        is multiplied with 'eps'."""
        numer = (conf_matrix.loc['m', 'f_pred'] + conf_matrix.loc['f', 'm_pred'] + eps * (
                conf_matrix.loc['m', 'u_pred'] + conf_matrix.loc['f', 'u_pred']))
        denom = (conf_matrix.loc['f', 'f_pred'] + conf_matrix.loc['f', 'm_pred'] + conf_matrix.loc['m', 'f_pred'] +
                 conf_matrix.loc['m', 'm_pred'] + eps * (
                        conf_matrix.loc['m', 'u_pred'] + conf_matrix.loc['f', 'u_pred']))
        return numer / denom

    def compute_all_errors(self, conf_matrix):
        self.confusion_matrix = conf_matrix
        error = {}
        error['error_with_unknown'] = self.compute_error_with_unknown(self.confusion_matrix)
        error['error_without_unknown'] = self.compute_error_without_unknown(self.confusion_matrix)
        error['error_unknown'] = self.compute_error_unknown(self.confusion_matrix)
        error['error_gender_bias'] = self.compute_error_gender_bias(self.confusion_matrix)
        error['weighted_error'] = self.compute_weighted_error(self.confusion_matrix)
        return error
    
        # error_with_unknown = self.compute_error_with_unknown(self.confusion_matrix)
        # error_without_unknown = self.compute_error_without_unknown(self.confusion_matrix)
        # error_unknown = self.compute_error_unknown(self.confusion_matrix)
        # error_gender_bias = self.compute_error_gender_bias(self.confusion_matrix)
        # weighted_error = self.compute_weighted_error(self.confusion_matrix)
        # return [error_with_unknown, error_without_unknown, error_gender_bias, error_unknown, weighted_error]