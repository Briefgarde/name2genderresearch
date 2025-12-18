"""
This entire evaluator was taken almost directly from https://peerj.com/articles/cs-156/ with only minor modifications. 
The github of said paper is available here : https://github.com/GenderGapSTEM-PublicationAnalysis/name_gender_inference/tree/main
"""

import pandas as pd
from enum import Enum
from statsmodels.stats.contingency_tables import mcnemar

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
    
    def get_confusion_matrix(self, df=None, col_true:str='correctGender', col_pred:str='predictedGender'):
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

    def compute_all_errors(self, conf_matrix, eps=0.2):
        self.confusion_matrix = conf_matrix
        error = {}
        error['error_with_unknown'] = self.compute_error_with_unknown(self.confusion_matrix)
        error['error_without_unknown'] = self.compute_error_without_unknown(self.confusion_matrix)
        error['error_unknown'] = self.compute_error_unknown(self.confusion_matrix)
        error['error_gender_bias'] = self.compute_error_gender_bias(self.confusion_matrix)
        error['weighted_error'] = self.compute_weighted_error(self.confusion_matrix, eps)
        return error
    
        # error_with_unknown = self.compute_error_with_unknown(self.confusion_matrix)
        # error_without_unknown = self.compute_error_without_unknown(self.confusion_matrix)
        # error_unknown = self.compute_error_unknown(self.confusion_matrix)
        # error_gender_bias = self.compute_error_gender_bias(self.confusion_matrix)
        # weighted_error = self.compute_weighted_error(self.confusion_matrix)
        # return [error_with_unknown, error_without_unknown, error_gender_bias, error_unknown, weighted_error]


    # stastical tests

import numpy as np
from scipy.stats import chi2_contingency
from math import sqrt, exp, log


class StatisticalTester:
    def __init__(self, df_iso: pd.DataFrame, df_noiso: pd.DataFrame):
        """
        df_iso / df_noiso must contain:
            - 'index' : unique ID per name
            - 'predictedGender'
            - 'correctGender'
        """
        self.df_iso = df_iso.copy()
        self.df_noiso = df_noiso.copy()

    def get_contingency_table(self):
        common_ids = set(self.df_noiso["index"]) & set(self.df_iso["index"])
        df_iso_c = self.df_iso[self.df_iso["index"].isin(common_ids)]
        df_noiso_c = self.df_noiso[self.df_noiso["index"].isin(common_ids)]

        df_iso_c = df_iso_c.sort_values("index").reset_index(drop=True)
        df_noiso_c = df_noiso_c.sort_values("index").reset_index(drop=True)

        iso_correct = df_iso_c["predictedGender"] == df_iso_c["correctGender"]
        noiso_correct = df_noiso_c["predictedGender"] == df_noiso_c["correctGender"]

        a = sum(iso_correct & noiso_correct) # correct in both
        b = sum(~iso_correct & noiso_correct) # incorrect with ISO, correct without => adding ISO worsened the acc
        c = sum(iso_correct & ~noiso_correct) # correct with ISO, incorrect without => adding ISO helped the acc
        d = sum(~iso_correct & ~noiso_correct) # incorrect with both 

        table = np.array([[a, b],
                          [c, d]])
        self.contingency_table = table
        return table

    def run_mcnemar_test(self):
        try : 
            table = self.contingency_table
        except:
            table = self.get_contingency_table()
        result = mcnemar(table, exact=True, correction=True)

        # Extract b and c for effect sizes
        b, c = table[0, 1], table[1, 0]
        n = b + c

        # Effect sizes
        odds_ratio = c / b if b != 0 else np.inf
        g = (c-b) / n if n > 0 else np.nan # range of [-1, 1]
        # to note : this isn't quite Cohen's G in the standard literature, but a close, unscaled version of it. 
        # to get the standard cohen's g metric, we can simply do g/2 to be able to use the standard interpretation. 
        log_or = log(odds_ratio) if np.isfinite(odds_ratio) else np.nan
        se_log_or = sqrt(1 / b + 1 / c) if (b > 0 and c > 0) else np.nan

        ci_low, ci_high = (
            exp(log_or - 1.96 * se_log_or),
            exp(log_or + 1.96 * se_log_or),
        ) if np.isfinite(log_or) else (np.nan, np.nan)
        # if result around 1 => no particular improvement
        # above 1 (ex : 1.4,2.8)=> improvement of 1.4x to 2.8x when adding the iso
        # below 1()

        return {
            "test": "McNemar",
            "statistic": result.statistic,
            "p_value": result.pvalue,
            "b": b,
            "c": c,
            "odds_ratio": odds_ratio,
            "ci_low" : ci_low,
            "ci_high" : ci_high,
            "cohen_g": g,
            "table" : table
        }

    def run_chi_square_test(self):
        # Compute simple accuracy tables
        iso_correct = (self.df_iso["predictedGender"] == self.df_iso["correctGender"]).sum()
        noiso_correct = (self.df_noiso["predictedGender"] == self.df_noiso["correctGender"]).sum()
        iso_total = len(self.df_iso)
        noiso_total = len(self.df_noiso)

        table = np.array([
            [iso_correct, iso_total - iso_correct],
            [noiso_correct, noiso_total - noiso_correct],
        ])

        chi2, p, dof, expected = chi2_contingency(table, correction=False)
        n = table.sum()
        cramer_v = sqrt(chi2 / (n * (min(table.shape) - 1)))

        # Accuracy difference
        delta_accuracy = (iso_correct / iso_total) - (noiso_correct / noiso_total)

        return {
            "test": "Chi-square",
            "chi2": chi2,
            "p_value": p,
            "cramers_v": cramer_v,
            "delta_accuracy": delta_accuracy,
            "table": table,
        }

    def summarize(self):
        paired = self.run_mcnemar_test()
        # independent = self.run_chi_square_test()
        return {
            "paired_result": paired,
            # "independent_result": independent,
        }

class EvalManager():
    serviceList = ["genderAPI.io", "genderize.IO", 
                "genderGuesser",
                "NamSor",
                "genderAPI.com"] # at discreetion, we might add NameAPI when this end up being tested
    sourceList = ["kaggle", "wikidata"]
    useLocalList = [True, False]

    metricList = ["error_with_unknown", "error_without_unknown", "error_unknown", "error_gender_bias", "weighted_error"]

    def __init__(self):
        pass
    def runAnalysis(self, masterdf:pd.DataFrame):
        df_perf = pd.DataFrame()
        df_stat = pd.DataFrame()

        for service in self.serviceList:
            
            for source in self.sourceList:
                
                df_iso = masterdf[
                    (masterdf['source']==source) &
                    (masterdf['serviceUsed']==service) & 
                    (masterdf['useLocalization']==True)
                    ]
                df_noiso = masterdf[
                    (masterdf['source']==source) &
                    (masterdf['serviceUsed']==service) & 
                    (masterdf['useLocalization']==False)
                    ]
                
                tester = StatisticalTester(df_iso=df_iso, df_noiso=df_noiso)
                summary = tester.summarize()
                summary = summary['paired_result']
                summary['service_used'] = service
                summary['source'] = source
                summary.pop("table")
                summary.pop("test")

                df_summary = pd.DataFrame(
                    data=[summary],
                )
                df_stat = pd.concat([df_stat, df_summary])
                
                for useLocal in self.useLocalList:
                    df_metric = masterdf[
                        (masterdf['source']==source) &
                        (masterdf['serviceUsed']==service) & 
                        (masterdf['useLocalization']==useLocal)
                        ]

                    eval = Evaluator(df_metric)
                    conf_matrix = eval.get_confusion_matrix()

                    
                    result = eval.compute_all_errors(conf_matrix)
                    result['service_used'] = service
                    result['source'] = source
                    result['useLocal'] = useLocal
                    df_result = pd.DataFrame([result])
                    df_perf = pd.concat([df_perf, df_result])



        df_stat = df_stat.loc[:, ["service_used", "source", "statistic", "p_value", "b", "c", "odds_ratio", "cohen_g", "ci_low", "ci_high"]]
        df_perf = df_perf.loc[:, ["service_used", "source", "useLocal", "error_with_unknown", "error_without_unknown", "error_unknown", "error_gender_bias", "weighted_error"]]
        df_stat = df_stat.sort_values(by=["service_used", "source"])
        df_perf = df_perf.sort_values(by=["service_used", "source"])
        return df_stat, df_perf
    
    def runAnalysisWithoutUnknown(self, masterdf:pd.DataFrame):
        df_no_unknown = masterdf.loc[masterdf['predictedGender']!='unknown', :]
        return self.runAnalysis(df_no_unknown)
    
    def getMeanMetric(self, df_perf)->pd.DataFrame:
        meanMetric_df = pd.DataFrame()
        metricCol = ["error_with_unknown", "error_without_unknown", 'error_unknown', 'error_gender_bias', 'weighted_error']

        for service in df_perf['service_used'].unique():
            serviceRow = df_perf.loc[df_perf['service_used']==service, :]
            meanList = []
            for metric in self.metricList:
                meanList.append(serviceRow[metric].mean())
            df = pd.DataFrame(data=[meanList], columns=self.metricList)
            df['service'] = service
            meanMetric_df = pd.concat([df, meanMetric_df])
        meanMetric_df = meanMetric_df.loc[:, ['service']+ self.metricList]
        return meanMetric_df
    
    def getMeanMetricPerState(self, df_perf:pd.DataFrame)->pd.DataFrame:
        metric_iso_df = df_perf[df_perf['useLocal']==True]
        mean_metric_iso = self.getMeanMetric(metric_iso_df)
        mean_metric_iso['useLocal'] = True
        metric_Noiso_df = df_perf[df_perf['useLocal']==False]
        mean_metric_Noiso = self.getMeanMetric(metric_Noiso_df)
        mean_metric_Noiso['useLocal'] = False
        result =pd.concat([mean_metric_iso, mean_metric_Noiso])
        result = result.loc[:, ['service', 'useLocal']+self.metricList]
        return result

    def getConfMatrix(self, df:pd.DataFrame, useLocal:bool):
        for service in self.serviceList:
            df_metric = df[
                        (df['serviceUsed']==service) & 
                        (df['useLocalization']==useLocal)
                        ]
                
            eval = Evaluator(df_metric)
            conf_matrix = eval.get_confusion_matrix()
            conf_matrix = conf_matrix.iloc[:2]
            word = "with" if useLocal else "without"
            print(f"Conf matrix for {service} {word} useLocal")
            print(conf_matrix)
            print(conf_matrix.to_numpy().sum())
            print("------------")

    def getMetricWholeDataset(self, masterdf:pd.DataFrame):
        df_perf = pd.DataFrame()
        df_stat = pd.DataFrame()

        for service in self.serviceList:
                df_iso = masterdf[
                    
                    (masterdf['serviceUsed']==service) & 
                    (masterdf['useLocalization']==True)
                    ]
                df_noiso = masterdf[
                    
                    (masterdf['serviceUsed']==service) & 
                    (masterdf['useLocalization']==False)
                    ]
                
                tester = StatisticalTester(df_iso=df_iso, df_noiso=df_noiso)
                summary = tester.summarize()
                summary = summary['paired_result']
                summary['service_used'] = service
                
                summary.pop("table")
                summary.pop("test")

                df_summary = pd.DataFrame(
                    data=[summary],
                )
                df_stat = pd.concat([df_stat, df_summary])
                
                for useLocal in self.useLocalList:
                    df_metric = masterdf[
                        
                        (masterdf['serviceUsed']==service) & 
                        (masterdf['useLocalization']==useLocal)
                        ]

                    eval = Evaluator(df_metric)
                    conf_matrix = eval.get_confusion_matrix()

                    
                    result = eval.compute_all_errors(conf_matrix)
                    result['service_used'] = service
                    
                    result['useLocal'] = useLocal
                    df_result = pd.DataFrame([result])
                    df_perf = pd.concat([df_perf, df_result])



        df_stat = df_stat.loc[:, ["service_used",  "statistic", "p_value", "b", "c", "odds_ratio", "cohen_g", "ci_low", "ci_high"]]
        df_perf = df_perf.loc[:, ["service_used",  "useLocal", "error_with_unknown", "error_without_unknown", "error_unknown", "error_gender_bias", "weighted_error"]]
        df_stat = df_stat.sort_values(by=["service_used"])
        df_perf = df_perf.sort_values(by=["service_used"])
        return df_stat, df_perf
    

    def metricPerCountry(self, masterdf:pd.DataFrame, threshhold=500):
        countryRepresentation = masterdf['localization'].value_counts()
        listCountry = masterdf['localization'].dropna().unique()

        df_perf = pd.DataFrame()

        for service in self.serviceList:
        
            for useLocal in self.useLocalList:
                    
                for country in listCountry:
                    population = countryRepresentation.loc[country]
                    if population>threshhold:
                        df_metric = masterdf[
                            (masterdf['serviceUsed']==service) & 
                            (masterdf['useLocalization']==useLocal) & 
                            (masterdf['localization']==country)
                        ]

                        eval = Evaluator(df_metric)
                        conf_matrix = eval.get_confusion_matrix()

                        
                        result = eval.compute_all_errors(conf_matrix)
                        result['service_used'] = service
                        result['useLocal'] = useLocal
                        result['country']=country
                        result['population'] = population/10
                        df_result = pd.DataFrame([result])
                        df_perf = pd.concat([df_perf, df_result])
                    else:
                        pass



        df_perf = df_perf.loc[:, ["service_used", "country", "useLocal", 'population', "error_with_unknown", "error_without_unknown", "error_unknown", "error_gender_bias", "weighted_error"]]
        df_perf = df_perf.sort_values(by=["service_used"])
        return df_perf
    


