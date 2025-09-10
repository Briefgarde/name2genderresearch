import requests
import pandas as pd
import json
from abc import ABC, abstractmethod
from dotenv import load_dotenv
import os
import gender_guesser.detector as gender

load_dotenv()


class ServiceHandler(ABC):
    @abstractmethod
    def __init__(self, datasource : pd.DataFrame):
        super().__init__()
        self.datasource = datasource

    @abstractmethod
    def callAPI():
        raise NotImplementedError
    
    @abstractmethod
    def parse_response():
        raise NotImplementedError

    @abstractmethod
    def get_prediction():
        raise NotImplemented

# Genderize.io is an online API available at https://genderize.io/, 
# and its documentation is available at https://genderize.io/documentation
# Genderize.io propose a single endpoint that accepts a 'name' parameter as well as a localization variable.
# When passing a name, the service recommands to always use a first name only, 
# otherwise it'll attempts parsing the full name with no guarantee of results. 
# The API expect to receive the APIKEY as a parameter in the URL. 
class GenderizeIoHandler(ServiceHandler):
    key = os.getenv("genderizeIO_key")
    url = f"https://api.genderize.io/"
    def __init__(self, datasource: pd.DataFrame, hasSubscription:bool):
        super().__init__(datasource)
        self.hasSubscription = hasSubscription

    def callAPI(self, useLocalization:bool)->list[str]:
        """
        By using a rather simple caller like request here (which is synchronous)
        we guarantee that we're preserving the order of operation of the list. 
        This means the responses can then be linked safely to the main datasource index. 
        """
        responses = []
        for _, row in self.datasource.iterrows():
            querystring = {'name': row['firstName'], }
            if self.hasSubscription:
                querystring['apikey'] = self.key
            if useLocalization:
                querystring['country'] = row['isoCountry']

            response = requests.get(self.url, params=querystring)
            responses.append(response.text) 
        # we only care about the main payload the API return. 
        # The rest of the info (such as response code or latency) can be neglected. 

        return responses

        
# genderize.io returns a JSON per call :
# {
#     "count": 1094417,
#     "name": "peter",
#     "gender": "male",
#     "probability": 1
# }
# This is the standard JSON that is returned even if localization is used
    def parse_response(self, responses:list[str], useLocalization:bool)->pd.DataFrame:
        response_list = []
        for i in range(len(responses)):
            r_dict = json.loads(responses[i])
            source = self.datasource.iloc[i]
            fullName = source['fullName']
            namePassed = r_dict.get('name')
            correct_gender = source['gender']
            predicted_gender = r_dict.get('gender')
            localization = source['isoCountry']
            service_used = 'genderize.IO'

            # elements that are not guaranteed to be shared by every service are noted as extra
            extra_count = r_dict.get('count')
            extra_probability = r_dict.get('probability')

            response_list.append([i, 
                                  fullName, 
                                  namePassed, 
                                  correct_gender, 
                                  predicted_gender, 
                                  localization,
                                  useLocalization,
                                  service_used, 
                                  extra_count, 
                                  extra_probability])

        return pd.DataFrame(response_list, columns=['index', 
                                                    'fullName', 
                                                    'namePassed', 
                                                    'correctGender', 
                                                    'predictedGender',
                                                    'localization',
                                                    'useLocalization',
                                                    'serviceUsed',
                                                    'extraCount',
                                                    'extraProbability'])
    
    def get_prediction(self, useLocalization:bool)->pd.DataFrame:
        response = self.callAPI(useLocalization)
        prediction = self.parse_response(response, useLocalization)
        return prediction
    
class GenderAPI_IO_Handler(ServiceHandler):
    key = os.getenv("genderAPI_IO_key")
    url = "https://api.genderapi.io/api"
    def __init__(self, datasource):
        super().__init__(datasource)

    def callAPI(self, useLocalization:bool, useAI:bool=False, force:bool=False)->list[str]:
        responses = []

        for _, row in self.datasource.iterrows():
            payload = {'name' : row['firstName']}
            if useLocalization:
                payload['country'] = row['isoCountry']
            if useAI:
                payload['askToAI'] = useAI
            if force:
                payload['forceToGenderize']=force

            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.key}"
            }

            response = requests.post(self.url, headers=headers, json=payload)
            responses.append(response.text)

        return responses
    

    # {
    # "status": true,
    # "used_credits": 1,
    # "remaining_credits": 4999,
    # "expires": 1743659200,
    # "q": "Alice", # name as input
    # "name": "Alice", # name considered
    # "gender": "female",
    # "country": "US", # Country considered by the system. It can explicitly be different to the one passed ! 
    # "total_names": 10234,
    # "probability": 98,
    # "duration": "4ms"
    # }
    def parse_response(self, responses:list[str], useLocalization:bool, useAI:bool=False, force:bool=False)->pd.DataFrame:
        response_list = []
        for i in range(len(responses)):
            r_dict = json.loads(responses[i])
            source = self.datasource.iloc[i]

            fullName = source['fullName']
            namePassed = r_dict.get('name')
            correct_gender = source['gender']
            predicted_gender = r_dict.get('gender')
            localization = source['isoCountry']
            service_used = 'genderAPI.io'

            # extras
            extra_total_names = r_dict.get('total_names')
            extra_probability = r_dict.get('probability')
            extra_country_used_by_service = r_dict.get('country')

            response_list.append([
                i,
                fullName,
                namePassed,
                correct_gender,
                predicted_gender,
                localization, 
                useLocalization,
                service_used, 
                extra_total_names,
                extra_probability,
                extra_country_used_by_service,
                useAI,
                force
            ])
        return pd.DataFrame(response_list, columns=['index', 
                                                    'fullName', 
                                                    'namePassed', 
                                                    'correctGender', 
                                                    'predictedGender',
                                                    'localization',
                                                    'useLocalization',
                                                    'serviceUsed',
                                                    'extraTotalName',
                                                    'extraProbability',
                                                    'extraCountryUsedByService',
                                                    'extraUsedAI',
                                                    'extraForcedGenderize'
        ])

    def get_prediction(self, useLocalization:bool, useAI:bool=False, force:bool=False)->pd.DataFrame:
        response = self.callAPI(useLocalization)
        prediction = self.parse_response(response, useLocalization, useAI, force)
        return prediction
            

class GenderGuesserHandler(ServiceHandler):
    # This dictionary is there to guide how to interpret the predicted gender that the service answers first and align it with the answer of the other services. 
    # The actual response of the service is categorized in another columns. 
    gender_result_dict = {
        'male' : 'male',
        'mostly male': 'male',
        'female' : 'female',
        'mostly female' : 'female',
        'andy' : 'unknown',
        'unknown' : 'unknown'
    }

    def __init__(self, datasource:pd.DataFrame):
        super().__init__(datasource)
        self.detector = gender.Detector()

    def callAPI(self, useLocalization:bool)->list[str]:
        responses = []
        for _, r in self.datasource.iterrows():
            name = r['firstName']
            if useLocalization:
                country = r['isoCountry']
                response = self.detector.get_gender(name=name, country=country)
            else : 
                response = self.detector.get_gender(name=name)
            responses.append(response)
        return responses


    def parse_response(self, responses:list[str], useLocalization:bool):
        """
        The responses from genderGuesser are very simple, since they are not a fully constructed JSON but a simple string containing the prediction. 
        GenderGuesser can answer several strings : male, female, mostly male, mostly female, andy (for androgynous) and unknown. 
        """
        response_list = []

        for i in range(len(responses)):
            r = responses[i]
            source = self.datasource.iloc[i]

            fullName = source['fullName']
            namePassed = source['firstName']
            correct_gender = source['gender']
            predicted_gender = self.gender_result_dict.get(r)
            localization = source['isoCountry']
            service_used = 'genderGuesser'

            # extra
            extra_precise_gender_predicted = r

            response_list.append([
                i,
                fullName,
                namePassed,
                correct_gender,
                predicted_gender, 
                localization,
                useLocalization,
                service_used, 
                extra_precise_gender_predicted
            ])

        return pd.DataFrame(response_list, columns=['index', 
                                                    'fullName', 
                                                    'namePassed', 
                                                    'correctGender', 
                                                    'predictedGender',
                                                    'localization',
                                                    'useLocalization',
                                                    'serviceUsed',
                                                    'extraPreciseGenderPredicted'
        ])


    def get_prediction(self, useLocalization:bool)->pd.DataFrame:
        responses = self.callAPI(useLocalization)
        return self.parse_response(responses, useLocalization)
    

class GenderAPI_com_Handler(ServiceHandler):
    key = os.getenv('genderAPI_com_key')
    url = 'https://gender-api.com/v2/gender'

    def __init__(self, datasource):
        super().__init__(datasource)

    def callAPI(self, useLocalization:bool, useFullName:bool):
        headers = {
            'Content-Type' : 'application/json',
            'Authorization': f'Bearer {self.key}'
        }
        responses = []
        for _, row in self.datasource.iterrows():
            payload = {}
            if not useFullName:
                payload['first_name'] = row['firstName']
            else:
                payload['full_name'] = row['fullName']
            
            if useLocalization:
                payload['country'] = row['isoCountry']

            response = requests.post(url=self.url, headers=headers, json=payload)
            responses.append(response.text) 

        return responses
    

#     {
#    "input":{
#       "full_name":"Clara Benson", # this says "first_name" if not useFullName
#       "country":"GH" # only present if useLocalization
#    },
#    "details":{
#       "credits_used":1,
#       "duration":"39ms",
#       "samples":81,
#       "country":"GH", # set to null if not UseLocal
#       "first_name_sanitized":"clara"
#    },
#    "result_found":true,
#    "last_name":"Benson", # only if useLastName
#    "first_name":"Clara", # This would also use any middle name if present
#    "probability":0.98,
#    "gender":"female"
# }
    def parse_response(self, responses:list[str], useLocalization:bool, useFullName:bool)->pd.DataFrame:
        response_list = []
        for i in range(len(responses)):
            r_dict = json.loads(responses[i])
            source = self.datasource.iloc[i]

            fullName = source['fullName']
            namePassed = r_dict['input']['full_name'] if useFullName else r_dict['input']['first_name']
            correct_gender = source['gender']
            predicted_gender = r_dict.get('gender')
            localization = source['isoCountry']
            service_used = 'genderAPI.com'

            #extra
            extra_probability = r_dict.get('probability')
            extra_identifiedFirstName = r_dict.get('first_name')
            extra_identifiedLastName = r_dict.get('last_name')

            response_list.append([
                i,
                fullName,
                namePassed,
                correct_gender,
                predicted_gender,
                localization,
                useLocalization,
                service_used,
                extra_probability,
                extra_identifiedFirstName,
                extra_identifiedLastName
            ])

        return pd.DataFrame(response_list, columns=[
            'index',
            'fullName',
            'namePassed',
            'correctGender',
            'predictedGender',
            'localization',
            'useLocalization',
            'serviceUsed',
            'extraProbability',
            'extraIdentifiedFirstName',
            'extraIdentifiedLastName'
        ])



    def get_prediction(self, useLocalization:bool, useFullName:bool)->pd.DataFrame:
        responses = self.callAPI(useLocalization, useFullName)

        return self.parse_response(responses, useLocalization, useFullName)
