import requests
import pandas as pd
import json
from abc import ABC, abstractmethod
from dotenv import load_dotenv
import os

load_dotenv()


class ServiceHandler(ABC):
    @abstractmethod
    def __init__(self, datasource : pd.DataFrame):
        super().__init__()
        self.datasource = datasource
    def build_request():
        raise NotImplementedError

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
    
    def build_request():
        raise Exception("This service does not require a built request")

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
    
class GenderAPIHandler(ServiceHandler):
    key = os.getenv("genderAPIkey")
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
            service_used = 'genderAPI'

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
            


