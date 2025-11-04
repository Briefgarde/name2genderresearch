import requests
import pandas as pd
import json
from abc import ABC, abstractmethod
from dotenv import load_dotenv
import os
import gender_guesser.detector as gender
from enum import Enum
import asyncio
import aiohttp

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
    def __init__(self, datasource: pd.DataFrame, hasSubscription=True):
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
            if useLocalization and not pd.isna(row['isoCountry']):
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
    # This dictionary is there to guide how to interpret the predicted gender 
    # that the service answers first and align it with the answer of the other services. 
    # The actual response of the service is categorized in another columns. 
    gender_result_dict = {
        'male' : 'male',
        'mostly_male': 'male',
        'female' : 'female',
        'mostly_female' : 'female',
        'andy' : 'unknown',
        'unknown' : 'unknown'
    }
    COUNTRIES = {'GB': 'great_britain',
 'IE': 'ireland',
 'US': 'usa',
 'IT': 'italy',
 'MT': 'malta',
 'PT': 'portugal',
 'ES': 'spain',
 'FR': 'france',
 'BE': 'belgium',
 'LU': 'luxembourg',
 'NL': 'the_netherlands',
 'DE': 'germany',
 'AT': 'austria',
 'CH': 'swiss',
 'IS': 'iceland',
 'DK': 'denmark',
 'NO': 'norway',
 'SE': 'sweden',
 'FI': 'finland',
 'EE': 'estonia',
 'LV': 'latvia',
 'LT': 'lithuania',
 'PL': 'poland',
 'CZ': 'czech_republic',
 'SK': 'slovakia',
 'HU': 'hungary',
 'RO': 'romania',
 'BG': 'bulgaria',
 'BA': 'bosniaand',
 'HR': 'croatia',
 'MK': 'macedonia',
 'ME': 'montenegro',
 'RS': 'serbia',
 'SI': 'slovenia',
 'AL': 'albania',
 'GR': 'greece',
 'RU': 'russia',
 'BY': 'belarus',
 'MD': 'moldova',
 'UA': 'ukraine',
 'AM': 'armenia',
 'AZ': 'azerbaijan',
 'GE': 'georgia',
 'TR': 'turkey',
 'SA': 'arabia',
 'IL': 'israel',
 'CN': 'china',
 'IN': 'india',
 'JP': 'japan',
 'KR': 'korea',
 'VN': 'vietnam'}

    def __init__(self, datasource:pd.DataFrame):
        super().__init__(datasource)
        self.detector = gender.Detector()

    def callAPI(self, useLocalization:bool)->list[str]:
        responses = []
        for _, r in self.datasource.iterrows():
            name = r['firstName']
            if useLocalization:
                country = self.COUNTRIES.get(r['isoCountry'], "other_countries")
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


    def get_prediction(self, useLocalization:bool=False)->pd.DataFrame:
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
            
            if useLocalization and not pd.isna(row['isoCountry']):
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



class NamSorEndpoint(Enum):
    FIRST_NAME = "genderBatch"
    FIRST_NAME_GEO = "genderGeoBatch"
    FULL_NAME = "genderFullBatch"
    FULL_NAME_GEO = "genderFullGeoBatch"

class NamSorHandler(ServiceHandler):
    base_url = "https://v2.namsor.com/NamSorAPIv2/api2/json/"
    method = "POST"
    key = os.getenv('nameSor_key')

    def __init__(self, datasource):
        super().__init__(datasource)

    def callAPI(self, endpoint: NamSorEndpoint):
        url = self.base_url + endpoint.value
        headers = {
            "X-API-KEY": self.key,
            "Accept": "application/json",
            "Content-Type": "application/json"
        }

        responses = []
        for _, row in self.datasource.iterrows():
            payload = {'personalNames' : []}
            
            if (endpoint == NamSorEndpoint.FIRST_NAME) or (endpoint==NamSorEndpoint.FIRST_NAME_GEO):
                payload['personalNames'].append({'firstName':row['firstName']})
            else:
                payload['personalNames'].append({'name' : row['fullName']})
            
            if (endpoint == NamSorEndpoint.FIRST_NAME_GEO) or (endpoint == NamSorEndpoint.FULL_NAME_GEO):
                if not pd.isna(row['isoCountry']):
                    payload['personalNames'][0]['countryIso2'] = row['isoCountry']

            response = requests.request(self.method, url, json=payload, headers=headers)
            responses.append(response.text)

        return responses
    
    def parse_response(self, responses:list[str], endpoint:NamSorEndpoint):
        response_list = []

        for i in range(len(responses)):
            r_dict = json.loads(responses[i])
            r_dict = r_dict['personalNames'][0]
            source = self.datasource.iloc[i]

            fullName = source['fullName']
            namePassed = r_dict['firstName'] if endpoint in [NamSorEndpoint.FIRST_NAME, NamSorEndpoint.FIRST_NAME_GEO] else r_dict['name']
            correct_gender = source['gender']
            predicted_gender = r_dict.get('likelyGender')
            localization = source['isoCountry']
            usedLocalization = True if (endpoint == NamSorEndpoint.FIRST_NAME_GEO) or (endpoint == NamSorEndpoint.FULL_NAME_GEO) else False
            service_used = 'NamSor'

            # extra
            extra_genderScale = r_dict['genderScale']
            extra_score = r_dict['score']
            extra_probabilityCalibrated = r_dict['probabilityCalibrated']
            extra_script = r_dict['script']
            
            response_list.append([
                i,
                fullName,
                namePassed,
                correct_gender, 
                predicted_gender,
                localization,
                usedLocalization,
                service_used,
                extra_genderScale,
                extra_score,
                extra_probabilityCalibrated,
                extra_script
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
            'extraGenderScale',
            'extraScore',
            'extraProbabilityCalibrated',
            'extraScript'
        ])

    def get_prediction(self, endpoint:NamSorEndpoint):
        response = self.callAPI(endpoint)
        return self.parse_response(response, endpoint)
    

class NameAPIHandler(ServiceHandler):
    key = os.getenv('nameAPI_key_prefix') + '-' + os.getenv('nameAPI_key_suffix')
    url = 'https://api.nameapi.org/rest/v5.3/genderizer/persongenderizer?apiKey='
    method = 'POST'

    genderResult_mapping = {'MALE': 'male', 'FEMALE': 'female'}
    
    def __init__(self, datasource):
        super().__init__(datasource)

    async def _fetch(self, session, payload, idx):
        """Async helper to fetch one response and preserve order with idx."""
        async with session.post(self.url + self.key, headers={"Content-Type": "application/json"}, json=payload) as resp:
            text = await resp.text()
            return idx, text  # keep index for ordering

    async def callAPI(self, useFullName: bool) -> list[str]:
        """Asynchronous API caller for NameAPI (preserves order)."""
        tasks = []
        async with aiohttp.ClientSession() as session:
            for i, row in self.datasource.iterrows():
                payload = {
                    "inputPerson": {
                        "type": "NaturalInputPerson",
                        "personName": {
                            "nameFields": [
                                {
                                    "string": row['fullName'] if useFullName else row['firstName'],
                                    "fieldType": "FULLNAME" if useFullName else "SURNAME",
                                }
                            ]
                        }
                    }
                }
                tasks.append(self._fetch(session, payload, i))

            results = await asyncio.gather(*tasks)  # list of (idx, response)
            results.sort(key=lambda x: x[0])        # sort by index to preserve order
            responses = [r for _, r in results]     # strip index
            return responses


    # {"gender":"FEMALE","maleProportion":null,"confidence":0.9333333333333333}    
    def parse_response(self, responses:list[str], useFullName:bool)->pd.DataFrame:
        response_list = []

        for i in range(len(responses)):
            r_dict = json.loads(responses[i])
            source = self.datasource.iloc[i]

            fullName = source['fullName']
            namePassed = source['fullName'] if useFullName else source['firstName']
            correctGender = source['gender']
            extra_preciseGenderPredicted = r_dict.get('gender')
            predictedGender = self.genderResult_mapping.get(extra_preciseGenderPredicted) if extra_preciseGenderPredicted in self.genderResult_mapping.keys() else 'unknown'
            localization = source['isoCountry']
            useLocalization = False # NameAPI can technically use a variant of the country code, but uses ISO-639 code of the language
            # Since it is not equivalent, I'll not be using it. 
            serviceUsed = 'NameAPI'

            # extra
            extra_maleProportion = r_dict.get('maleProportion')
            extra_confidence = r_dict.get('confidence')

            response_list.append([
                i,
                fullName,
                namePassed,
                correctGender,
                predictedGender,
                localization,
                useLocalization,
                serviceUsed,
                extra_preciseGenderPredicted,
                extra_maleProportion,
                extra_confidence
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
            'extrapreciseGenderPredicted',
            'extraMaleProportion',
            'extraConfidence'
        ])

    async def get_prediction(self, useFullName:bool)->pd.DataFrame:
        responses = await self.callAPI(useFullName)
        return self.parse_response(responses, useFullName)