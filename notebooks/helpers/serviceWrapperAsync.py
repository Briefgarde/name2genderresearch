import aiohttp
import asyncio
import pandas as pd
from abc import ABC, abstractmethod
import os
import json
from enum import Enum 

class ServiceWrapper(ABC):
    def __init__(self, datasource: pd.DataFrame):
        super().__init__()
        self.datasource = datasource

    # Each subclass must still implement its logic
    @abstractmethod
    def build_request(self, row, idx, **kwargs):
        """
        Subclass: return (method, url, headers, payload, params, idx).
        """
        pass

    @abstractmethod
    def parse_response(self, responses: list[str], **kwargs) -> pd.DataFrame:
        """
        Subclass: take raw responses and turn them into a DataFrame.
        """
        pass

    async def _fetch(self, session, method, url, headers, payload, params, idx):
        """Async HTTP request helper, preserving order with idx."""
        if method.upper() == "GET":
            async with session.get(url, headers=headers, params=params) as resp:
                return idx, await resp.text()
        elif method.upper() == "POST":
            async with session.post(url, headers=headers, json=payload, params=params) as resp:
                return idx, await resp.text()
        else:
            raise ValueError(f"Unsupported HTTP method: {method}")

    async def _run_async_requests(self, **kwargs) -> list[str]:
        """Generic async runner that subclasses can use."""
        tasks = []
        async with aiohttp.ClientSession() as session:
            for i, row in self.datasource.iterrows():
                method, url, headers, payload, params, idx = self.build_request(row, i, **kwargs)
                tasks.append(self._fetch(session, method, url, headers, payload, params, idx))

            results = await asyncio.gather(*tasks)  # preserves order
            results.sort(key=lambda x: x[0])        # explicit safeguard
            responses = [r for _, r in results]
            return responses

    async def get_prediction_async(self, **kwargs) -> pd.DataFrame:
        """Main entrypoint: run requests and parse responses."""
        responses = await self._run_async_requests(**kwargs)
        return self.parse_response(responses, **kwargs)


class NameAPIWrapper(ServiceWrapper):
    key = os.getenv('nameAPI_key_prefix') + '-' + os.getenv('nameAPI_key_suffix')
    url = 'https://api.nameapi.org/rest/v5.3/genderizer/persongenderizer?apiKey='
    method = 'POST'
    genderResult_mapping = {'MALE': 'male', 'FEMALE': 'female'}

    def build_request(self, row, idx, useFullName: bool = False):
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
        headers = {"Content-Type": "application/json"}
        return self.method, self.url + self.key, headers, payload, None, idx

    def parse_response(self, responses: list[str], useFullName: bool = False) -> pd.DataFrame:
        response_list = []
        for i in range(len(responses)):
            r_dict = json.loads(responses[i])
            source = self.datasource.iloc[i]

            fullName = source['fullName']
            namePassed = source['fullName'] if useFullName else source['firstName']
            correctGender = source['gender']
            extra_preciseGenderPredicted = r_dict.get('gender')
            predictedGender = self.genderResult_mapping.get(extra_preciseGenderPredicted, 'unknown')
            localization = source['isoCountry']
            serviceUsed = 'NameAPI'

            response_list.append([
                i, fullName, namePassed, correctGender, predictedGender,
                localization, False, serviceUsed,
                extra_preciseGenderPredicted, r_dict.get('maleProportion'), r_dict.get('confidence')
            ])

        return pd.DataFrame(response_list, columns=[
            'index','fullName','namePassed','correctGender','predictedGender',
            'localization','useLocalization','serviceUsed',
            'extraPreciseGenderPredicted','extraMaleProportion','extraConfidence'
        ])
    
class NamSorEndpoint(Enum):
    FIRST_NAME = "genderBatch"
    FIRST_NAME_GEO = "genderGeoBatch"
    FULL_NAME = "genderFullBatch"
    FULL_NAME_GEO = "genderFullGeoBatch"
class NamSorWrapper(ServiceWrapper):
    base_url = "https://v2.namsor.com/NamSorAPIv2/api2/json/"
    method = "POST"
    key = os.getenv('nameSor_key')


    def build_request(self, row, idx, endpoint:NamSorEndpoint):
        headers = {
            "X-API-KEY": self.key,
            "Accept": "application/json",
            "Content-Type": "application/json"
        }
        url = self.base_url + endpoint.value

        payload = {'personalNames' : []}
            
        if (endpoint == NamSorEndpoint.FIRST_NAME) or (endpoint==NamSorEndpoint.FIRST_NAME_GEO):
            payload['personalNames'].append({'firstName':row['firstName']})
        else:
            payload['personalNames'].append({'name' : row['fullName']})
            
        if endpoint in [NamSorEndpoint.FULL_NAME_GEO, NamSorEndpoint.FIRST_NAME_GEO]:
            if not pd.isna(row['isoCountry']):
                payload['personalNames'][0]['countryIso2'] = row['isoCountry']

        return self.method, url, headers, payload, None, idx
    
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
            usedLocalization = True if endpoint in [NamSorEndpoint.FULL_NAME_GEO, NamSorEndpoint.FIRST_NAME_GEO] else False
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


class GenderAPI_com_Wrapper(ServiceWrapper):
    key = os.getenv('genderAPI_com_key')
    url = 'https://gender-api.com/v2/gender'
    method = "POST"

    def build_request(self, row, idx, useFullName:bool, useLocalization:bool):
        headers = {
            'Content-Type' : 'application/json',
            'Authorization': f'Bearer {self.key}'
        }

        payload = {}
        if not useFullName:
            payload['first_name'] = row['firstName']
        else:
            payload['full_name'] = row['fullName']
            
        if useLocalization and not pd.isna(row['isoCountry']):
            payload['country'] = row['isoCountry']

        return self.method, self.url, headers, payload, None, idx 

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
    


class GenderAPI_IO_Wrapper(ServiceWrapper):
    key = os.getenv("genderAPI_IO_key")
    url = "https://api.genderapi.io/api"
    method = "POST"

    def build_request(self, row, idx, useLocalization:bool, useAI:bool=False, force:bool=False):
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

        return self.method, self.url, headers, payload, None, idx
    
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
    
class GenderizeWrapper(ServiceWrapper):
    key = os.getenv("genderizeIO_key")
    url = f"https://api.genderize.io/"
    method = "GET"
    hasSubscription = False # THIS WILL NEED TO BE CHANGED WHEN WORKING WITH THE FULL THING

    # def build_request(self, row, idx, useLocalization:bool):
    #     querystring = {'name': row['firstName'], }
    #     if self.hasSubscription:
    #         querystring['apikey'] = self.key
    #     if useLocalization:
    #         querystring['country'] = row['isoCountry']


    #     return self.method, self.url, None, querystring, None, idx
    
    def build_request(self, row, idx, useLocalization:bool):
        params = {"name": row["firstName"]}
        if useLocalization and not pd.isna(row['isoCountry']):
            params["country"] = row["isoCountry"]
        if self.hasSubscription:
            params["apikey"] = self.key
        return "GET", self.url, {}, None, params, idx

    

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