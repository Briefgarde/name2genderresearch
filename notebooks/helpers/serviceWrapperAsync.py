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
        """Async HTTP request helper, preserving order with idx and tagging errors as JSON."""
        try:
            if method.upper() == "GET":
                async with session.get(url, headers=headers, params=params) as resp:
                    text = await resp.text()
                    # Tag non-200 responses as JSON-formatted error
                    if resp.status != 200:
                        return idx, json.dumps({
                            "error": f"HTTP {resp.status}",
                            "body": text[:200]  # keep only first 200 chars for brevity
                        })
                    return idx, text

            elif method.upper() == "POST":
                async with session.post(url, headers=headers, json=payload, params=params) as resp:
                    text = await resp.text()
                    if resp.status != 200:
                        return idx, json.dumps({
                            "error": f"HTTP {resp.status}",
                            "body": text[:200]
                        })
                    return idx, text

            else:
                raise ValueError(f"Unsupported HTTP method: {method}")

        except Exception as e:
            # Tag any network or runtime failure as a JSON error
            return idx, json.dumps({
                "error": "Request failed",
                "details": str(e)
            })

    async def _run_async_requests(self, **kwargs) -> list[str]:
        """Generic async runner that subclasses can use."""
        tasks = []
        async with aiohttp.ClientSession() as session:
            for i, row in self.datasource.iterrows():
                method, url, headers, payload, params, idx = self.build_request(row, i, **kwargs)
                tasks.append(self._fetch(session, method, url, headers, payload, params, idx))

            results = await asyncio.gather(*tasks)  # preserves order
            # results.sort(key=lambda x: x[0])        # explicit safeguard
            i_list, responses = zip(*results)
            return responses, i_list

    async def get_prediction_async(self, **kwargs) -> pd.DataFrame:
        """Main entrypoint: run requests and parse responses."""
        responses, i_list = await self._run_async_requests(**kwargs)
        return self.parse_response(responses, i_list, **kwargs)


class NameAPIWrapper(ServiceWrapper):
    key = os.getenv('nameAPI_key_prefix') + '-' + os.getenv('nameAPI_key_suffix')
    url = "https://api.nameapi.org/rest/v5.3/genderizer/persongenderizer" #?apiKey='
    method = 'POST'
    genderResult_mapping = {'MALE': 'male', 'FEMALE': 'female'}

    def build_request(self, row, idx, useFullName: bool = False):
        # payload = {
        #     "inputPerson": {
        #         "type": "NaturalInputPerson",
        #         "personName": {
        #             "nameFields": [
        #                 {
        #                     "string": row['fullName'] if useFullName else row['firstName'],
        #                     "fieldType": "FULLNAME" if useFullName else "SURNAME",
        #                 }
        #             ]
        #         }
        #     }
        # }

        string = row['firstName'] if not useFullName else row['fullName']
        field_type = "GIVENNAME" if not useFullName else "SURNAME"

        querystring = {"apiKey":self.key}

        payload = {
            "context": {
                "priority": "REALTIME",
                "properties": []
            },
            "inputPerson": {
                "type": "NaturalInputPerson",
                "personName": {"nameFields": [
                        {
                            "string": string,
                            "fieldType": field_type
                        }
                    ]}
            }
        }
        headers = {"Content-Type": "application/json"}

        return self.method, self.url, headers, payload, querystring, idx

    def parse_response(self, responses: list[str], i_list, useFullName: bool = False) -> pd.DataFrame:
        response_list = []
        for i in range(len(responses)):
            # from source 
            source = self.datasource.iloc[i]
            fullName = source['fullName']
            namePassed = source['fullName'] if useFullName else source['firstName']
            correctGender = source['gender']
            localization = source['isoCountry']
            serviceUsed = 'NameAPI'
            # r_dict
            ## fall back first
            extra_preciseGenderPredicted = "ERROR"
            predictedGender = "ERROR"
            maleProportion = "ERROR"
            confidence = "ERROR"

            ## try actual value
            try:
                r_dict = json.loads(responses[i])
                if "error" in r_dict and isinstance(r_dict, dict): #something bad happened
                    print(f"[ERROR] API error for {fullName}: {r_dict['error']}")
                    raise ValueError(r_dict["error"])
                extra_preciseGenderPredicted = r_dict.get('gender')
                predictedGender = self.genderResult_mapping.get(extra_preciseGenderPredicted, 'unknown')
                maleProportion = r_dict.get('maleProportion')
                confidence = r_dict.get('confidence')
            except Exception as e:
                print(f"[WARN] Failed to parse response for '{fullName}' (index {i_list[i]}): {e}")

            response_list.append([
                i_list[i], fullName, namePassed, correctGender, predictedGender,
                localization, False, serviceUsed,
                extra_preciseGenderPredicted, maleProportion, confidence
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
            
        if endpoint in [NamSorEndpoint.FIRST_NAME, NamSorEndpoint.FIRST_NAME_GEO]:
            payload['personalNames'].append({'firstName':row['firstName']})
        else:
            payload['personalNames'].append({'name' : row['fullName']})
            
        if endpoint in [NamSorEndpoint.FULL_NAME_GEO, NamSorEndpoint.FIRST_NAME_GEO]:
            if not pd.isna(row['isoCountry']):
                payload['personalNames'][0]['countryIso2'] = row['isoCountry']

        return self.method, url, headers, payload, None, idx
    
    def parse_response(self, responses:list[str], i_list, endpoint:NamSorEndpoint):
        response_list = []

        for i in range(len(responses)):
            # from source 
            source = self.datasource.iloc[i]
            fullName = source['fullName']
            correct_gender = source['gender']
            localization = source['isoCountry']
            service_used = 'NamSor'
            usedLocalization = True if endpoint in [NamSorEndpoint.FULL_NAME_GEO, NamSorEndpoint.FIRST_NAME_GEO] else False
            
            # from dict
            ## fallback : 
            namePassed = "ERROR"
            predicted_gender = "ERROR"
            extra_genderScale = "ERROR"
            extra_score = "ERROR"
            extra_probabilityCalibrated = "ERROR"
            extra_script = "ERROR"

            ## actual values
            try:
                r_dict = json.loads(responses[i])
                if "error" in r_dict and isinstance(r_dict, dict): #something bad happened
                    print(f"[ERROR] API error for {fullName}: {r_dict['error']}")
                    raise ValueError(r_dict["error"])
                r_dict = json.loads(responses[i])
                r_dict = r_dict['personalNames'][0]
                namePassed = r_dict['firstName'] if endpoint in [NamSorEndpoint.FIRST_NAME, NamSorEndpoint.FIRST_NAME_GEO] else r_dict['name']
                predicted_gender = r_dict.get('likelyGender')
                extra_genderScale = r_dict['genderScale']
                extra_score = r_dict['score']
                extra_probabilityCalibrated = r_dict['probabilityCalibrated']
                extra_script = r_dict['script']
            except Exception as e:
                print(f"[WARN] Failed to parse response for '{fullName}' (index {i_list[i]}): {e}")
            
            response_list.append([
                i_list[i],
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

    def parse_response(self, responses:list[str], i_list, useLocalization:bool, useFullName:bool)->pd.DataFrame:
        response_list = []
        for i in range(len(responses)):
            source = self.datasource.iloc[i]
            fullName = source['fullName']
            correct_gender = source['gender']
            localization = source['isoCountry']
            service_used = 'genderAPI.com'

            # fall back stuff
            namePassed = "ERROR"
            predicted_gender = "ERROR"
            extra_probability = "ERROR"
            extra_identifiedFirstName = "ERROR"
            extra_identifiedLastName = "ERROR"
            id = i_list[i]

            try:
                r_dict = json.loads(responses[i])
                if "error" in r_dict and isinstance(r_dict, dict): #something bad happened
                    print(f"[ERROR] API error for {fullName}: {r_dict['error']}")
                    raise ValueError(r_dict["error"])
                namePassed = r_dict['input']['full_name'] if useFullName else r_dict['input']['first_name']
                predicted_gender = r_dict.get('gender')
                extra_probability = r_dict.get('probability')
                extra_identifiedFirstName = r_dict.get('first_name')
                extra_identifiedLastName = r_dict.get('last_name')
            except Exception as e:
                print(f"[WARN] Failed to parse response for '{fullName}' (index {i_list[i]}): {e}")
            
            # r_dict = json.loads(responses[i])
            # source = self.datasource.iloc[i]

            # fullName = source['fullName']
            # namePassed = r_dict['input']['full_name'] if useFullName else r_dict['input']['first_name']
            # correct_gender = source['gender']
            # predicted_gender = r_dict.get('gender')
            # localization = source['isoCountry']
            # service_used = 'genderAPI.com'

            # #extra
            # extra_probability = r_dict.get('probability')
            # extra_identifiedFirstName = r_dict.get('first_name')
            # extra_identifiedLastName = r_dict.get('last_name')

            response_list.append([
                id,
                fullName,
                namePassed,
                correct_gender,
                predicted_gender,
                localization,
                useLocalization,
                service_used,
                useFullName,
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
            'extra_useFullName',
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
    
    def parse_response(self, responses:list[str], i_list, useLocalization:bool, useAI:bool=False, force:bool=False)->pd.DataFrame:
        response_list = []
        for i in range(len(responses)):
            # from source 
            source = self.datasource.iloc[i]
            fullName = source['fullName']
            correct_gender = source['gender']
            localization = source['isoCountry']
            service_used = 'genderAPI.io'
            try:
                useLocalization = useLocalization if isinstance(useLocalization, bool) else False
            except NameError:
                useLocalization = False


            #from dict
            ## fall back 
            namePassed = "ERROR"
            predicted_gender = "ERROR"
            extra_total_names = "ERROR"
            extra_probability = "ERROR"
            extra_country_used_by_service = "ERROR"

            ## actual value 
            try : 
                r_dict = json.loads(responses[i])
                if "error" in r_dict and isinstance(r_dict, dict): #something bad happened
                    print(f"[ERROR] API error for {fullName}: {r_dict['error']}")
                    raise ValueError(r_dict["error"])
                namePassed = r_dict.get('name')
                predicted_gender = r_dict.get('gender')
                extra_total_names = r_dict.get('total_names')
                extra_probability = r_dict.get('probability')
                extra_country_used_by_service = r_dict.get('country')
            except Exception as e:
                print(f"[WARN] Failed to parse response for '{fullName}' (index {i_list[i]}): {e}")


            # r_dict = json.loads(responses[i])
            # source = self.datasource.iloc[i]

            # fullName = source['fullName']
            # namePassed = r_dict.get('name')
            # correct_gender = source['gender']
            # predicted_gender = r_dict.get('gender')
            # localization = source['isoCountry']
            # service_used = 'genderAPI.io'

            # # extras
            # extra_total_names = r_dict.get('total_names')
            # extra_probability = r_dict.get('probability')
            # extra_country_used_by_service = r_dict.get('country')

            response_list.append([
                i_list[i],
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
    hasSubscription = True # THIS WILL NEED TO BE CHANGED WHEN WORKING WITH THE FULL THING
    
    def build_request(self, row, idx, useLocalization:bool):
        params = {"name": row["firstName"]}
        if useLocalization and not pd.isna(row['isoCountry']):
            params["country"] = row["isoCountry"]
        if self.hasSubscription:
            params["apikey"] = self.key
        return "GET", self.url, {}, None, params, idx

    

    def parse_response(self, responses:list[str], i_list, useLocalization:bool)->pd.DataFrame:
        response_list = []
        for i in range(len(responses)):
            # from source 
            source = self.datasource[i]
            fullName = source['fullName']
            correct_gender = source['gender']
            localization = source['isoCountry']
            service_used = 'genderize.IO'
            try:
                useLocalization = useLocalization if isinstance(useLocalization, bool) else False
            except NameError:
                useLocalization = False

            # from dict
            ## fallback
            namePassed = "ERROR"
            predicted_gender = "ERROR"
            extra_count = "ERROR"
            extra_probability = "ERROR"

            ## real value 
            try:
                r_dict = json.loads(responses[i])
                if "error" in r_dict and isinstance(r_dict, dict): #something bad happened
                    print(f"[ERROR] API error for {fullName}: {r_dict['error']}")
                    raise ValueError(r_dict["error"])
                namePassed = r_dict.get('name')
                predicted_gender = r_dict.get('gender')
                extra_count = r_dict.get('count')
                extra_probability = r_dict.get('probability')
            except Exception as e:
                print(f"[WARN] Failed to parse response for '{fullName}' (index {i_list[i]}): {e}")


            response_list.append([i_list[i], 
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