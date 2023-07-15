#!/usr/bin/python3
import os
import sys
import logging
import pymongo
import requests
import numpy as np
import pandas as pd
from typing import Optional, List, Dict
from datetime import datetime, timedelta
from requests.auth import HTTPBasicAuth, HTTPDigestAuth

# configuring default logging level to DEBUG
logging.basicConfig(level=logging.INFO)

# getting reference to logger object
logger = logging.getLogger(__name__)

# paths
par_dir = os.path.abspath(os.path.join(__file__, '..', '..', '..', '..'))
data_dir = os.path.join(par_dir, 'data')

# formatting string as '_' separated and in lowercase
def format_path_component(s: str) -> str:
    return '_'.join(s.split()).lower()

# create procesed file path
def odk_file_path(commodity: str, state: str, mandi: str) -> str:
    return os.path.join(data_dir, 'crawler_data', 'odk', format_path_component(commodity),  f'{format_path_component(state)}_{format_path_component(mandi)}_prices.csv')

# mongo username and password
username = 'odk_crawler'
password = 'guess123'

def merge_values_collection(values: List[int]) -> int:
    return sum(values)/len(values)

def crawl_odk_date_range(commodity: str, state: str, mandi: str, start_date: str, end_date: Optional[str]=None, collection_name: str = 'adilabad', form_id: str = 'Price_Surveillance_Form_Adilabad_Multi_Crop_V1.0', project_id: str = '5'):
    if end_date is None:
        end_date = (datetime.strptime(start_date, '%Y-%m-%d') + timedelta(days=1)).strftime('%Y-%m-%d')
    # form_id: str = 'Price_Surveillance_Form_Adilabad_Multi_Crop_V1.0'
    # project_id: str = '5'
    url: str = 'https://odk.gramvaani.org/v1/projects/'+ project_id  +'/forms/' + form_id + '.svc/Submissions?$filter=__system/submissionDate ge ' + start_date + ' and __system/submissionDate lt ' + end_date + '&$expand=*'
    
    logger.info(f'seding request to [{url}]')
    form_data_filter = requests.get( url, auth=HTTPBasicAuth(username='mcs212154@cse.iitd.ac.in', password='CCD@_iitd123_services'))
    
    if form_data_filter.status_code != 200:
        logger.error(f'status code not 200: {form_data_filter}')
        sys.exit(1)

    
    logger.info(f'received 200 status code')
    data_json = form_data_filter.json()['value']

    if len(data_json) == 0:
        logger.error(f'empty data for [{start_date}] - [{end_date}]')
        return
    
    # TODO: check if data is already present in mongodb, then no need to crawl data -> Not doing it
    # NOTE: this is not yet handled as we can search for records with date is present or not but 
    # it might happen that some more records are appended, therefore in current implementation
    # always crawling odk for dates supplied

    # saving data locally in mongodb to be used by firebase for pushing data to firestore
    database_name = 'odk_data'
    logger.info(f'saving data locally in mongodb db: [{database_name}], collection: [{collection_name}]')
    client = pymongo.MongoClient(f'mongodb://{username}:{password}@127.0.0.1:27017/{database_name}?authSource=admin')
    
    # getting reference to database and collection
    db = client.get_database()
    collection = db[collection_name]

    reported_values: Dict[str, List[int]] = {}

    for document in data_json:
        # checking whether the document already exists, 
        # update if already exists, or insert a new document if not already exists
        query = { '__id': document['__id'] }
        collection.update_one(query, {"$set": document}, upsert=True)

        # filtering entries based on commodity
        # TODO: CROP_NAME_ID validation based on commodity passed to function
        if not document['CROP_NAME_ID'] == '1':
            continue
        date = document['today']
        # TODO: verify whether key names are different based on commodity
        price = document['CROP_NAME']['RATE_OFFERED_ID']
        if date not in reported_values:
            reported_values[date] = []
        reported_values[date].append(price)

    logger.info('computing aggregated values for each day')
    values = {}
    for date in reported_values:
        values[date] = merge_values_collection(reported_values[date])

    new_odk_crawled_df = pd.DataFrame(values.items(), columns=['DATE', 'PRICE'])

    file_path = odk_file_path(commodity=commodity, state=state, mandi=mandi)
    old_odk_crawled_df = pd.DataFrame()
    if os.path.exists(file_path):
        logger.info(f'read old data from path [{file_path}]')
        old_odk_crawled_df = pd.read_csv(file_path, usecols=['DATE', 'PRICE'])
    
    logger.info('merging old crawled data with new crawled data')
    new_odk_crawled_df = pd.concat([old_odk_crawled_df, new_odk_crawled_df], ignore_index=True)
    new_odk_crawled_df.drop_duplicates(subset=['DATE'], keep='last', inplace=True)
    new_odk_crawled_df['DATE'] = pd.to_datetime(new_odk_crawled_df['DATE'])
    new_odk_crawled_df.sort_values(by=['DATE'], inplace=True)
    
    # create file directory if not exists
    os.makedirs(os.path.dirname(file_path), exist_ok=True)

    # converting prices from floating point to int
    new_odk_crawled_df['PRICE'] = new_odk_crawled_df['PRICE'].astype(int)

    logger.info(f'saving merged data to [{file_path}]')
    new_odk_crawled_df.to_csv(file_path, index=False)

if __name__ == '__main__':
    crawl_odk_date_range(commodity='soyabean', state='telangana', mandi='adilabad', start_date='2023-01-01', end_date='2023-03-31', collection_name='adilabad', form_id='Price_Surveillance_Form_Adilabad_Multi_Crop_V1.0', project_id='5')
