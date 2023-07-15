import os
import sys
import pymongo
import logging
import pandas as pd
from enum import Enum

import firebase_admin
from firebase_admin import credentials, firestore

# configuring default logging level to DEBUG
logging.basicConfig(level=logging.INFO)

# getting reference to logger object
logger = logging.getLogger(__name__)

# paths
par_dir = os.path.abspath(os.path.join(__file__, '..', '..', '..'))
data_dir = os.path.join(par_dir, 'data')

# firebase authentication
cred_obj = credentials.Certificate(os.path.join(par_dir, 'code', 'utilities', 'key_firebase.json'))
firebase_admin.initialize_app(cred_obj)
db = firestore.client()

# mongo credentials
username = 'odk_crawler'
password = 'guess123'

# enumeration to identify prices and arrivals in dataframe
class commodity_info_type(Enum):
    PRICES = 1
    ARRIVALS = 2

# enumeration to identify recommendations are short or long
class recommendation_type(Enum):
    SHORT = 1
    LONG = 2

# formatting string as '_' separated and in lowercase
def format_path_component(s: str) -> str:
    return '_'.join(s.split()).lower()

# create procesed file path
def imputed_file_path(commodity: str, state: str, mandi: str, info_type: bool) -> str:
    type_file = 'prices' if info_type == commodity_info_type.PRICES else 'arrivals' 
    return os.path.join(data_dir, 'imputed_data', format_path_component(commodity), type_file ,f'{format_path_component(state)}_{format_path_component(mandi)}_{type_file}.csv')

# TODO (verify): crop name added to path
def recommendation_file_path(commodity: str, target_state: str, target_mandi: str, surrogate_state: str, surrogate_mandi: str, rtype: recommendation_type, cur_date: str) -> str:
    identifier = format_path_component(target_state) + '_' + format_path_component(target_mandi) + '_' + format_path_component(surrogate_state) + '_' + format_path_component(surrogate_mandi)
    model_type = 'shortterm_models' if rtype == recommendation_type.SHORT else 'longterm_models'
    return os.path.join(data_dir, 'recommendation_data', model_type, 'tcn', commodity, identifier, 'archive', f'recommendation_{cur_date}.csv')

def trading_file_path() -> str:
    return os.path.join(data_dir, 'crawler_data', 'trading', 'prices.csv')

# return collection name based on commodity, state, mandi and commodity information type
def daily_data_collection_name(commodity: str, state: str, mandi: str, info_type: commodity_info_type) -> str:
    type = 'Price' if info_type == commodity_info_type.PRICES else 'Arrival'
    return f'{state.upper()}_{mandi.upper()}_{type}'

# return collection name based on commodity, state, mandi and recommendation type
def recommendation_collection_name(commodity: str, state: str, mandi: str, rtype: recommendation_type) -> str:
    cname = f'{state.upper()}_{mandi.upper()}_Recommendation'
    # adding 'weekly' prefix for long term forecasts
    if rtype == recommendation_type.LONG:
        cname = 'weekly_' + cname
    return cname

# return document name based on state, mandi and commodity values
def daily_odk_collection_name(commodity: str, state: str, mandi: str) -> str:
    return f'{state.upper()}_{mandi.upper()}_ODK'

# pushing daily interpolated prices to firebase
def push_data(commodity: str, state: str, mandi: str, start_date: str, end_date: str, info_type: commodity_info_type):
    # for f in arimaInterpolateFiles:
    file_path = imputed_file_path(commodity, state, mandi, info_type)
    logging.info(f'reading imputed data from [{file_path}]')
    imputed_mandi_df = pd.read_csv(file_path, usecols=['DATE', 'PRICE'], index_col=['DATE'])
    
    # TODO (CHECK): filter dates based on years, so as to add prices within same document
    filter_dates = {}
    for cur_date in pd.date_range(start=start_date, end=end_date):
        if cur_date.year not in filter_dates:
            filter_dates[cur_date.year] = []
        filter_dates[cur_date.year].append(cur_date.strftime('%Y-%m-%d')) 

    collection_name = daily_data_collection_name(commodity=commodity, state=state, mandi=mandi, info_type=info_type)
    for year, dates in filter_dates.items():
        # iterating over year by year
        document_name = str(year)
        logger.info(f'collection: {collection_name}, document: {document_name}')
        doc_ref = db.collection(collection_name).document(document_name)
        logger.info(f'filtered dates in year {year}: {dates}')
        year_imputed_mandi_df = imputed_mandi_df.loc[dates]
        
        # creating list of records to be appended in each document
        data = []
        for idx, row in year_imputed_mandi_df.iterrows():
            data.append({'DATE': str(idx), 'PRICE': str(row['PRICE'])})
        logger.info(f'data: {data}')
        
        # creating document if not exists or appending to document if it already exists
        # ArrayUnion function ensures idempotence if same data is pushed multiple times
        if len(data):
            logger.info('pushing data to firebase')
            if doc_ref.get().exists:
                doc_ref.update({'data': firestore.ArrayUnion(data)})
            else:
                doc_ref.set({'data': firestore.ArrayUnion(data)})

# pushing daily odk crawled data stored in local mongo database to firebase
def push_odk(commodity: str, state: str, mandi: str, start_date: str, end_date: str, mongo_collection_name: str = 'adilabad'):
    # authenticating using username and password, getting reference to mongo client 
    mongo_database_name = 'odk_data'
    logger.info(f'reading data locally from mongodb db: [{mongo_database_name}], collection: [{mongo_collection_name}]')
    client = pymongo.MongoClient(f'mongodb://{username}:{password}@127.0.0.1:27017/{mongo_database_name}?authSource=admin')
    
    # getting reference to database and collection
    mongo_db = client.get_database()
    collection = mongo_db[mongo_collection_name]

    # reading all records for each date segregated by years
    filter_dates = {}
    for cur_date in pd.date_range(start=start_date, end=end_date):
        if cur_date.year not in filter_dates:
            filter_dates[cur_date.year] = []
        filter_dates[cur_date.year].append(cur_date.strftime('%Y-%m-%d')) 
    
    firebase_collection_name = daily_odk_collection_name(commodity=commodity, state=state, mandi=mandi)
    for year, dates in filter_dates.items():
        # iterating over year by year
        firebase_document_name = str(year)
        logger.info(f'firebase: Collection Name [{firebase_collection_name}], Document Name [{firebase_document_name}]')
        doc_ref = db.collection(firebase_collection_name).document(firebase_document_name)


        # creating list of records to be appended in each document
        logger.info(f'filtered dates in year {year}: {dates}')
        data = []
        for cur_date in dates:
            find_criterion = {'today': cur_date}
            data.extend(list(collection.find(find_criterion)))
        
        # creating document if not exists or appending to document if it already exists
        # ArrayUnion function ensures idempotence if same data is pushed multiple times
        if len(data):
            logger.info('pushing data to firebase for year {year}')
            if doc_ref.get().exists:
                doc_ref.update({'data': firestore.ArrayUnion(data)})
            else:
                doc_ref.set({'data': firestore.ArrayUnion(data)})
        else:
            logger.info('no data found in mongo db for pushing to firebase')

# pushing daily/weekly recommentations to firebase based on value of rtype
def push_recommendation(commodity: str, target_state: str, target_mandi: str, surrogate_state: str, surrogate_mandi: str, start_date: str, end_date: str, rtype: recommendation_type):
    firebase_collection_name = recommendation_collection_name(commodity=commodity, state=target_state, mandi=target_mandi, rtype=rtype)
    
    # iterating over entire date range and pushing recommendation for each date
    for cur_date in pd.date_range(start=start_date, end=end_date):
        cur_date_str = cur_date.strftime('%Y-%m-%d')
        logger.info(f'reading recommendation for date {cur_date_str}')

        # firebase document name is same as current date in '%Y-%m-%d' format
        firebase_document_name = cur_date_str

        # reading dataframe from recommendation file path if it exists
        rfile_path = recommendation_file_path(commodity=commodity, target_state=target_state, target_mandi=target_mandi, surrogate_state=surrogate_state, surrogate_mandi=surrogate_mandi, rtype=rtype, cur_date=cur_date_str)
        if not os.path.exists(rfile_path):
            # raise error and proceed to next iteration
            logger.error(f'recommendation file [{rfile_path}] not found')
        rdf = pd.read_csv(rfile_path)

        # creating data and pushing data to firebase document
        data = []
        for _, row in rdf.iterrows():
            data.append(row.to_dict())

        logger.info(f'firebase: Collection Name [{firebase_collection_name}], Document Name [{firebase_document_name}]')
        logger.info(f'pushing data to firebase for date [{cur_date_str}]')
        db.collection(firebase_collection_name).document(firebase_document_name).set({'data': data})

# pushing daily trading prices for cotton and soyabean
def push_trading_prices(curr_date: str):
    df = pd.read_csv(trading_file_path(), index_col='DATE')
    
    collection_name = 'Trading_Prices'
    document_name = curr_date.split('-')[0]

    if curr_date not in df.index:
        logger.info(f'trading prices not found for date {curr_date}')
        return
    
    data = df.loc[curr_date].to_dict()
    data['date'] = curr_date
    logger.info(f'data : {data}')
    doc_ref = db.collection(collection_name).document(document_name)
    if doc_ref.get().exists:
        doc_ref.update({
            'data': firestore.ArrayUnion([data])
        })
    else:
        doc_ref.set({
            'data': firestore.ArrayUnion([data])
        })

if __name__ == '__main__':
    
    # push_data(commodity='soyabean', state='madhya pradesh', mandi='ujjain', start_date='2006-02-28', end_date='2006-03-10', info_type=commodity_info_type.PRICES)
    # push_odk(commodity='soyabean', state='telangana', mandi='adilabad', start_date='2006-02-28', end_date='2006-03-10', mongo_collection_name='adilabad')
   
    # push_recommendation(commodity='soyabean', target_state='rajasthan', target_mandi='kota', surrogate_state='madhya pradesh', surrogate_mandi='mahidpur', start_date='2006-06-02', end_date='2006-06-10', rtype=recommendation_type.SHORT)
    # TODO (verify): verify for long term recommendation
    # push_recommendation(commodity='soyabean', state='rajasthan', mandi='kota', surrogate_state='madhya pradesh', surrogate_mandi='mahidpur', start_date='2006-06-02', end_date='2006-06-10', rtype=recommendation_type.LONG)
    
    # pushing prices for current day if already crawled using crawl_trading_prices api
    # from datetime import date
    # push_trading_prices(date.today().strftime('%Y-%m-%d'))
