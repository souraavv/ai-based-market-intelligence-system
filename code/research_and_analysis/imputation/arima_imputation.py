from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import math
import pmdarima as pm
import matplotlib.pyplot as plt
from enum import Enum

import os
import logging
import sys


class commodity_info_type(Enum):
    PRICES = 1
    ARRIVALS = 2

# configuring default logging level to DEBUG
logging.basicConfig(level=logging.INFO)

# getting reference to logger object
logger = logging.getLogger(__name__)

# paths
par_dir = os.path.abspath(os.path.join(__file__, '..', '..', '..', '..'))
data_dir = os.path.join(par_dir, 'data')

# formatting string as '_' separated and in lowercase
def format_path_component(path: str) -> str:
    return '_'.join(path.split()).lower()

# create procesed file path
def raw_processed_file_path(commodity: str, state: str, mandi: str, type_prices: commodity_info_type) -> str:
    type_file = 'prices' if type_prices == commodity_info_type.PRICES else 'arrivals' 
    return os.path.join(data_dir, 'crawler_data', 'raw_processed', format_path_component(commodity), type_file ,f'{format_path_component(state)}_{format_path_component(mandi)}_{type_file}.csv')

# create procesed file path
def imputed_file_path(commodity: str, state: str, mandi: str, type_prices: bool) -> str:
    type_file = 'prices' if type_prices == commodity_info_type.PRICES else 'arrivals' 
    return os.path.join(data_dir, 'imputed_data', format_path_component(commodity), type_file ,f'{format_path_component(state)}_{format_path_component(mandi)}_{type_file}.csv')

# create procesed file path
def odk_file_path(commodity: str, state: str, mandi: str) -> str:
    return os.path.join(data_dir, 'crawler_data', 'odk', format_path_component(commodity),  f'{format_path_component(state)}_{format_path_component(mandi)}_prices.csv')

# helper function to perform imputation using auto_arima
def helper_arima_imputation(n, x, y):
    logger.info(f'interpolation started')
    value = {} # Dictionary
    for i in range(len(x)):
        value[x[i]] = y[i]
    yi = [] # List
    i = 0
    while i < n:
        if i in value.keys():
            yi += [value[i]]
            i += 1
        else:
            train_data = np.array(yi)
            model = pm.arima.auto_arima(train_data, start_p=1, max_p=2, start_q=1, max_q=2, d=None, max_d=1, 
                                    suppress_warnings=True, seasonal=False, stepwise=True, error_action="ignore")
            num = 0
            while i < n and i not in value.keys():
                i += 1
                num += 1
            predictions = model.predict(num)
            yi = list(yi + predictions.tolist())
    logger.info(f'interpolation finished')
    return yi

# arima imputation of mandi uptil end_date
# NOTE: if end_date is smaller than max_date of already imputed dataframe, dataframe would be shrinked to end_date
def arima_imputation(commodity: str, state: str, mandi: str, end_date: str, info_type: commodity_info_type, start_date: str = '2006-01-01') -> None:
    start_date, end_date = pd.to_datetime(start_date), pd.to_datetime(end_date)

    # names of column in a dataframe
    date_col_name, missing_col_name = 'DATE', 'ARRIVAL'
    if info_type == commodity_info_type.PRICES:
        missing_col_name = 'PRICE'
    
    logger.info(f'imputation started')
    logger.info(f'commodity - {commodity}, state - {state}, mandi - {mandi}, till_date - {end_date}')

    ## reading imputed_mandi_df from saved file, if it exists
    file_path = imputed_file_path(commodity=commodity, state=state, mandi=mandi, type_prices=info_type)
    imputed_mandi_df = pd.DataFrame()

    # file_path may not exist, updating imputed_mandi_df if file exists
    if os.path.exists(file_path):
        imputed_mandi_df = pd.read_csv(
            file_path,
            parse_dates=['DATE'])

    ## extending imputed mandi dataframe with crawled values to save imputation cost

    processed_file_path = raw_processed_file_path(commodity=commodity, state=state, mandi=mandi, type_prices=info_type)
    if not os.path.exists(processed_file_path):
        logger.error(f'processed raw data missing [{processed_file_path}], imputation must be done after processing')
        sys.exit(1)
    
    processed_mandi_df = pd.read_csv(
        processed_file_path, 
        parse_dates=['DATE'])
    
    # setting 'IMPUTED' column in processed_mandi_df
    processed_mandi_df['IMPUTED'] = 1
    processed_mandi_df.loc[processed_mandi_df[missing_col_name].notnull(), 'IMPUTED'] = 0
    
    # joining imputed data with processed data
    logger.info(f'merging imputed data with processed data')
    imputed_mandi_df = pd.concat([imputed_mandi_df, processed_mandi_df], ignore_index=True)
    
    # dropping nan values
    imputed_mandi_df.dropna(subset=[missing_col_name], inplace=True)
    
    # drop duplicates values corresponding to date keeping the last
    # ensures that we prefer actual values rather than imputed values
    imputed_mandi_df.drop_duplicates(subset=['DATE'], keep='last', inplace=True)
    
    # keeping last of duplicate values may distory sequence, sorting time-series by date
    imputed_mandi_df.sort_values(by=['DATE'], inplace=True)


    ## extending imputed mandi dataframe with odk values to save imputation cost

    forms_file_path = odk_file_path(commodity=commodity, state=state, mandi=mandi)
    # odk forms data only contain prices and needs to be merged only if it exists
    if info_type == commodity_info_type.PRICES and os.path.exists(forms_file_path):
        logger.info(f'odk forms data found at [{forms_file_path}]')
        
        forms_df = pd.read_csv(
            forms_file_path,
            parse_dates=['DATE'])
        
        # setting 'IMPUTED' column in forms_df
        forms_df['IMPUTED'] = 0

        imputed_mandi_df = pd.concat([imputed_mandi_df, forms_df], ignore_index=True)

        # computing mean of only non-imputed values
        # return appropriate series with 'IMPUTED' as well as missing_col
        # TODO: return entire row with 'IMPUTED' values as well
        def imputed_forms_custom_mean(group):
            return pd.Series({
                missing_col_name: group.loc[group['IMPUTED']==0, missing_col_name].mean().astype(int),
                'IMPUTED': 0
            })
        
        imputed_mandi_df = imputed_mandi_df.groupby(by='DATE').apply(imputed_forms_custom_mean).reset_index()

    ## imputation of missing values

    # extend time series to till_date by filling in np.nan values
    imputed_mandi_df.set_index('DATE', inplace=True)
    imputed_mandi_df = imputed_mandi_df.reindex(pd.date_range(start_date, end_date), fill_value=np.nan)
    
    # marking null values with 1 which are to be imputed
    imputed_mandi_df['IMPUTED'].fillna(1, inplace=True)
    imputed_mandi_df['DATE'] = imputed_mandi_df.index
    imputed_mandi_df.reset_index(inplace=True, drop=True)

    # saving imputed_list and dates_list, used later when creating updated_imputed_mandi_df
    imputed_list = imputed_mandi_df['IMPUTED'].tolist()
    dates_list = imputed_mandi_df[date_col_name].tolist()
    cnt_start_missing_values = 0
    first_not_nan_value = None

    for idx, row in imputed_mandi_df.iterrows():
        if pd.isnull(row[missing_col_name]):
            cnt_start_missing_values += 1
        else:
            first_not_nan_value = row[missing_col_name]
            break
    
    if (cnt_start_missing_values > 0) and (first_not_nan_value is None):
        logger.exception(f'all values are nan, imputation cannot be performed')
        sys.exit(1)
    
    # backfilling first non-null value backwards
    while cnt_start_missing_values >= 0:
        imputed_mandi_df.loc[cnt_start_missing_values, missing_col_name] = first_not_nan_value
        cnt_start_missing_values -= 1
    
    imputed_mandi_df['id'] = imputed_mandi_df.index
    n = imputed_mandi_df.shape[0]
    
    imputed_mandi_df.dropna(subset=[missing_col_name], inplace=True)
    x = imputed_mandi_df['id'].values
    y = imputed_mandi_df[missing_col_name].values

    yi = helper_arima_imputation(n, x, y)
    
    # creating updated imputed mandi df
    updated_imputed_mandi_df = pd.DataFrame({
        date_col_name: dates_list,
        missing_col_name: yi,
        'IMPUTED': imputed_list
    })

    # predicted prices cannot be zero or negative, so converting them to nan and performing linear imputed
    # updated_imputed_mandi_df.loc[updated_imputed_mandi_df[missing_col_name] <= 0, 'IMPUTED'] = 0
    updated_imputed_mandi_df.loc[updated_imputed_mandi_df[missing_col_name] <= 0, missing_col_name] = np.nan
    updated_imputed_mandi_df[missing_col_name] = updated_imputed_mandi_df[missing_col_name].interpolate(method="linear")

    # create file directory if not exists
    os.makedirs(os.path.dirname(file_path), exist_ok=True)

    # saving updated_imputed_mandi_df
    logger.info(f'saving updated_imputed_mandi_df at [{file_path}]')
    # converting type of all columns except date to int
    updated_imputed_mandi_df[missing_col_name] = updated_imputed_mandi_df[missing_col_name].astype(int)
    updated_imputed_mandi_df['IMPUTED'] = updated_imputed_mandi_df['IMPUTED'].astype(int)
    updated_imputed_mandi_df.to_csv(file_path, index=False)

# impute input_file_path and save to output_file_path
# NOTE (verify): assumption missing values in missing_col_name should be nan
def impute_file(input_file_path: str, output_file_path: str):
    # reading dataframe
    logger.info(f'reading input dateframe from [{input_file_path}]')
    date_col_name, missing_col_name = 'DATE', 'PRICE'
    input_df = pd.read_csv(input_file_path, usecols=[date_col_name, missing_col_name])

    # saving imputed_list and dates_list, used later when creating updated_imputed_mandi_df
    dates_list = input_df[date_col_name].tolist()
    cnt_start_missing_values = 0
    first_not_nan_value = None

    for idx, row in input_df.iterrows():
        if pd.isnull(row[missing_col_name]):
            cnt_start_missing_values += 1
        else:
            first_not_nan_value = row[missing_col_name]
            break
    
    if (cnt_start_missing_values > 0) and (first_not_nan_value is None):
        logger.exception(f'all values are nan, imputation cannot be performed')
        sys.exit(1)
    
    # backfilling first non-null value backwards
    while cnt_start_missing_values >= 0:
        input_df.loc[cnt_start_missing_values, missing_col_name] = first_not_nan_value
        cnt_start_missing_values -= 1
    
    input_df['id'] = input_df.index
    n = input_df.shape[0]
    
    input_df.dropna(subset=[missing_col_name], inplace=True)
    x = input_df['id'].values
    y = input_df[missing_col_name].values

    # performing arima interpolation
    yi = helper_arima_imputation(n, x, y)
    
    # creating updated imputed mandi df
    output_df = pd.DataFrame({
        date_col_name: dates_list,
        missing_col_name: yi,
    })

    # predicted prices cannot be zero or negative, so converting them to nan and performing linear imputed
    output_df.loc[output_df[missing_col_name] <= 0, missing_col_name] = np.nan
    output_df[missing_col_name] = output_df[missing_col_name].interpolate(method="linear")

    # create file directory if not exists
    logger.info(f'creating output dir path {os.path.dirname(output_file_path)}, it not exists')
    os.makedirs(os.path.dirname(output_file_path), exist_ok=True)

    # converting type of all columns except date to int
    output_df[missing_col_name] = output_df[missing_col_name].astype(int)
    
    # saving output_df
    logger.info(f'saving output dataframe at [{output_file_path}]')
    output_df.to_csv(output_file_path, index=False)

if __name__ == '__main__':
    arima_imputation(commodity='soyabean', state='telangana', mandi='adilabad', end_date='2006-03-31', info_type=commodity_info_type.PRICES)
    # arima_imputation(commodity='soyabean', state='madhya pradesh', mandi='ujjain', end_date='2006-03-31', info_type=commodity_info_type.ARRIVALS)
    # impute_file(input_file_path=, output_file_path=)
