import os
import sys
import logging
import pandas as pd
import numpy as np

from typing import List, Dict, Set
from selenium import webdriver
from datetime import datetime, timedelta
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service
from selenium.common.exceptions import NoSuchElementException, StaleElementReferenceException

# chrome options for webdriver
chrome_options = webdriver.ChromeOptions()
chrome_options.add_argument('--no-sandbox')
chrome_options.add_argument('--window-size=1420,1080')
# chrome_options.add_argument('--headless')
chrome_options.add_argument('--disable-gpu')

# configuring default logging level to DEBUG
logging.basicConfig(level=logging.INFO)

# getting reference to logger object
logger = logging.getLogger(__name__)

# paths
par_dir = os.path.abspath(os.path.join(__file__, '..', '..', '..', '..'))
data_dir = os.path.join(par_dir, 'data', 'crawler_data')
webdriver_path = os.path.join(par_dir, 'code', 'utilities', 'chromedriver')
mandis_info_path = os.path.join(data_dir, 'mandis_info.csv')

# formatting string as '_' separated and in lowercase
def format_path_component(s: str) -> str:
    return '_'.join(s.split()).lower()

# create raw file path
def raw_file_path(commodity: str, state: str, year: str, month: str) -> str:
    return os.path.join(data_dir, 'raw', format_path_component(commodity), f'{format_path_component(state)}_{year}_{month}.csv')

# create procesed file path
def raw_processed_file_path(commodity: str, state: str, mandi: str, type_prices: bool) -> str:
    type_file = 'prices' if type_prices else 'arrivals' 
    return os.path.join(data_dir, 'raw_processed', format_path_component(commodity), type_file ,f'{format_path_component(state)}_{format_path_component(mandi)}_{type_file}.csv')

months_dict = {
    'january': 1,
    'february': 2,
    'march': 3,
    'april': 4,
    'may': 5,
    'june': 6,
    'july': 7,
    'august': 8,
    'september': 9,
    'october': 10,
    'november': 11,
    'december': 12
}

# TODO(VERIFY): Ensure format of date in date_range '%Y-%m-%d'
# crawl agmarknet for range of dates
def crawl_agmarknet_date_range(commodity: str, state: str, start_date: str, end_date: str) -> None:

    # aggregate date_range by years and months
    agg_date_range: Dict[str, List[str]] = dict()
    for i in pd.date_range(start=start_date, end=end_date):
        # converting format of month
        y, m = i.strftime('%Y-%B').split('-')
        if y not in agg_date_range:
            agg_date_range[y] = set()
        agg_date_range[y].add(m)

    # check if month is already crawled, then avoid crawling it
    for y in agg_date_range:
        # making list of unique month to be crawled for each year in date_range
        # agg_date_range[y] = list(agg_date_range[y])
        
        # if there exists file for next month then we can ignore crawling for current month
        filtered_months = set()
        for m in agg_date_range[y]:
            ny, nm = (datetime.strptime(f'{y}-{m}-01', '%Y-%B-%d') + timedelta(days=31)).strftime('%Y-%B').split('-')
            nfile_path = raw_file_path(commodity=commodity, state=state, year=ny, month=nm)
            if os.path.exists(nfile_path):
                continue
            filtered_months.add(m)
        # sorting month names so that crawling starts in sequence
        agg_date_range[y] = sorted(filtered_months, key=lambda month: months_dict[month.lower()])

    # crawling raw data for each month in a year and processing crawled data
    logger.info(f'aggregated date range: {agg_date_range}')
    for y in agg_date_range:
        for m in agg_date_range[y]:
            crawl_agmarknet(commodity=commodity, state=state, month=m, year=y)
            process_raw(commodity=commodity, state=state, month=m, year=y)

# crawl agmarknet for a given month and save raw files
# format: year - %Y, month - %B
def crawl_agmarknet(commodity: str, state: str, month: str, year: str, retries: int = 5) -> None:
    
    logger.info(f'crawler started')
    logger.info(f'commodity - {commodity}, state - {state}, month - {month}, year - {year}')

    # Constructing file path
    file_path = raw_file_path(commodity=commodity, state=state, year=year, month=month)
    
    # Create file directory if not exists
    os.makedirs(os.path.dirname(file_path), exist_ok=True)

    # Retrying for retries number of iterations
    for i in range(retries):
        logger.info(f'iteration: {i}')
        driver = None
        try:
            logger.debug('instantiating driver')
            driver = webdriver.Chrome(webdriver_path, options=chrome_options)
            driver.get('http://agmarknet.gov.in/PriceAndArrivals/DatewiseCommodityReport.aspx#')
            driver.implicitly_wait(300)
            driver.find_element(by='id', value='cphBody_cboYear').send_keys(int(year))
            driver.find_element(by='id', value='cphBody_cboMonth').send_keys(month)
            driver.find_element(by='id', value='cphBody_cboState').send_keys(state.capitalize())
            driver.find_element(by='id', value='cphBody_cboCommodity').send_keys(commodity.capitalize())
            driver.find_element(by='id', value='cphBody_btnSubmit').click()
            table = driver.find_element(by='id', value='cphBody_gridRecords')
            rows = table.find_elements(By.TAG_NAME, "tr")
            
            # Reading data from crawled page
            st: str = ''
            for row in rows:
                cells = row.find_elements(By.TAG_NAME, "td")
                if not cells:
                    cells = row.find_elements(By.TAG_NAME, "th") 
                for cell in cells:
                    st += cell.text+','
                st+='\n'
            
            # Writing data to file path
            logger.info(f'writing data to file: {file_path}')
            with open(file_path, 'w') as fp:
                fp.write(st)

            # Close driver and break loop
            logger.debug('closing driver')
            driver.close()
            break

        except(NoSuchElementException, StaleElementReferenceException) as e:
            logger.exception("exception caught")
            if driver is not None:
                logger.debug('closing driver on exception')
                driver.close()

    # TODO: Not get succeded even after retries
    if (not os.path.exists(file_path)) or (os.stat(file_path).st_size == 0):
        logger.exception(f'crawl_agmarknet failed after {retries} retries')
        sys.exit(1)

    logger.info(f'crawling completed')

# finding last day corresponding to each month for reindexing crawled raw dataframe 
# for entire month for each mandi
def last_day_of_month(any_day):
    # The day 28 exists in every month. 4 days later, it's always next month
    next_month = any_day.replace(day=28) + timedelta(days=4)
    # subtracting the number of the current day brings us back one month
    return next_month - timedelta(days=next_month.day)

# process raw files
# format: year - %Y, month - %B
def process_raw(commodity: str, state: str, month: str, year: str) -> None:

    file_path = raw_file_path(commodity=commodity, state=state, year=year, month=month)
    
    # Formatting file
    if not os.path.exists(file_path):
        logger.error(f'file_path: [{file_path}] not exists')
        sys.exit(1)
    
    df: pd.DataFrame = pd.read_csv(file_path, warn_bad_lines=False, usecols=[0, 1, 2, 4, 5, 6])
    df.columns = ['MANDINAME', 'DATE', 'ARRIVAL', 'MIN', 'MAX', 'PRICE']
    df.replace('NR', np.nan, inplace = True)
    df['MANDINAME'].ffill(inplace = True)
    df[['ARRIVAL', 'MIN', 'MAX', 'PRICE']] = df[['ARRIVAL', 'MIN', 'MAX', 'PRICE']].apply(pd.to_numeric, errors='coerce')
    df.drop_duplicates(['MANDINAME', 'DATE'], inplace = True)
    df['MANDINAME'] = df['MANDINAME'].str.upper()

    # processing file
    mandis_info = pd.read_csv(mandis_info_path)
    df['DATE'] = pd.to_datetime(df['DATE'], format='%d/%m/%Y', errors='coerce')
    df.dropna(subset=['DATE'], inplace=True)
    df.sort_values(['DATE', 'MANDINAME'], inplace=True)
    df = pd.merge(df, mandis_info, how='left', left_on=['MANDINAME'], right_on=['MANDINAME'])
    df = df[['MANDINAME', 'DATE', 'ARRIVAL', 'MIN', 'MAX', 'PRICE', 'MANDICODE']]

    # separating mandi-wise data
    for mandi in df['MANDINAME'].unique():
        logger.info(f'processing started for state: {state}, mandi : {mandi}')
        
        # creating df for a particular mandi
        mandi_df = df.loc[df['MANDINAME'] == mandi, ['DATE', 'ARRIVAL', 'PRICE']].copy()
        mandi_df.reset_index(inplace=True, drop=True)
        
        # extending dataframe with np.nan for dates missed between min and max day
        mandi_df['DATE'] = pd.to_datetime(mandi_df['DATE'])
        mandi_df.drop_duplicates(subset=['DATE'], keep='last', inplace=True)
        mandi_df.set_index('DATE', inplace=True)
        # since processing is performed month wise finding the first and last day of the month for reindexing
        # min_day, max_day = mandi_df.index.min(), mandi_df.index.max()
        min_day = datetime.strptime(f'{year}-{month}', '%Y-%B')
        max_day = last_day_of_month(min_day)
        logger.info(f'reindexing for month - {month}, year - {year}')
        mandi_df = mandi_df.reindex(pd.date_range(min_day, max_day), fill_value=np.nan)
        mandi_df['DATE'] = mandi_df.index
        mandi_df['PRICE'].replace(0.0, np.nan, inplace=True)
        mandi_df['ARRIVAL'].replace(0.0, np.nan, inplace=True)
        
        # seperating df for prices and arrivals for a particular mandi
        arrival_df = mandi_df[['DATE', 'ARRIVAL']]
        price_df = mandi_df[['DATE', 'PRICE']]

        # Format mandi name
        mandi = format_path_component(mandi)

        # merging with existing arrival and prices dataframe
        price_file_path = raw_processed_file_path(commodity=commodity, state=state, mandi=mandi, type_prices=True)
        arrival_file_path = raw_processed_file_path(commodity=commodity, state=state, mandi=mandi, type_prices=False)
        
        # reading crawled prices and arrivals df if present
        old_price_df = pd.DataFrame()
        if os.path.exists(price_file_path):
            old_price_df = pd.read_csv(price_file_path, parse_dates=['DATE'], usecols=['DATE', 'PRICE'])

        old_arrival_df = pd.DataFrame()
        if os.path.exists(arrival_file_path):
            old_arrival_df = pd.read_csv(arrival_file_path, parse_dates=['DATE'], usecols=['DATE', 'ARRIVAL']) 
        
        # merging prices and arrivals df with crawled prices and arrivals df
        logger.info('merging newly and old prices data')
        new_price_df = pd.concat([old_price_df, price_df], ignore_index=True)
        logger.info('merging newly and old arrival data')
        new_arrival_df = pd.concat([old_arrival_df, arrival_df], ignore_index=True)

        # dropping duplicate dates from price and arrival df, keeping last entry
        # previously value for a date might be null but later it got updated
        new_price_df.drop_duplicates(subset=['DATE'], keep='last', inplace=True)
        new_arrival_df.drop_duplicates(subset=['DATE'], keep='last', inplace=True)

        # to ensure that crawling and formatting of any intermediate dates can be performed
        # sorting prices and arrival df by date
        new_price_df.sort_values(by=['DATE'], inplace=True)
        new_arrival_df.sort_values(by=['DATE'], inplace=True)

        # Create file directory if not exists
        os.makedirs(os.path.dirname(price_file_path), exist_ok=True)
        os.makedirs(os.path.dirname(arrival_file_path), exist_ok=True)
        
        # saving merged prices and arrival df
        logger.info('saving newly prices data')
        new_price_df['PRICE'] = new_price_df['PRICE'].astype(int)
        new_price_df.to_csv(price_file_path, index=False)
        logger.info('saving newly arrivals data')
        new_arrival_df['ARRIVAL'] = new_arrival_df['ARRIVAL'].astype(int)
        new_arrival_df.to_csv(arrival_file_path, index=False)

if __name__ == '__main__':
    crawl_agmarknet_date_range('Soyabean', 'Rajasthan', start_date='2006-05-01', end_date='2006-12-31')
