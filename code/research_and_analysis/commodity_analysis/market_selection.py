import os
import re
import sys
import logging
import pandas as pd
import time

from typing import List, Tuple
from selenium import webdriver
from datetime import datetime
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import Select, WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.keys import Keys

# chrome options for webdriver
chrome_options = webdriver.ChromeOptions()
chrome_options.add_argument('--no-sandbox')
chrome_options.add_argument('--window-size=1420,1080')
chrome_options.add_argument('--headless')
chrome_options.add_argument('--disable-gpu')

# configuring default logging level to DEBUG
logging.basicConfig(level=logging.DEBUG)

# getting reference to logger object
logger = logging.getLogger(__name__)

# paths
par_dir = os.path.abspath(os.path.join(__file__, '..', '..', '..', '..'))
data_dir = os.path.join(par_dir, 'data')
webdriver_path = os.path.join(par_dir, 'code', 'utilities', 'chromedriver')
markets_info_path = os.path.join(data_dir, 'crawler_data', 'markets_covered.csv')

# import crawl_agmarknet_date_range
sys.path.append(os.path.join(par_dir, 'code', 'crawler', 'agmarknet_crawler'))
from wholesale_crawler import crawl_agmarknet_date_range

# formatting string as '_' separated and in lowercase
def format_path_component(s: str) -> str:
    return '_'.join(s.split()).lower()

# create total count save path
def total_count_save_path(commodity: str, date_from: str, date_to: str) -> str:
    return os.path.join(data_dir, 'commodity_analysis_data', f'./total_count_{commodity.lower()}_{date_from.replace("-", "_")}_{date_to.replace("-", "_")}.csv')

# create procesed file path
def raw_processed_file_path(commodity: str, state: str, mandi: str, type_prices: bool) -> str:
    type_file = 'prices' if type_prices else 'arrivals' 
    return os.path.join(data_dir, 'crawler_data', 'raw_processed', format_path_component(commodity), type_file ,f'{format_path_component(state)}_{format_path_component(mandi)}_{type_file}.csv')

def crawl_market_total_count(commodity: str, state: str, district: str, market: str, date_from: str, date_to: str, retries: int = 5) -> int:
    logger.debug(f'{date_from}-{date_to}')
    select_ids = ['ddlArrivalPrice', 'ddlCommodity', 'ddlState', 'ddlDistrict', 'ddlMarket']
    input_ids = ['txtDate', 'txtDateTo']
    for i in range(retries):
        logger.debug(f"Iteration: {i}")
        driver = None
        try:
            logger.debug('instantiating driver')
            driver = webdriver.Chrome(webdriver_path, options=chrome_options)
            driver.get("https://agmarknet.gov.in/")
            time.sleep(4)
            driver.implicitly_wait(300)

            # Select Choices
            for id, value in zip(select_ids, ['Price', commodity, state, district, market]):
                # Explicit wait until value appears in element text
                WebDriverWait(driver, 10).until(
                    EC.visibility_of_element_located((By.ID, id)))
                WebDriverWait(driver, 20).until(
                    EC.text_to_be_present_in_element((By.ID, id), value))
                # Select desired option from select element or input desired text into input box
                element = Select(driver.find_element_by_id(id))

                element.select_by_visible_text(value)

            # waiting for page to reload, otherwise returns stale element exception
            time.sleep(4)

            # Input Choices
            for id, value in zip(input_ids, [date_from, date_to]):
                # Explicit wait until element appears in DOM
                WebDriverWait(driver, 10).until(
                    EC.visibility_of_element_located((By.ID, id)))
                # Input desired text into input box and clear prior values
                element = driver.find_element_by_id(id)
                element.clear()
                element.send_keys(value)
            go_btn = driver.find_element_by_id('btnGo')
            go_btn.send_keys(Keys.RETURN)

            WebDriverWait(driver, 10).until(
                EC.visibility_of_element_located((By.ID, 'cphBody_LabComName')))
            span_element_text: str = driver.find_element_by_id('cphBody_LabComName').text
            found = re.search(r".*?Total-(\d*).*", span_element_text)
            
            logger.debug('closing driver')
            driver.close()
            if found is not None:
                total = found.group(1)
                logger.debug(f'total count - {total}')
                return int(total)
            else:
                return 0
        except Exception as e:
            logger.exception("caught exception (crawl_total_count)\n")
            if driver is not None:
                logger.debug('closing driver on exception')
                driver.close()
    return 0

# crawl total available data count percentage for all mandis in a given state
# Note: all input string parameters are capitalize, dates are in format of '%d-%b-%Y'
def crawl_states_total_count(states: List[str], commodity: str, date_from: str, date_to: str) -> pd.DataFrame:
    select_choices = []
    markets_df = pd.read_csv(markets_info_path)
    for state in states:
        state_df = markets_df[markets_df['State'] == state]
        for _, row in state_df.iterrows():
            select_choices.append([commodity, state, row['District Name'], row['Market']])

    total = (datetime.strptime(date_to, '%d-%b-%Y') - datetime.strptime(date_from, '%d-%b-%Y')).days
    logger.info(f'total days between {date_from} - {date_to}: {total}')

    total_count_data = []
    for choice in select_choices:
        logger.debug(f'crawling started for {choice[:]}')
        start_time = time.time()

        total_count = crawl_market_total_count(commodity=choice[0], state=choice[1], district=choice[2], market=choice[3], date_from=date_from, date_to=date_to)
        percent_total_count = int(total_count/total * 100)
        choice.append(percent_total_count)

        # TODO: saving crawled infomation in total_count_bkp file to prevent crawling of market if its already crawled
        line = ','.join([str(c) for c in choice]) + '\n'
        with open(os.path.join('/', 'tmp', 'total_count_bkp.csv'), 'a') as f:
            f.write(line)

        total_count_data.append(choice)

        end_time = time.time()
        logger.debug(f'total time taken to crawl {choice[:]} {(end_time - start_time) / 60}')

    total_count_df = pd.DataFrame(total_count_data, columns=['Commodity', 'State', 'District', 'Market', 'Percentage'])
    
    return total_count_df

# finding maximum continuous missing days from the input series
def max_continuous_missing(data_series: pd.Series) -> int:
    # Step 1: Create a mask to identify missing values (NaNs)
    missing_mask = data_series.isnull()

    # Step 2: Use 'cumsum()' to create groups for continuous missing values
    groups = missing_mask.ne(missing_mask.shift()).cumsum()

    # Step 3: Use 'groupby()' to get the maximum chunk of continuous missing values
    max_chunk = data_series[missing_mask].groupby(groups).size().max()

    return max_chunk


# filter markets based on percentage of available data and maximum continuous missing days
# date_from and date_to dates are in the format of '%Y-%m-%d'
def filter_markets(states: List[str], commodity: str, date_from: str, date_to: str, percent_data: int, continuous_missing: int) -> List[Tuple[str, str]]:
    save_path = total_count_save_path(commodity, date_from, date_to)
    # reading already crawled total count's file
    if os.path.exists(save_path):
        logger.info(f'reading already crawled total counts file from [{save_path}]')
        total_count_df = pd.read_csv(save_path)
    else:
        logger.info(f'crawling and saving total counts to file at [{save_path}]')

        # crawl and save data to local disk
        total_count_df: pd.DataFrame = crawl_states_total_count(states=states, commodity=commodity, date_from=datetime.strptime(date_from, '%Y-%m-%d').strftime('%d-%b-%Y'), date_to=datetime.strptime(date_to, '%Y-%m-%d').strftime('%d-%b-%Y'))
        
        # saving total_count_df onto disk
        logger.info(f'saving crawled data to [{save_path}]')
        
        # creating directory for save_path if not exists
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        total_count_df.to_csv(save_path, index=False)

    # filter markets based on available data
    total_count_df: pd.DataFrame = total_count_df.loc[total_count_df['Percentage'] > percent_data]
    
    # check total_count_df is empty
    if total_count_df.empty:
        logger.info(f'no mandis satisfying percent available criteria of {percent_data}')

    # crawl data for filtered states
    filtered_states: List[str] = total_count_df['State'].unique().tolist()
    for state in filtered_states:
        crawl_agmarknet_date_range(commodity=commodity.capitalize(), state=state.capitalize(), start_date=date_from, end_date=date_to)

    filtered_markets: List[Tuple[str, str]] = []

    # filter markets based on max continuous missing days
    for _, row in total_count_df.iterrows():
        processed_file_path = raw_processed_file_path(commodity=commodity, state=row['State'], mandi=row['Market'], type_prices=True)
        if not os.path.exists(processed_file_path):
            continue
        price_df = pd.read_csv(processed_file_path)
        max_missing_days = max_continuous_missing(price_df['PRICE'])
        if max_missing_days >= continuous_missing:
            continue
        # append state and market with continuous missing days lesser than continuous_missing count
        filtered_markets.append((row['State'], row['Market']))

    # check total_count_df is empty
    if len(filtered_markets) == 0:
        logger.info(f'no mandis satisfying continuous missing days criteria of {continuous_missing}')

    return filtered_markets

if __name__ == '__main__':
    # filtered_markets = filter_markets(states=['Rajasthan'], commodity='Soyabean', date_from='2006-01-01', date_to='2006-12-31', percent_data=10, continuous_missing=60)
    
