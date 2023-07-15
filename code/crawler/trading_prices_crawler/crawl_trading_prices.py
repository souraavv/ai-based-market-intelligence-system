#!/usr/bin/python3
import os
import logging
import requests
import pandas as pd
from datetime import date
from bs4 import BeautifulSoup

from pprint import pprint

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

def trading_file_path() -> str:
    return os.path.join(data_dir, 'crawler_data', 'trading', 'prices.csv')

# NOTE: Can only crawl prices for current day, 
# NOTE: cannot be used to crawl historical commodity trading prices
def crawl_trading_prices():
    curr_date = date.today().strftime("%Y-%m-%d")
    url = 'https://tradingeconomics.com/commodities'
    logger.info(f'crawling trading prices for {curr_date} from {url}')
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/68.0.3440.84 Safari/537.36',
        'accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,image/apng,*/*;q=0.8',
    }
    r = requests.get(url, headers=headers)

    soup = BeautifulSoup(r.text, 'html.parser')

    trSoybean = soup.find('tr', {'data-symbol': 'S 1:COM'})
    tdPriceSoybean = trSoybean.find('td', id = 'p').text
    unitSoybean = trSoybean.find('div').text
    priceSoybean = float(tdPriceSoybean.strip()) / 100.00

    trCotton = soup.find('tr', {'data-symbol': 'CT1:COM'})
    tdPriceCotton = trCotton.find('td', id = 'p').text
    unitCotton = trCotton.find('div').text
    priceCotton = float(tdPriceCotton.strip()) / 100.00

    # API calls for USD to rupees conversion
    API_KEY="770b2ebfcd3b71b9284a7f011e2e0beb"
    conversionFactorResponse = requests.get(f'http://api.exchangeratesapi.io/v1/latest?access_key={API_KEY}&symbols=INR,USD')
    conversionFactorJson = conversionFactorResponse.json()
    conversionFactor = float(conversionFactorJson["rates"]["INR"]) / float(conversionFactorJson["rates"]["USD"])

    # Converting prices to INR/KG
    priceSoybeanINRPerKg = int(((priceSoybean / 56) * 2.20462262) * conversionFactor * 100)
    priceCottonINRPerKg = int((priceCotton * 2.20462262) * conversionFactor * 100)

    # Creating a new dataframe if not present, append to existing if already present
    
    data = {
        'DATE' : curr_date,
        'SOYABEAN' : priceSoybeanINRPerKg,
        'COTTON' : priceCottonINRPerKg
    }

    # saving crawled data
    save_file_path = trading_file_path()
    logger.info(f"writing crawled data for date [{curr_date}] to dataframe [{save_file_path}]")
    
    save_df = pd.DataFrame()

    # reading save_df if already exists
    if os.path.exists(save_file_path):
        save_df = pd.read_csv(save_file_path)
    
    # append crawled data to save_df
    # preventing double append data for the same day
    if data['DATE'] not in save_df['DATE'].values:
        save_df = save_df.append(data, ignore_index=True)
        save_df.to_csv(save_file_path, index=False)
    else:
        logger.info(f"entry of crawled data for date [{curr_date}] in [{save_file_path}] already exists")

if __name__ == '__main__':
    # crawling trading prices for current date
    crawl_trading_prices()
