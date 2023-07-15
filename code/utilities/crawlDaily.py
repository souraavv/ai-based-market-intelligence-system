from datetime import datetime, timedelta

import pandas as pd
 
from fillMissingData import interpolateFile, mergeNansWithInterpolatedData, appendInterpolatedData
from liveFormatWholesaleData import formatData
from liveProcessData import processWholesaleData
from liveSeparateData import seperateAGWholesaleData
from liveWholesaleCrawler import extractWholesaleData
from code.utilities.push_firebase import push_data, pushDailyODK, push_longterm_recommendation, db, push_trading_prices
from xforms_data_script import crawlDataODK
from crawlTradingPrices import crawl_trading_prices
from tcnMultivariateRangeBased2 import finalModelCode  # FIXME change file name from 2 -> simple
from longtermforecast.tcnLongTermMultivariate import longTermModel 

# FIXME : for adilabad, add telangana
dictAG = {
    'COMMODITY':{
        'Soyabean': ['Madhya Pradesh', 'Rajasthan', 'Maharashtra', 'Telangana']
    }
}

# FIXME : add adilabad.
arimaInterpolateFiles = ['RAJASTHAN_KOTA_Price.csv', 'RAJASTHAN_BHAWANI MANDI_Price.csv', 
                        'MADHYA PRADESH_MAHIDPUR_Price.csv', 'MADHYA PRADESH_GANJBASODA_Price.csv', 'MADHYA PRADESH_RATLAM_Price.csv', 
                        'MADHYA PRADESH_SONKATCH_Price.csv', 'MAHARASHTRA_NAGPUR_Price.csv', 'MAHARASHTRA_LATUR_Price.csv', 'TELANGANA_ADILABAD_Price.csv']

# Crawl data from the last day till which data is present to the current date - 1.
df = pd.read_csv(f'/home/baadalvm/Playground_Ronak_Sourav/Practice/Data/PlottingData/SOYABEAN/ARIMA_Interpolated_Data/MAHARASHTRA_NAGPUR_Price.csv')
df['DATE'] = pd.to_datetime(df['DATE'], format='%Y-%m-%d') # Converting 'DATE' column to datetime type
# start from last day for which data is crawled + 1
start = max(df['DATE']).to_pydatetime() + timedelta(days=1)
# end points to current day but data is not required to be crawled for current day, will be excluded in the range of for loop
end = datetime.today()
dateList = [start + timedelta(days=x) for x in range(0, (end-start).days)]

trading_prices_fpath = '/home/baadalvm/Playground_Ronak_Sourav/Practice/Data/TradingData/trading_prices.csv'


for currDate in dateList:
    currDate = currDate.strftime("%Y-%m-%d")
    print (f'currDate : {currDate}')
    # -----Crawl data from Agmarknet
    extractWholesaleData(dictAG, currDate)
    formatData()
    processWholesaleData(dictAG)
    seperateAGWholesaleData(dictAG, currDate)
    mergeNansWithInterpolatedData(arimaInterpolateFiles, currDate)
    # ----Crawl data from ODK FIXME I'm off because you shifted to nagpur, please turn on when move to Adilabad.
    crawlDataODK(currDate)
    # Daily_odk_data : new 13-dec
    pushDailyODK(currDate)
    # ----For arima interpolation.
    interpolateFile(arimaInterpolateFiles)
    # ----Append interpolated files data to error models data
    appendInterpolatedData(['MAHARASHTRA_NAGPUR_Price.csv', 'MAHARASHTRA_LATUR_Price.csv'], 4, currDate)
    # ----Push data to firebase
    push_data(arimaInterpolateFiles, currDate)
    # ----TCN Multivariate Range Based Model
    finalModelCode()
    lrt = longTermModel()
    if lrt != -1:
       push_longterm_recommendation(db, lrt)
    # ---Trading Prices
    crawl_trading_prices(trading_prices_fpath)
    push_trading_prices(trading_prices_fpath, currDate)
