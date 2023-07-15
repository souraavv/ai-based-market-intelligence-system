from datetime import datetime, timedelta
from liveWholesaleCrawler import extractWholesaleData
from liveFormatWholesaleData import formatData
from liveProcessData import processWholesaleData
from liveSeparateData import seperateAGWholesaleData
from fillMissingData import interpolateFile, mergeNansWithInterpolatedData
import pandas as pd

dictAG = {
    'COMMODITY':{
        'Soyabean': ['Madhya Pradesh', 'Rajasthan']
    }
}

arimaInterpolateFiles = ['RAJASTHAN_KOTA_Price.csv', 'RAJASTHAN_BHAWANI MANDI_Price.csv', 'MADHYA PRADESH_MAHIDPUR_Price.csv', 'MADHYA PRADESH_GANJBASODA_Price.csv', 'MADHYA PRADESH_RATLAM_Price.csv', 'MADHYA PRADESH_SONKATCH_Price.csv']

# Crawl data from the last day till which data is present to the current date - 1.
df = pd.read_csv(f'/home/baadalvm/Playground_Ronak_Sourav/Practice/Data/PlottingData/SOYABEAN/ARIMA_Interpolated_Data/RAJASTHAN_KOTA_Price.csv')
df['DATE'] = pd.to_datetime(df['DATE'], format='%Y-%m-%d') # Converting 'DATE' column to datetime type
# start from last day for which data is crawled + 1

start = max(df['DATE']).to_pydatetime() + timedelta(days=1)
# end points to current day but data is not required to be crawled for current day, will be excluded in the range of for loop
end = datetime.today()
dateList = [start + timedelta(days=x) for x in range(0, (end-start).days)]

for currDate in dateList:
    currDate = currDate.strftime("%Y-%m-%d")
    print (f'currDate : {currDate}')
    # -----Crawl data from Agmarknet
    extractWholesaleData(dictAG, currDate)
    formatData()
    processWholesaleData(dictAG)
    seperateAGWholesaleData(dictAG, currDate)
    mergeNansWithInterpolatedData(arimaInterpolateFiles, currDate)
    # ----For arima interpolation.
    interpolateFile(arimaInterpolateFiles)
    