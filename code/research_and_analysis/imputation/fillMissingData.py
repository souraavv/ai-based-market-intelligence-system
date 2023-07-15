from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import math
import pmdarima as pm
import matplotlib.pyplot as plt

def arima_interpolation(xi, x, y):
    value = {} # Dictionary
    for i in range(len(x)):
        value[x[i]] = y[i]
    yi = [] # List
    i = 0
    while i < xi:
        
        if i in value.keys():
            yi += [value[i]]
            i += 1
        else:
            train_data = np.array(yi)
            model = pm.arima.auto_arima(train_data, start_p=1, d=None, start_q=1, max_p=2, max_d=1, max_q=2,
                                    suppress_warnings=True, seasonal=False, stepwise=True, error_action="ignore")
            num = 0
            while i < xi and i not in value.keys():
                i += 1
                num += 1
            predictions = model.predict(num)
            yi = list(yi + predictions.tolist())
    return yi

def interpolateFile(files):
    print (f"-------- Started with Interpolation --------- ")
    for f in files:
        df = pd.read_csv(f'/home/baadalvm/Playground_Ronak_Sourav/Practice/Data/PlottingData/SOYABEAN/Nans_Data_Temp/{f}')
        cols = df.columns
        startDate = min(df['DATE'])
        startYear = startDate[:4]
        col0 = cols[0] # 'Date'
        col1 = cols[1] # 'Price'
        x = 0
        dates = df['DATE'].tolist()
        for index, row in df.iterrows():
            if math.isnan(row[col1]):
                x += 1
            else:
                val = row[col1]
                break
        
        while x >= 0:
            df.loc[x, col1] = val
            x -= 1

        df.reset_index(inplace=True)
        df['x'] = [i for i in range(df.shape[0])] # add column 'x' to df, 0...(nrows-1)
        xi = df.shape[0] # Rows in df
        df = df[df[cols[1]].notna()] # price is not na
        x = np.array(df['x']) # out of 0...(nrows-1), which values has non zero price
        y = np.array(df[cols[1]]) # Price
        yp = None
        yi = arima_interpolation(xi, x, y)
        df1 = pd.DataFrame({col1: yi})
        df1[col0] = dates
        df1.loc[df1[col1] < 0, col1] = np.nan # Predicted price cannot be negative
        df1[col1] = df1[col1].interpolate(method="linear") # Linear interpolation for np.nan values
        col0, col1 = col1, col0
        df1 = df1[cols]
        df1.to_csv(f'/home/baadalvm/Playground_Ronak_Sourav/Practice/Data/PlottingData/SOYABEAN/ARIMA_Interpolated_Data/{f}', index=False)
        print (f'Filled missing values for file {f}')
    print (f"---------- Finished Interpolation ----------- ")

def mergeNansWithInterpolatedData(files, currDate):
    currDate = datetime.strptime(currDate, '%Y-%m-%d')
    prevDate = (currDate - timedelta(days=1)).strftime('%Y-%m-%d')
    for f in files:
        df1 = pd.read_csv(f'/home/baadalvm/Playground_Ronak_Sourav/Practice/Data/PlottingData/SOYABEAN/ARIMA_Interpolated_Data/{f}')
        df2 = pd.read_csv(f'/home/baadalvm/Playground_Ronak_Sourav/Practice/Data/PlottingData/SOYABEAN/Nans_Data/{f}')
        # Taking data uptil previous date from ARIMA_Interpolated_Data
        df1['DATE'] = pd.to_datetime(df1['DATE'], format='%Y-%m-%d')
        df1 = df1.loc[df1['DATE'] <= prevDate]
        # Taking data from current day onwards from Nans_Data
        # Nans_Data is final processed output from crawling agmarknet data, either it will contain the crawled value or it would be empty
        df2['DATE'] = pd.to_datetime(df2['DATE'], format='%Y-%m-%d')
        df2 = df2.loc[df2['DATE'] >= currDate]
        df3 = pd.concat([df1,df2], axis=0)
        df3.to_csv(f'/home/baadalvm/Playground_Ronak_Sourav/Practice/Data/PlottingData/SOYABEAN/Nans_Data_Temp/{f}', index=False)

def appendInterpolatedData(ifiles, cnt, currDate):
    # Kota and Mahidpur append last value
    for f in ifiles:
        sdf = pd.read_csv(f'/home/baadalvm/Playground_Ronak_Sourav/Practice/Data/PlottingData/SOYABEAN/ARIMA_Interpolated_Data/{f}')
        curr_data = sdf.loc[sdf['DATE'] == currDate, :]
        for i in range(1, cnt+1):
            tdf = pd.read_csv(f'/home/baadalvm/Playground_Ronak_Sourav/Practice/Data/PlottingData/SOYABEAN/ARIMA_Interpolated_Data/{i}_{f}')
            tdf = pd.concat([tdf, curr_data])
            tdf.to_csv(f'/home/baadalvm/Playground_Ronak_Sourav/Practice/Data/PlottingData/SOYABEAN/ARIMA_Interpolated_Data/{i}_{f}', index=False)