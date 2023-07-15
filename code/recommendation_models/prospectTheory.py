import os 
import sys
import numpy as np
import pandas as pd
import datetime as dt
import matplotlib.pyplot as plt
from datetime import date, datetime
from datetime import timedelta  

pd.plotting.register_matplotlib_converters()

parDir = '/home/baadalvm/Playground_Ronak_Sourav/Temp'

# Setup vairables
freq = 15

def value_function(x):
    if x >= 0:
        return (x**0.86)
    else:
        return -2.61*((-x)**1.06)
    
def weighting_function(p):
    x = 0.98*(p**0.83)
    y = ((1 - p)**0.83)
    return x/(x + y)
    
def confidence(l, mean):
    p = 0
    for i in l:
        if i >= mean:
            p += 1
    return p/len(l)

def gain(l, mean_price):
    result = 0
    count = 0
    for i in l:
        if(i >= mean_price):
            result += i
            count += 1
    if(count == 0):
        return 0
    return result/count

def loss(l, mean_price):
    result = 0
    count = 0
    for i in l:
        if(i < mean_price):
            result += i
            count += 1
    if(count == 0):
        return 0
    return result/count
    
def reduce(final_df):
    reduced_df = pd.DataFrame()
    gains = []
    losses = []
    confidences = []
    mean_gains = []
    mean_losses = []
    mean_prices = []
    # In for loop i represent the ith 30 day interval.
    for i in range(0, len(final_df), 30): # This loops only run single times. 
        predictions = final_df.iloc[i:i + 30]['PREDICTED'].to_numpy() # Fetch 30 days predicted value for the ith interval
        mean_price = sum(predictions[0])/len(predictions[0]) # Day_0 prices are used to compute the mean price
        # prediction[j] : represents the prediction of Day j.

        for day in range(len(predictions)): # The length is 30 (for 30 days) in the ith interval.
            today_mean_price = sum(predictions[day]) / len(predictions[day]) # Mean price for today.
            if(day == 0):
                gains.append(0)
                losses.append(0)
            else:
                gains.append(gain(predictions[day], mean_price) - mean_price)
                losses.append(loss(predictions[day], mean_price) - mean_price)
            confidences.append(confidence(predictions[day], mean_price))
            mean_gains.append(gain(predictions[day], mean_price))
            mean_losses.append(loss(predictions[day], mean_price))
            mean_prices.append(today_mean_price)

    reduced_df['GAIN'] = gains #  sum(prices >= mean_price) / Cnt(price >= meanPrice)  - mean_price
    reduced_df['LOSS'] = losses # sum(prices < mean_price) / Cnt(price < meanPrice) - mean price
    reduced_df['CONFIDENCE'] = confidences # Fraction of value which are more than mean prices on day_0 (reference point) : p(+) ; similarly p(-) = 1 - p(+)
    reduced_df['MEAN_GAIN'] = mean_gains # Mean gain  sum(prices >= mean_price) / cnt(prices >= mean_price)
    reduced_df['MEAN_LOSS'] = mean_losses # Mean loss sum(prices < mean_price) / cnt(prices < mean_price)
    reduced_df['MEAN_PRICE'] = mean_prices # Mean price of the day.
    return reduced_df
    
def prospect(reduced_df):
    prospect_df = pd.DataFrame()
    prices = []
    for index, row in reduced_df.iterrows():
        eff_price = value_function(row['GAIN'])*weighting_function(row['CONFIDENCE']) + value_function(row['LOSS'])*weighting_function(1 - row['CONFIDENCE'])
        prices.append(eff_price)

    prospect_df['PREDICTED'] = pd.Series(prices)
    prospect_df['MEAN_PRICE'] = reduced_df['MEAN_PRICE']
    prospect_df['CONFIDENCE'] = reduced_df['CONFIDENCE']
    prospect_df['MEAN_GAIN'] = reduced_df['MEAN_GAIN']
    prospect_df['MEAN_LOSS'] = reduced_df['MEAN_LOSS']
    prospect_df['DATE'] = reduced_df['DATE']
    
    return prospect_df

    
"""
For mergePredictionsOfEachModel
    Q. why is dfs[0] started with start?
    Ans. It needs to start from 30 to 59, but we haven't appended 30 zeroes to the dfs[0]
    and thus it is same to start with 0.
    --------------------------------------
    Without Error models:
        prices[i] contains range of values predicted for day i, in the given 30 day interval.

                    1 (0) 2(0) ... ............ (29) zeroes. 
                    |   |                        
        Above         29V 28V....               (1V) 
                    |   |
                dfs 0  1  2  3  4  5              29
        --------------------------------------------
        prices[0]   V, V, V, V, V, V .............V
        prices[1]   V, X, V, V, V, V .............V
                    V, X, X, V, V, V .............V
                    ...
        prices[29]  V, X, X, X, X, X,.............X

        Remeber value @start idx is push from dfs[0] for every iteration Itr{j}.
        Why ? because our start contains the prediction of 30th day in the ith interval. Thus it cover every day Day_1 to Day_30 of (i + 1)th interval.
        And also read the [Q] at the top,that tell why not we are starting from 30 itself like 30-59, why we are using 0-29. Reason is we haven't
        appended 30 zeroes in front of it. Whereas in other we have appended zeroes. And thus this becomes a special case.
        
        Itr1: start = 0, and j = (1, 30)
        Itr2: start = 1, and j = (2, 30)
        Itr3: start = 2, and j = (3, 30)

    With Error models: Almost similar.
    
"""

def mergePredictionsOfModel(modelId): 
    '''
    Convert data into the format which then can be passed to the further processing.
    Values are shifted in the first file by 0 days, second by 1 day, third by 2 days, ....., etc
    Values in the first file will be 30, second file will be 31, third file will be 32, ....., etc
    Remember:: We didn't trim values from the file. 
    '''
    for day in range(0, 30):
        df = pd.read_csv(f'{parDir}/Data/ModelData/Predicted_Data/Error_Model_{modelId}_Day_{day}.csv')
        df1 = pd.DataFrame(columns=['PREDICTED'])
        if day == 0:
            df1['PREDICTED'] = df['PREDICTED']
        else:
            zeroSeries = pd.Series([0]* day) # Since we have 10 error models, so day0 = 300, day1 = 290, day2 = 280, to make it equal we need to 
                                                  # append 10 * day zeroes before.
            predictedSeries = pd.concat([zeroSeries, df['PREDICTED']], ignore_index=True) 
            df1['PREDICTED'] = predictedSeries
        df1.to_csv(f"{parDir}/Data/ForecastedData/Error_Model_{modelId}_Day_{day}.csv", index=False) #TODO optimize this by rewriting to the same file which we have opened.(Just check the correctness)

    # dfs[i] -> Contain dataframe of the "Model_idx_Day_i.csv"    
    dfs = []
    for day in range(30):
        temp_df = pd.DataFrame()
        df = pd.read_csv(f'{parDir}/Data/ForecastedData/Error_Model_{modelId}_Day_{day}.csv')
        temp_df = temp_df.append(df, ignore_index = True)
        dfs.append(temp_df)


    predicted = []
    for row in range(30, 60):
        start = row % 30
        prices = []
        prices.append(dfs[0].iloc[start]['PREDICTED']) # Speical case : reasoning provided above in comments.
        for j in range(start + 1, 30):
            prices.append(dfs[j].iloc[row]['PREDICTED'])
        predicted.append(prices)

    errorModelDf = pd.DataFrame()
    # predicted is list of list contains where each index contains range of prices generated for a single day.
    errorModelDf["PRICE_RANGE"] = predicted
    # A single file which is merge of all days  each day contain a list of values. 
    errorModelDf.to_csv(f"{parDir}/Data/ForecastedData/ErrorModelRecommendation/Err_Model_{modelId}.csv", index=False) 
    
    
def generateRecommendation(noOfModels, recommendationIdx):
    '''
    We have 5 error models (Error_model_i)
    > Merge all the output for a given day_i from all the files i.e 5 files from error models.
    > TODO: Once merged we may need to sample out random 5 value from each column correspond to a day.
    > And then we need to run prospect theory at last.
    '''
    final_df = pd.DataFrame()
    predicted = [[] for i in range(30)]
    # For each file produced by error model, iterate through the data on each day and append it to predicted.
    for idx in range(0, noOfModels):
        df = pd.read_csv(f'{parDir}/Data/ForecastedData/ErrorModelRecommendation/Err_Model_{idx}.csv')
        for day in range(0, 30):
            prices = df.loc[day, "PRICE_RANGE"] # TODO DONE also push these from above. recommendation function
            prices = prices[1:-1].split(',')
            prices = [float(price) for price in prices]
            predicted[day] += prices
    
    # TODO : DONE (confirmed with sir) In case if you want to sample out only 5 value for each day.
    final_df['PREDICTED'] = predicted
    # Saving final_df in case of any overwrite to these files.
    # final_df.to_csv(f'{parDir}/Data/ForecastedData/PersistPastPredictions/prediction_{recommendationIdx}.csv', index=False)
    # Call to reduce final_df
    reduced_df = reduce(final_df)
    # Generate dates for the 30 day interval
    today = datetime.strptime("2006-01-01", "%Y-%m-%d") + timedelta(days=recommendationIdx)
    dates = [(today + timedelta(days=i)).strftime('%Y-%m-%d') for i in range(30)]
    reduced_df['DATE'] = dates
    # Call to prospect which takes reduced df and produce outputs.
    return dates[0], prospect(reduced_df)
