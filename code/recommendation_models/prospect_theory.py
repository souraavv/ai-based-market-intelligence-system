import logging 
import datetime as dt
import pandas as pd
from datetime import date, datetime
from datetime import timedelta  

α: float = 0.86
β: float = 1.06
λ: float = 2.61
δ: float = 0.98 
γ: float = 0.83

# Global constants
SHORTTERM_MODEL: str = 'shortterm_models'
LONGTERM_MODEL: str = 'longterm_models'

''' Value function '''
def value_function( x) -> float:
    return (x ** α) if x >= 0.0 else -λ * ((-x) ** β)
''' Weight function '''
def weight_function( p) -> float: 
    return (δ * (p ** γ)) / ( δ * (p ** γ) + (1 - p) ** γ)

def confidence(range_of_prices: list, mean_price: float) -> float:
    cnt_better_than_mean: int = 0
    for price in range_of_prices:
        if price >= mean_price:
            cnt_better_than_mean += 1
    return cnt_better_than_mean / len(range_of_prices)

def gain(range_of_prices: list, mean_price: float) -> float: 
    gain: float = 0
    gain_occurence: int = 0
    for price in range_of_prices:
        if price >= mean_price:
            gain = gain + price 
            gain_occurence = gain_occurence + 1 
    return 0 if gain_occurence == 0 else (gain / gain_occurence)

def loss(range_of_prices: list, mean_price: float) -> float:
    loss: float = 0
    loss_occurence: int = 0
    for price in range_of_prices:
        if price >= mean_price:
            loss = loss + price 
            loss_occurence = loss_occurence + 1
    return 0 if loss_occurence == 0 else (loss / loss_occurence)

def generate_recommendations(merged_forecasts_ranges, recommendation_date, forecast_type: str) -> pd.DataFrame:
    forecasts_length: int = 30 if forecast_type == SHORTTERM_MODEL else 12
    prospect_dict: dict = {'GAIN': list(), 'LOSS': list(), 'CONFIDENCE': list(), 'MEAN_GAIN': list(), 
                            'MEAN_LOSS': list(), 'MEAN_PRICE': list()}
    today_mean_price: float = sum(merged_forecasts_ranges[0]) / len(merged_forecasts_ranges[0])
    for day in range(0, forecasts_length):
        day_mean_price: float = sum(merged_forecasts_ranges[day]) / len(merged_forecasts_ranges[day])
        prospect_dict['MEAN_PRICE'].append(day_mean_price)
        if day == 0:
            prospect_dict['GAIN'].append(0)
            prospect_dict['LOSS'].append(0)
        else:
            prospect_dict['GAIN'].append(gain(range_of_prices=merged_forecasts_ranges[day], mean_price=today_mean_price) - today_mean_price)
            prospect_dict['LOSS'].append(loss(range_of_prices=merged_forecasts_ranges[day], mean_price=today_mean_price) - today_mean_price)
        prospect_dict['CONFIDENCE'].append(confidence(range_of_prices=merged_forecasts_ranges[day], mean_price=today_mean_price))
        prospect_dict['MEAN_GAIN'].append(gain(range_of_prices=merged_forecasts_ranges[day], mean_price=today_mean_price))
        prospect_dict['MEAN_LOSS'].append(loss(range_of_prices=merged_forecasts_ranges[day], mean_price=today_mean_price))
    
    prospect_value = lambda row: value_function(x=row['GAIN']) * weight_function(p=row['CONFIDENCE']) + \
                                                        value_function(x=row['LOSS']) * weight_function(p=1 - row['CONFIDENCE'])
    
    logging.debug(f'Prospect_dict: {prospect_dict}')
    prospect_df = pd.DataFrame(data=prospect_dict)
    if forecast_type == SHORTTERM_MODEL:
        prospect_df['DATE'] = pd.date_range(start=recommendation_date, periods=30, freq='1D')
    elif forecast_type == LONGTERM_MODEL:
        prospect_df['DATE'] = pd.date_range(start=recommendation_date, periods=12, freq='7D')
    prospect_df['PREDICTED'] = prospect_df.apply(prospect_value, axis=1)
    return prospect_df


'''
Information

For mergePredictionsOfEachModel
    Q. why is store_dataframes[0] started with start?
    Ans. It needs to start from 30 to 59, but we haven't appended 30 zeroes to the dfs[0]
    and thus it is same to start with 0.
    --------------------------------------
    Without Error models:
        prices[i] contains range of values predicted for day i, in the given 30 day interval.

                    1 (0) 2(0) ... ............ (29) zeroes. 
                    |   |                        
        Above         29V 28V....               (1V) 
                    |   |
 store_dataframes    0  1  2  3  4  5              29
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

    With Pertubed models: Almost similar.
'''

