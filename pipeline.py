from code.crawler.agmarknet_crawler.wholesale_crawler import crawl_agmarknet_date_range, format_path_component
from code.crawler.odk_crawler.odk_form_data_crawler import crawl_odk_date_range
from code.research_and_analysis.imputation.arima_imputation import arima_imputation
from code.crawler.trading_prices_crawler.crawl_trading_prices import crawl_trading_prices
from code.utilities.push_firebase import push_data, push_odk, push_trading_prices, commodity_info_type

import pandas as pd
import os
from datetime import datetime

# TODO: logic for start date
# start_date = dataframe read value with minimum or default value if dataframe is absent

end_date = '2007-01-01' #datetime.today().strftime('%Y-%m-%d')
target_state = 'rajasthan'
target_mandi = 'kota'
surrogate_state = 'madhya pradesh' 
surrogate_mandi = 'mahidpur'
lag_days=3
training_start_date='2006-01-01'
recommendation_start_date=end_date
commodity='soyabean'


mandi_unique_identifier: str = f'{format_path_component(target_state)}_{format_path_component(target_mandi)}_{format_path_component(surrogate_state)}_{format_path_component(surrogate_mandi)}'

archive_dir: str = os.path.join('data', 'recommendation_data', 'shortterm_models',
                                'tcn', commodity, mandi_unique_identifier, 'archive')

try:
    recommendation_files = os.listdir(archive_dir)
    recommendation_files = sorted(recommendation_files)
    # directory exits but never used for recommendation (might not get hit)
    if len(recommendation_files) == 0:
        start_date = training_start_date
    else:
        file_name: str = recommendation_files[-1]
        print (file_name)
        date = file_name.split('_')[1].split('.')[0]
        print (date)
        start_date = date
except FileNotFoundError as e:
    print (f'''You might be running for this mandi and surrogate mandi first time
             and thus directory doesnot exists 
             
             Setting up a default start date to training start date
             if want separate change manually in the pipeline file''')
    # setting up a default date
    start_date = training_start_date


# Crawl agmarknet data for list of mandis
crawl_agmarknet_mandis = {
    'soyabean': {
                    'rajasthan': ['kota', 'bhawani mandi'],
                    'madhya pradesh': ['mahidpur', 'ratlam', 'sonkatch'],
                    'maharashtra': ['nagpur'],
                    'telangana': ['adilabad']
                }
}

for commodity in crawl_agmarknet_mandis:
    for state in crawl_agmarknet_mandis[commodity]:
        crawl_agmarknet_date_range(commodity=commodity, state=state, start_date=start_date, end_date=end_date)
        for mandi in crawl_agmarknet_mandis[commodity][state]:
            push_data(commodity=commodity, state=state, mandi=mandi, start_date=start_date, end_date=end_date, info_type=commodity_info_type.PRICES)

# Crawl odk data for list of mandis
crawl_odk_mandis = {
    'soyabean': {
                    'telangana': ['adilabad']
                }
}

for commodity in crawl_odk_mandis:
    for state in crawl_odk_mandis[commodity]:
        for mandi in crawl_odk_mandis[commodity][state]:
            crawl_odk_date_range(commodity=commodity, state=state, mandi=mandi, start_date=start_date, end_date=end_date)
            push_odk(commodity=commodity, state=state, mandi=mandi, start_date=start_date, end_date=end_date)


# generate and push recommendations for short term model
tcn_short_model = TCNModel(target_state='maharashtra', target_mandi = 'nagpur',
                         surrogate_state='madhya pradesh', surrogate_mandi = 'mahidpur', 
                         lag_days=3, training_start_date='2006-01-01', recommendation_start_date=end_date,
                         commodity='soyabean', forecast_type='shortterm')

tcn_short_model.forecast(till_date=end_date)

# tcn_long_model = TCNModel(target_state=target_state, 
#                                target_mandi = 'nagpur',
#                                surrogate_state='madhya pradesh', 
#                                surrogate_mandi = 'mahidpur', 
#                                lag_days=3, 
#                                training_start_date='2006-01-01', 
#                                recommendation_start_date=end_date,
#                                commodity='soyabean', 
#                                forecast_type='longterm')

# tcn_long_model.forecast(till_date=end_date)

# Crawl and push international commodity trading prices
crawl_trading_prices()
push_trading_prices(end_date)
