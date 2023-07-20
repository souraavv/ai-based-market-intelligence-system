from code.crawler.agmarknet_crawler.wholesale_crawler import crawl_agmarknet_date_range
from code.crawler.odk_crawler.odk_form_data_crawler import crawl_odk_date_range
from code.research_and_analysis.imputation.arima_imputation import arima_imputation
from code.crawler.trading_prices_crawler.crawl_trading_prices import crawl_trading_prices
from code.utilities.push_firebase import push_data, push_odk, push_trading_prices, commodity_info_type

import pandas as pd
from datetime import datetime

# start_date = dataframe read value with minimum or default value if dataframe is absent
start_date = None
end_date = datetime.today().strftime('%Y-%m-%d')

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
tcn_model = TCNModel(target_state='maharashtra', target_mandi = 'nagpur',
                         surrogate_state='madhya pradesh', surrogate_mandi = 'mahidpur', 
                         lag_days=3, training_start_date='2006-01-01', recommendation_start_date=end_date,
                         commodity='soyabean')

# NOTE: recommedations are pushed internally
tcn_model.forecast(till_date=end_date)

# TODO: generate and push recommendations for long term model

# Crawl and push international commodity trading prices
crawl_trading_prices()
push_trading_prices(end_date)
