from code.research_and_analysis.imputation.arima_imputation import arima_imputation, commodity_info_type

filtered_markets = [('Rajasthan', 'Baran'), ('Rajasthan', 'Atru'), ('Rajasthan', 'Chhabra'), ('Rajasthan', 'Bundi'), ('Rajasthan', 'Nimbahera'), ('Rajasthan', 'Pratapgarh'), ('Rajasthan', 'Bhawani Mandi'), ('Rajasthan', 'Jhalarapatan'), ('Rajasthan', 'Itawa'), ('Rajasthan', 'Kota'), ('Rajasthan', 'Ramaganj Mandi')]
for state, mandi in filtered_markets:
    print(f'imputing started for {state}, {mandi}')
    arima_imputation(commodity='soyabean', state=state, mandi=mandi, end_date='2006-12-31', info_type=commodity_info_type.PRICES)
print(filtered_markets)
