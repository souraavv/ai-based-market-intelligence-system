import os 
import logging 
import pandas as pd 

logging.basicConfig(level=logging.DEBUG)
# Setup the root pat
current_dir: str = os.path.dirname(os.path.abspath(__file__))
root_dir: str = os.path.abspath(os.path.join(current_dir, '..', '..', '..'))
data_dir: str = os.path.join(root_dir, 'data')

# Find the correlation between two mandis by shifting one of them
def find_correlation(mandi_a, mandi_b):
    corr_at = dict()
    for i in range(-10, 11): # [-10, 10]
        # Shift price time series of mandi b and find the correlation with mandi _a
        corr_at[i] = mandi_a['PRICE'].corr(mandi_b['PRICE'].shift(i))
    lag_days: int = max(corr_at.items(), key=lambda item: item[1])[0]
    correlation_val: float = corr_at[lag_days]
    return (lag_days, correlation_val)

def format_path_component(s: str) -> str:
    return '_'.join(s.split()).lower()

#NOTE: This analysis completely depends on the date range of price that is passed, based on different dateranges, you may find different surrogate pair
#NOTE: Thus it is recommended to perform this analysis at-least over 10 years, to get a clear idea on surrogate pair
def get_surrogate_mandis(mandis_list: list, commodity: str):
    
    for idx in range(0, len(mandis_list)):
        state, mandi = mandis_list[idx]
        state = format_path_component(state)
        mandi = format_path_component(mandi)
        mandis_list[idx] = (state, mandi)
        
    analysis_data_dir: str = os.path.join(data_dir, 'analysis_data')
    imputed_data_dir = os.path.join(data_dir, 'imputed_data', commodity, 'prices')
    
    
    surrogate_pair_df = {
        "Target mandi": [], 
        "Surrogate mandi": [],
        "Lag days": [],
        "Correlation": [],
    }
    n: int = len(mandis_list)
    # Setup the matrix (n, n), where n is the total number of matrix, with some default value(doesn't matter)
    correlation_matrix = {mandi_b: {mandi_a: -1 for _, mandi_a in mandis_list} for _, mandi_b in mandis_list}
    lag_days_matrix = {mandi_b: {mandi_a: -10 for _, mandi_a in mandis_list} for _, mandi_b in mandis_list}
    for (state_a, mandi_a) in mandis_list:
        for (state_b, mandi_b) in mandis_list:
            # If this is same mandi then we can set default value, but they will not be used for further computation of surrogate pair
            if mandi_a == mandi_b:
                correlation_matrix[mandi_a][mandi_a] = 1.0
                lag_days_matrix[mandi_a][mandi_a] = 0.0
                continue     
            # find the lags days corresponding to the maximum correlation between mandi_a and mandi_b
            mandi_a_df = pd.read_csv(os.path.join(imputed_data_dir, f'{state_a}_{mandi_a}_prices.csv'))
            mandi_b_df: pd.DataFrame = pd.read_csv(os.path.join(imputed_data_dir, f'{state_b}_{mandi_b}_prices.csv'))
            lag_days, correlation_val = find_correlation(mandi_a=mandi_a_df, mandi_b=mandi_b_df)
            correlation_matrix[mandi_a][mandi_b] = correlation_val
            lag_days_matrix[mandi_a][mandi_b] = lag_days
    
    for _, mandi_a in mandis_list:
        surrogate_mandi: str = str()
        surrogate_mandi_corr: float = -100 # setup the minimum value 
        lag_days: int = -100
        for _, mandi_b in mandis_list:
            if (mandi_a == mandi_b): # ignore this case while selecting surrogate mandi
                continue 
            # Check if mandi_b can be a candidate to a surrogate mandi
            # this is only possible if lags_days comes out to be +ve, since a positive shift
            # in mandi_b timeseries tell us that we bring past of mandi_b to match with present/future of 
            # mandi_a, this means the event happened in mandi_b before mandi_a
            if lag_days_matrix[mandi_a][mandi_b] > 0:
                if (correlation_matrix[mandi_a][mandi_b] > surrogate_mandi_corr):
                    # if this is the best correlation we found till
                    surrogate_mandi_corr = correlation_matrix[mandi_a][mandi_b]
                    surrogate_mandi = mandi_b 
                    lag_days = lag_days_matrix[mandi_a][mandi_b]
                elif (correlation_matrix[mandi_a][mandi_b] == surrogate_mandi_corr):
                    # if you find better mandi which have the event at the earlier most date, then pick that
                    if len(surrogate_mandi) > 0 and (lag_days_matrix[mandi_a][surrogate_mandi] < lag_days_matrix[mandi_a][mandi_b]):
                        surrogate_mandi = mandi_b
                        lag_days = lag_days_matrix[mandi_a][mandi_b]
            # Populate the dataframe
        if lag_days != -100:
            surrogate_pair_df['Target mandi'].append(mandi_a)
            surrogate_pair_df['Surrogate mandi'].append(surrogate_mandi)
            surrogate_pair_df['Lag days'].append(lag_days)
            surrogate_pair_df['Correlation'].append(surrogate_mandi_corr)
    
    surrogate_pair_df = pd.DataFrame(surrogate_pair_df)
    file_name = os.path.join(analysis_data_dir, f'{commodity}.csv')
    logging.info(f"Find the complete table at {file_name}")     
    print (surrogate_pair_df.head(10))
    # Save the file to the local storage
    surrogate_pair_df.to_csv(file_name, index=False)

get_surrogate_mandis(mandis_list=[('Rajasthan', 'Baran'), ('Rajasthan', 'Atru'), ('Rajasthan', 'Chhabra'), ('Rajasthan', 'Bundi'), ('Rajasthan', 'Nimbahera'), ('Rajasthan', 'Pratapgarh'), ('Rajasthan', 'Bhawani Mandi'), ('Rajasthan', 'Jhalarapatan'), ('Rajasthan', 'Itawa'), ('Rajasthan', 'Kota'), ('Rajasthan', 'Ramaganj Mandi')]
                        , commodity='soyabean')