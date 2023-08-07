
import os
import sys
import shutil
import random
import pickle
import logging
import gc
import joblib
import warnings
import numpy as np
import pandas as pd
import tensorflow as tf

from tcn import TCN
from ast import Tuple
from functools import reduce
from keras.models import load_model
from tensorflow.keras import callbacks
from datetime import datetime, timedelta
from tensorflow.keras.layers import Dense
from sklearn.preprocessing import MaxAbsScaler
from tensorflow.keras.models import Sequential 


current_dir: str = os.path.dirname(os.path.abspath(__file__))
code_dir: str = os.path.abspath(os.path.join(current_dir, '..', '..', '..', '..'))
sys.path.append(os.path.join(code_dir, 'research_and_analysis', 'imputation'))
sys.path.append(os.path.join(code_dir, 'recommendation_models'))
sys.path.append(os.path.join(code_dir, 'utilities'))

from arima_imputation import impute_file
from prospect_theory import generate_recommendations
from push_firebase import push_recommendation, recommendation_type

# from code.research_and_analysis.imputation.arima_imputation import impute_file

''' Note that there are four levels : DEBUG, INFO, WARNING, pertubed 
    Debug allow to show all the other four
'''
warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.DEBUG)

class TCNModel:
    def __init__(self, target_state: str, target_mandi: str, surrogate_state: str, surrogate_mandi: str, 
                 lag_days: int, training_start_date: str, recommendation_start_date: str, commodity: str, forecast_type: str) -> None:
        ''' Global constants '''
        # MODEL Type specific, note that for shotterm count is provided in days, for longterm it is provided in weeks
        self.RETRAIN_FREQUENCY: int = 15 if forecast_type == 'shortterm' else 7
        self.INPUT_SIZE: int = 60 if forecast_type == 'shortterm' else 52
        self.OUTPUT_SIZE: int = 30 if forecast_type == 'shortterm' else 12
        self.DAY_JUMP: int = 1 if forecast_type == 'shortterm' else 7
        # Model type independent
        self.PERTUBED_MODEL_COUNT: int = 5
        self.MODEL_FILE_INFO: str = 'model_info.pkl'
        self.DATE_FORMAT: str = '%Y-%m-%d'
        
        ''' User inputs (default set)'''
        self.commodity_name: str = commodity
        self.recommendation_start_date: str = recommendation_start_date
        self.training_start_date: str = training_start_date
        self.forecast_type: str = f'{forecast_type}_models'
        self.mandi_unique_identifier: str = str()
        ''' Mandi and surrogate mandi '''
        self.target_state: str = target_state
        self.target_mandi: str = target_mandi
        self.surrogate_state: str = surrogate_state
        self.surrogate_mandi: str = surrogate_mandi
        self.lag_days: int = lag_days
        self.standardize_mandi_names()
        ''' setup paths ''' 
        self.current_file_path: str = os.path.abspath(path=__file__)
        self.current_dir: str = os.path.dirname(self.current_file_path)
        self.root_dir: str = os.path.abspath(os.path.join(self.current_dir, '..', '..', '..', '..', '..'))
        self.data_dir: str = os.path.join(self.root_dir, 'data')
        self.logs_dir: str = os.path.join(self.root_dir, 'logs_dir')
        self.prices_dir: str = os.path.join(self.data_dir, 'imputed_data', self.commodity_name, 'prices')
        self.model_dir: str = os.path.join(self.data_dir, 'model_data', self.forecast_type, 'tcn', self.commodity_name, self.mandi_unique_identifier)
        self.saved_model_dir: str = os.path.join(self.model_dir, 'saved_model')
        self.saved_scaler_dir: str =  os.path.join(self.model_dir, 'saved_scaler')
        self.persist_data_dir: str = os.path.join(self.model_dir, 'persist_data')
        self.forecasted_data_dir: str =  os.path.join(self.data_dir, 'forecasted_data', self.forecast_type, 'tcn', self.commodity_name, self.mandi_unique_identifier)
        self.raw_processed_prices_dir: str = os.path.join(self.data_dir, 'crawler_data', 'raw_processed', self.commodity_name, 'prices')
        self.raw_processed_arrivals_dir: str = os.path.join(self.data_dir, 'crawler_data', 'raw_processed', self.commodity_name,'arrivals')
        self.raw_forecasted_dir: str = os.path.join(self.forecasted_data_dir, 'raw')
        self.processed_forecasted_dir: str = os.path.join(self.forecasted_data_dir, 'processed')
        self.recommendation_data_dir: str = os.path.join(self.data_dir, 'recommendation_data', self.forecast_type, 'tcn', self.commodity_name, self.mandi_unique_identifier)
        self.archive_recommendation_dir: str = os.path.join(self.recommendation_data_dir, 'archive')
        self.forecast_ranges_dir: str = os.path.join(self.recommendation_data_dir, 'forecast_ranges')
        
        ''' TCN Model hyperparameters '''
        self.MODEL_HYPERPARAMS: dict = {
            'learning_rate': 0.01,
            'unit1': 32,
            'dropout1': 0.3,
            'unit2': 32,
            'dropout2': 0.1,
            'batch_size': 64,
        }

        ''' Standardarized the names of input mandis '''
        ''' clean files from the previous run ''' #TODO: See whether we need to call this from the other file only first time
        #self.clean_old_files() #TODO: Decide location (most prob. in the file from where it is called)
        self.initiate_directories()
        
    '''
    @standardize_mandi_names: A consistent naming convention across all files
    '''
    def standardize_mandi_names(self) -> None:
        logging.debug(msg='Standardize the names of mandi')
        self.target_state: str = '_'.join(self.target_state.split(sep=' ')).lower()
        self.target_mandi: str = '_'.join(self.target_mandi.split(sep=' ')).lower()
        self.surrogate_state: str = '_'.join(self.surrogate_state.split(sep=' ')).lower()
        self.surrogate_mandi: str = '_'.join(self.surrogate_mandi.split(sep=' ')).lower()
        self.mandi_unique_identifier: str = f'{self.target_state}_{self.target_mandi}_{self.surrogate_state}_{self.surrogate_mandi}'

    '''
    @clean_model_files(...)
    - clean all saved model, scalers and persist storage, in case if the experiment is re-run
    '''
    def clean_old_files(self) -> None:
        logging.warning(msg='This will remove the previous run model for this mandi, BE CAREFUL')
        for model_id in range(self.PERTUBED_MODEL_COUNT):
            logging.debug(f'[model {model_id}] Removing saved models')
            saved_model_path: str = os.path.join(self.saved_model_dir, f'model_{model_id}')
            for root, dirs, files in os.walk(top=saved_model_path, topdown=False):
                for file in files:
                    os.remove(path=os.path.join(root, file))
                for dir in dirs:
                    shutil.rmtree(os.path.join(root, dir))
            logging.debug(f'[model {model_id}] Removing scalers')
            saved_scaler_path: str = os.path.join(self.saved_scaler_dir, f'model_{model_id}')
            for root, dirs, files in os.walk(top=saved_scaler_path, topdown=False):
                for file in files:
                    os.remove(path=os.path.join(root, file))
                for dir in dirs:
                    shutil.rmtree(os.path.join(root, dir))
            
            logging.debug(f'[model {model_id}] Removing raw forecasts from last run')
            model_raw_forecasts_dir = os.path.join(self.raw_forecasted_dir, f'model_{model_id}')
            for root, dirs, files in os.walk(top=model_raw_forecasts_dir, topdown=False):
                for file in files:
                    os.remove(path=os.path.join(root, file))
            logging.debug(f'[model {model_id}] Removing processed forecasts ')
            model_processed_forecasts_dir: str = os.path.join(self.processed_forecasted_dir, f'model_{model_id}')
            for root, dirs, files in os.walk(top=model_processed_forecasts_dir, topdown=False):
                for file in files:
                    os.remove(path=os.path.join(root, file))


        logging.debug('Removing persist data dir')
        for root, dirs, files in os.walk(top=self.persist_data_dir, topdown=False):
            for file in files:
                os.remove(path=os.path.join(root, file)) 
            for dir in dirs:
                shutil.rmtree(os.path.join(root, dir))      
                                
        logging.debug(f'Removing archive files for recommendations')
        archive_recommendation_dir: str = os.path.join(self.archive_recommendation_dir)
        for root, dirs, files in os.walk(top=archive_recommendation_dir, topdown=False):
            for file in files:
                os.remove(path=os.path.join(root, file))
        
        logging.debug(f'Removing range of forecasts from recommendations dir')
        for root, dirs, files in os.walk(top=self.forecast_ranges_dir, topdown=False):
            for file in files:
                os.remove(path=os.path.join(root, file))
        
                           
    ''' 
    @initiate_directories(...)
    - For the first time if this mandi is set for forecasts, then all the relevant directories are 
      first created
    '''   
    def initiate_directories(self) -> None:
        logging.debug('Inititate model directory')
        if not os.path.exists(path=self.model_dir):
            os.makedirs(name=self.model_dir)
            os.makedirs(name=self.saved_model_dir)
            os.makedirs(name=self.saved_scaler_dir)
            os.makedirs(name=self.persist_data_dir)
            for model_id in range(self.PERTUBED_MODEL_COUNT):
                os.makedirs(name=os.path.join(self.saved_model_dir, f'model_{model_id}'))
                os.makedirs(name=os.path.join(self.saved_scaler_dir, f'model_{model_id}'))
        logging.debug(msg='Inititate forecastes directory')
        if not os.path.exists(path=self.forecasted_data_dir):
            os.makedirs(name=self.forecasted_data_dir)
            os.makedirs(name=self.raw_forecasted_dir)
            os.makedirs(name=self.processed_forecasted_dir)
            for model_id in range(self.PERTUBED_MODEL_COUNT):
                os.makedirs(name=os.path.join(self.raw_forecasted_dir, f'model_{model_id}'))
                os.makedirs(name=os.path.join(self.processed_forecasted_dir, f'model_{model_id}'))
        logging.debug(msg='Inititate recommendation directory')
        if not os.path.exists(self.recommendation_data_dir):
            os.makedirs(name=self.recommendation_data_dir)
            os.makedirs(name=self.archive_recommendation_dir)
            os.makedirs(name=self.forecast_ranges_dir)
         
         
    '''
    @get_weekly_average_prices
    - This is used by the weekly price forecasts to average out prices of each week to a single value
    '''
    def get_weekly_average_prices(self, prices): 
        avg_prices = list()
        for i in range(0, len(prices), self.DAY_JUMP):
            weekly_avg = list()
            weekly_avg.extend(prices[i: i + self.DAY_JUMP, :].mean(axis=0))
            avg_prices.append(weekly_avg)
        return np.array(avg_prices)   
    
    '''
    @split_train_and_test_data
    - Splits the merged prices into train and test data 
    '''
    def split_train_and_test_data(self, merged_prices_df: pd.DataFrame):
        logging.debug(msg="Spliting of train and test called!")
        leave_size: int = (self.INPUT_SIZE - self.OUTPUT_SIZE) * self.DAY_JUMP
        merged_prices = merged_prices_df[[self.target_mandi, self.surrogate_mandi]].values                    
        train_data, test_data = np.array(merged_prices[:-leave_size, :]), np.array(merged_prices[-leave_size:, :]) 
        return train_data, test_data
    
    '''
    @get_merged_prices(...)
        - We have deployed total 5 model (4 of them are pertubed model and one is build over original prices data)
        - This function takes the model_id and build the dataframe by merging the corresponding price 
          timeseries of that model_id
    '''
    
    def merge_target_and_surrogates_prices(self, model_id: int) -> pd.DataFrame:
        store_data_frames: list = list()
        # For target mandi
        target_mandi_prices_loc: str = os.path.join(self.prices_dir, f'{self.target_state}_{self.target_mandi}_prices.csv')
        if model_id > 0:
            target_mandi_prices_loc: str = os.path.join(self.prices_dir, f'{self.target_state}_{self.target_mandi}_prices_{model_id}.csv')
        target_df: pd.DataFrame = pd.read_csv(os.path.join(target_mandi_prices_loc), index_col='DATE')
        target_df = target_df.loc[:, ~(target_df.columns.str.match(pat='Unnamed'))]
        store_data_frames.append(target_df)
        # For surrogate mandi
        surrogate_mandi_prices_loc: str = os.path.join(self.prices_dir, f'{self.surrogate_state}_{self.surrogate_mandi}_prices.csv')
        if model_id > 0:
            surrogate_mandi_prices_loc: str = os.path.join(self.prices_dir, f'{self.surrogate_state}_{self.surrogate_mandi}_prices_{model_id}.csv')
        surrogate_df: pd.DataFrame = pd.read_csv(os.path.join(surrogate_mandi_prices_loc), index_col='DATE')
        surrogate_df = surrogate_df.loc[:, ~(surrogate_df.columns.str.match(pat='Unnamed'))]
        # Shift the mandi prices based on lag days observed and fill newly become empty filled by backward fill
        surrogate_df['PRICE'] = surrogate_df['PRICE'].shift(periods=self.lag_days).bfill()
        store_data_frames.append(surrogate_df)
        
        merged_prices_df: pd.DataFrame = reduce(lambda left, right: pd.merge(left = left, right = right, on = ['DATE'], how = 'inner'), store_data_frames)
        rename_columns = [self.target_mandi, self.surrogate_mandi]
        merged_prices_df.set_axis(labels=rename_columns, axis='columns', inplace=True)
        merged_prices_df.index = pd.to_datetime(arg=merged_prices_df.index, format=self.DATE_FORMAT)
        return merged_prices_df
    
    '''
    @prepare_training_data
    - The given training_data is iterated using sliding window techinque by taking stride of 1
    - The training_input is a list contains inputs of size self.INPUT_SIZE and same for training_output
    '''
    def prepare_training_data(self, training_data):
        logging.debug(msg="Preparing training dataset")
        total: int = (self.INPUT_SIZE + self.OUTPUT_SIZE) * self.DAY_JUMP
        training_input: list = list()
        training_output: list = list()
        for i in range(0, len(training_data) - total, 1):
            # Default set to shortterm model 
            train_input_i = training_data[i: i + self.INPUT_SIZE * self.DAY_JUMP]
            train_output_i = training_data[i + self.INPUT_SIZE * self.DAY_JUMP: i + total, 0]
            # If model type is longterm then use get_weekly_average_prices
            if self.forecast_type == 'longterm_models':
                train_input_i = self.get_weekly_average_prices(training_data[i: i + self.INPUT_SIZE * self.DAY_JUMP])
                train_output_i = self.get_weekly_average_prices(training_data[i + self.INPUT_SIZE * self.DAY_JUMP: i + total])[:, 0]
            training_input.append(train_input_i)
            training_output.append(train_output_i)
        training_input = np.array(training_input)
        training_output = np.array(training_output)
        logging.debug(f'Input shape = {training_input.shape}, Output shape = {training_output.shape}')
        return training_input, training_output

    '''
    @scale_and_transform_data
    - Used to scale down the prices value using max abs scaler
    - These scalers are further required to bring data back during forecast, thus
        they are return back and stored so that later can be used
    '''

    def scale_and_transform_data(self, training_input, training_output):
        logging.debug(msg="Scaling and transforming data")
        input1_scaler: MaxAbsScaler = MaxAbsScaler()
        input2_scaler: MaxAbsScaler = MaxAbsScaler()
        output_scaler: MaxAbsScaler = MaxAbsScaler()
        training_input[:, :, 0] = input1_scaler.fit_transform(training_input[:, :, 0].reshape(-1, 1)).reshape(training_input.shape[0], -1)
        training_input[:, :, 1] = input2_scaler.fit_transform(training_input[:, :, 1].reshape(-1, 1)).reshape(training_input.shape[0], -1)
        training_output = output_scaler.fit_transform(training_output)
        return training_input, training_output, input1_scaler, input2_scaler, output_scaler

    def scheduler(self, epoch, lr):
        if epoch < 10:
            return lr
        else:
            return lr * tf.math.exp(-0.1)
    '''
    @train_model
    - Defines TCN model architecture, optimizers and other hyper parameters like learning rate
        loss function and others
    - The function return the trained model back.
    '''
    def train_model(self, training_input: np.ndarray, training_output: np.ndarray, learning_rate: float, 
                    batch_size: int):
        logging.debug("Defining model parameters and fitting model on the data")
        logging.debug(msg=f'input shape = {training_input.shape[1]} and {training_input.shape[2]}')
        model: Sequential = Sequential([
                    TCN(input_shape = (training_input.shape[1], training_input.shape[2])),
                    Dense(self.OUTPUT_SIZE)
        ])

        optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
        model.compile(loss = 'mse', optimizer = optimizer)
        early_stopping = callbacks.EarlyStopping(monitor = 'val_loss', patience = 5, 
                                                mode = 'min', min_delta = 0.0001, 
                                                restore_best_weights = True)
        lr_scheduler = callbacks.LearningRateScheduler(self.scheduler)
        logging.debug(f'input = {training_input.shape}, output = {training_output.shape}')
        model.fit(training_input, training_output, epochs = 100, shuffle = False, 
                    validation_split = 0.2, callbacks = [early_stopping, lr_scheduler], 
                    verbose = 0, batch_size = batch_size)
        return model

    ''' 
    @generate_pertubed_model_dataset
    - Only called if this dataset doesn't exits.
    - This function is used to prepared the data set for a given pertubed models
        where count of these model is number_of_models
    - We first check the all the chunks sizes of continuous available and missing days and then
        shuffle this list of chunks sizes (for both). And Now from the original interpolated 
        data we take out missing in the same chunk sizes and later interpolated using ARIMA
    '''
    def generate_pertubed_model_dataset(self) -> None:
        logging.info(msg='Generating pertubed model dataset')
        mandis: list = [(self.target_state, self.target_mandi), (self.surrogate_state, self.surrogate_mandi)]
        logging.debug(mandis)
        for mandi_state, mandi_name in mandis:
            # mandi_df is the actual price df which is fetched from the agmarknet
            mandi_df = pd.read_csv(os.path.join(self.raw_processed_prices_dir, f'{mandi_state}_{mandi_name}_prices.csv'))
            mandi_df['DATE'] = pd.to_datetime(mandi_df['DATE'], format=self.DATE_FORMAT)
            # missing_days_count and available_days_count will store the continuous missing days and avail days length in a list 
            missing_days_count, available_days_count = list(), list()
            current_missing_count, current_available_count = 0, 0
            # Basic logic to compute the continuous missing and available days
            for _, row in mandi_df.iterrows():
                if pd.isnull(row['PRICE']):
                    current_missing_count: int = current_missing_count + 1
                    if current_available_count:
                        available_days_count.append(current_available_count)
                        current_available_count = 0
                else:
                    current_available_count: int = current_available_count + 1
                    if current_missing_count:
                        missing_days_count.append(current_missing_count)
                        current_missing_count = 0
            if current_missing_count:
                missing_days_count.append(current_missing_count)
            if current_available_count:
                available_days_count.append(current_available_count)
            
            # Generate the pertubed/pertubed dataset for the number of self.PERTUBED_MODEL_COUNT
            for model_id in range(1, self.PERTUBED_MODEL_COUNT):
                logging.debug(f'Generating dataset for the pertubed models (or pertubed models) : {model_id}')
                # Randomly shuffle the continuous missing and available days 
                random.shuffle(available_days_count)
                random.shuffle(missing_days_count)
                missing_idx, missing_len = 0, len(missing_days_count)
                available_idx, available_len = 0, len(available_days_count)
                # Initially set the pertubed df to the original interpolated df 
                pertubed_model_df: pd.DataFrame = pd.read_csv(os.path.join(self.prices_dir, f'{mandi_state}_{mandi_name}_prices.csv'), index_col=False)
                pertubed_model_df['DATE'] = pd.to_datetime(pertubed_model_df['DATE'], format=self.DATE_FORMAT)
                pertubed_model_df['IMPUTED'] = 0
                current_idx: int = 0
                # Generate the data in alternate fashion, first Missing, then avail from the missing_days_count and available_days_count
                while (missing_idx < missing_len) and (available_idx < available_len):
                    pertubed_model_df.iloc[current_idx:(current_idx + missing_days_count[missing_idx] - 1), 1] = np.nan
                    
                    current_idx = current_idx + missing_days_count[missing_idx] + available_days_count[available_idx]
                    missing_idx = missing_idx + 1
                    available_idx = available_idx + 1
                if missing_idx < missing_len:
                    pertubed_model_df.iloc[current_idx:(current_idx + missing_days_count[missing_idx] - 1), 1] = np.nan
                pertubed_model_df.loc[pertubed_model_df['PRICE'].isna(), 'IMPUTED'] = 1
                # pertubed_files.append(f'{model_id + 1}_{mandi_name}')
                # Input location is where these pertubed model raw input with missing value will get stored
                pertubed_model_input_file: str = os.path.join(self.raw_processed_prices_dir, f'{mandi_state}_{mandi_name}_prices_{model_id}.csv')
                logging.info(f'pertubed_mode_input = {pertubed_model_input_file}')
                pertubed_model_df.to_csv(pertubed_model_input_file, index=False)
                # Interpolated file for the pertubed model locations, this will be later used by tcn model which get trained on pertubed dataset of pertubed model {model_id}
                pertubed_model_interpolated_output_file: str = os.path.join(self.prices_dir, f'{mandi_state}_{mandi_name}_prices_{model_id}.csv')
                impute_file(input_file_path=pertubed_model_input_file, 
                            output_file_path=pertubed_model_interpolated_output_file) 
                
    def ready_for_recommend(self) -> bool: 
        for model_id in range(0, self.PERTUBED_MODEL_COUNT):
            model_dir = os.path.join(self.raw_forecasted_dir, f'model_{model_id}')
            for day in range(0, self.OUTPUT_SIZE):
                if not (os.path.exists(path=os.path.join(model_dir, f'day_{day}.csv'))):
                    return False
        return True
            
    '''
    @setup_forecast_files
    -NOTE: Below is a sample explanation
    for shotterm (for longterm it is same except we will have 12 files instead of 30)
    ------------------------------------------------------------------------------
    This is an utility function. The purpose was to rotate files. Since there are 
    overlapping forecast that are used by each Day. For example Day x use forecast
    from all the past 29 days including itself. Similarly Day x - 1 use forecast 
    from past 29 days and itself. In both Day x and Day x - 1, there are 29 days 
    common forecast. So thus the below function setup the indexing with reference 
    to the current day on which we are. For example day at index i for Day x - 1
    will appear at index i - 1 (since window slides by 1)
    
    Steps: For all pertubed_models (Note order matters of the below steps)
        1. Remove Day 1
        2. Do the move operation Day 2 to Day 29 -> Day 1 to Day 28
        3. And finally move Day 0-> Day 29, 
        
        $ Initial condition 
        Day 1, ..., Day 28, Day 29  [Today: Day 0] [Tomorrow] (default idx/location for today's 
        model to save it's forecast)]
        <-------------------------------->  
        After setup_forecast_file_for_model is called
        All day from Day 2 to Day 29 shift left by 1 i.e Day 2 becomes Day 1 and Day 29 become Day 28,
        and thus this allow Day 0 to occupy it's correct position i.e Day 29
        This will also make clear why we are removing Day 1 as it is output of this 30 day window 
        and no more required
        
        [Day 1 <- Day 2], [Day 2 <- Day 3] ..... , [Day 28 <- Day 29], [Day 29 <- Day 0] 
        So now [Tomorrow] we will have 29 files and we will generate one on the that day i.e Day 0, 
        thus we have in total 30 files. So basically all this setup is done for the next day, so that 
        it can safely produce output at Day 0[Tomorrow], so that Day 0 output of [Today] get shifted to Day_29
    '''
    def setup_forecast_files_for_models(self):
        logging.debug('File move is called!')
        # For each pertubed model
        for model_id in range(0, self.PERTUBED_MODEL_COUNT):
            logging.debug(f'Files are moved for model_id = {model_id}')
            model_dir = os.path.join(self.raw_forecasted_dir, f'model_{model_id}')
            # We remove the day 1 as it is the oldest day in the past (loosing the self.OUTPUT_SIZE day moving window)
            if os.path.exists(path=os.path.join(model_dir, 'day_1.csv')):
                os.remove(path=os.path.join(model_dir, 'day_1.csv'))    
            # For day 2 to day self.OUTPUT_SIZE, shift all of them by 1 to the left on number line, i.e day - 1
            for day in range(2, self.OUTPUT_SIZE):
                original_file_path: str = os.path.join(model_dir, f'day_{day}.csv')
                renamed_file_path: str = os.path.join(model_dir, f'day_{day - 1}.csv')
                if os.path.exists(path=original_file_path):
                    os.rename(src=original_file_path, dst=renamed_file_path)
            # Rename day 0 to day (self.OUTPUT_SIZE - 1)
            if os.path.exists(os.path.join(model_dir, 'day_0.csv')):
                os.rename(src=os.path.join(model_dir, 'day_0.csv'),
                                       dst=os.path.join(model_dir, f'day_{self.OUTPUT_SIZE - 1}.csv'))
    '''
    @prepare_forecast_input(...)
    - An utility function which is used to prepare the forecast input 
    '''
    def prepare_forecast_input(self, merged_prices_df: pd.DataFrame, model_id: int):
        logging.debug('Prepare forecast input called..')
        forecast_input = merged_prices_df[[self.target_mandi, self.surrogate_mandi]].values
        # Shotterm models (default value)
        forecast_input = forecast_input[-(self.INPUT_SIZE * self.DAY_JUMP):, :]
        # If it is longterm then use get_weekly_average_prices to get the avg prices
        if self.forecast_type == 'longterm_models':
            forecast_input = self.get_weekly_average_prices(forecast_input[-(self.INPUT_SIZE) * self.DAY_JUMP:, :])
        forecast_input = np.array([np.array(forecast_input)])
        logging.debug(f'Forecast input shape: {forecast_input.shape}')
        ''' Use scalers to scale the input '''
        input_scaler_target = joblib.load(os.path.join(self.saved_scaler_dir, f'model_{model_id}', f'input_scaler_{self.target_mandi}.save'))
        input_scaler_surrogate = joblib.load(os.path.join(self.saved_scaler_dir, f'model_{model_id}', f'input_scaler_{self.surrogate_mandi}.save'))
        ''' Reshape the input to the desire input required by tcn model '''
        transformed_target = input_scaler_target.transform(forecast_input[:, :, 0].reshape(-1, 1)) 
        transformed_surrogate = input_scaler_surrogate.transform(forecast_input[:, :, 1].reshape(-1, 1)) 
        forecast_input[:, :, 0] = transformed_target.reshape(forecast_input.shape[0], -1)
        forecast_input[:, :, 1] = transformed_surrogate.reshape(forecast_input.shape[0], -1)
        logging.debug(f'forecast input shape = {forecast_input.shape}')
        return forecast_input
    
    '''
    @fill_pertubed_dataset:
    - Every day we fill all the dates which are not present in pertubed dataset, but are present in the 
      original crawled (imputed) file
    '''
    def fill_pertubed_dataset(self):
        # original cralwed + imputed prices file
        original_imputed_prices_df: pd.DataFrame = pd.read_csv(os.path.join(self.prices_dir, f'{self.target_state}_{self.target_mandi}_prices.csv'))
        original_imputed_prices_df['DATE'] = pd.to_datetime(original_imputed_prices_df['DATE'])
        # For each pertubed model
        for model_id in range(1, self.PERTUBED_MODEL_COUNT):
            pertubed_prices_path: str = os.path.join(self.prices_dir, f'{self.surrogate_state}_{self.surrogate_mandi}_prices_{model_id}.csv')
            pertubed_prices_df: pd.DataFrame = pd.read_csv(pertubed_prices_path)
            pertubed_prices_df['DATE'] = pd.to_datetime(pertubed_prices_df['DATE'])
            # All the dates which are not present in pertubed, but present in original are simply copied to pertubed
            pertubed_prices_df = pd.concat([pertubed_prices_df, 
                                                 original_imputed_prices_df[~original_imputed_prices_df['DATE'].isin(pertubed_prices_df['DATE'])]])
            pertubed_prices_df.to_csv(pertubed_prices_path)
         
    '''
    @generate_forecasts_and_persist(...)
    - Generates forecast daily, if the day is also the model retrain day, then first model is retrained
      and only after that forecasts are generated
    - If not the retrain day, then already saved tcn model is used, loaded to the memory and forecasts 
      for next self.OUTPUT_SIZE days
    '''
    
    def generate_forecasts_and_persist(self, model_id: int, merged_prices_df: pd.DataFrame,
                        retrain_day: bool, learning_rate, 
                        batch_size)-> None:
        logging.info(msg=f'Running for target_mandi = {self.target_mandi} and surrogate_mandi = {self.surrogate_mandi}')
        # If the is the retrain day (which happen every self.RETRAIN_FREQUENCY) we first retrain the model and save back the fresh model
        if retrain_day == True:
            # Generate the fresh pertubed dataset, since new 15 days get added
            logging.info(msg=f''' [TRAINING DAY]: For Model : [{model_id}], 
                                   This is a training Day, Prediction will happen after training ''')
            training_data, _ = self.split_train_and_test_data(merged_prices_df=merged_prices_df)
            training_input, training_output = self.prepare_training_data(training_data=training_data)
            training_input, training_output, input1_scaler, input2_scaler, output_scaler =  self.scale_and_transform_data(training_input=training_input, training_output=training_output)
            training_output = np.expand_dims(training_output, axis = 2)
            logging.debug(f'training input shape = {training_input.shape}')
            logging.debug(f'training output shape = {training_output.shape}')
            ''' Train the model for the input and corresponding output'''
            tcn_model = self.train_model(training_input=training_input, training_output=training_output, 
                                         learning_rate=learning_rate, batch_size=batch_size) 
            logging.debug('Saving the TCN Model')
            ''' Save the TCN model, so that later it can use for forecasts  '''
            tcn_model.save(os.path.join(self.saved_model_dir, f'model_{model_id}'), include_optimizer=True) 
            ''' Save the scalers used for input and output '''
            joblib.dump(input1_scaler, os.path.join(self.saved_scaler_dir, f'model_{model_id}', f'input_scaler_{self.target_mandi}.save'))
            joblib.dump(input2_scaler, os.path.join(self.saved_scaler_dir, f'model_{model_id}', f'input_scaler_{self.surrogate_mandi}.save'))
            joblib.dump(output_scaler, os.path.join(self.saved_scaler_dir, f'model_{model_id}', 'output_scaler.save'))
            tf.compat.v1.reset_default_graph()
            del tcn_model  
            del training_input
            del training_output
            
         
        logging.info(f'''Started forecast for model = {model_id}''')
        ''' Load the TCN model from the disk to the memory for the forecasts ''' 
        tcn_model = load_model(os.path.join(self.saved_model_dir, f'model_{model_id}'))
        optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
        tcn_model.compile(loss = 'mse', optimizer = optimizer)
        ''' Prepare data for forecasting (from past 60 days prices (target mandi prices, surrogate mandi prices) as input )'''
        forecast_input = self.prepare_forecast_input(merged_prices_df=merged_prices_df, model_id=model_id)
        ''' Generate forecast using the loaded TCN model'''
        forecast_output = tcn_model.predict(forecast_input)
        ''' Inverse transforms, since input is in transformed space, we inverse transform so that prices get back to their original domain'''
        output_scaler = joblib.load(os.path.join(self.saved_scaler_dir, f'model_{model_id}', 'output_scaler.save'))
        forecast_output = output_scaler.inverse_transform(forecast_output)
        forecast_output = forecast_output[0, :] 
        ''' Persist the output to the disk '''
        persist_day_df = pd.DataFrame(data={'FORECASTS': forecast_output.tolist()})
        persist_day_df.to_csv(os.path.join(self.raw_forecasted_dir, f'model_{model_id}', 'day_0.csv'), index=False)
        logging.debug(msg='Done with saving forecasts')
        tf.compat.v1.reset_default_graph()
        del tcn_model
        del forecast_input
        del forecast_output
    '''
    @merge_forecasts_of_models
    - This will merge the forecasts for each model, this is an utility function 
      used by the prospect theory 
      Information
    '''
    def merge_forecasts_of_models(self): 
        # merged_forecasts_range[i]: contains all the forecasts generated for day i, by all the model (one main model and 4 pertubed model)
        merged_forecasts_ranges: list = [list() for day in range(0, self.OUTPUT_SIZE + 1)]
        for model_id in range(0, self.PERTUBED_MODEL_COUNT):
            # Day 0 remain unchanged, for all day >= 1 and <= self.OUTPUT_SIZE 
            # Store all the self.OUTPUT_SIZE dataframes corresponds to the forecast of that particular day, use of this is later.
            store_dataframes: list = list()
            for day in range(0, self.OUTPUT_SIZE):
                load_file_path: str = os.path.join(self.raw_forecasted_dir, f'model_{model_id}', f'day_{day}.csv')
                save_file_path: str = os.path.join(self.processed_forecasted_dir, f'model_{model_id}', f'day_{day}.csv')
                raw_forecasts_df: pd.DataFrame = pd.read_csv(load_file_path)
                raw_forecasts_df = raw_forecasts_df.reset_index(drop=True)
                processed_forecast_df = pd.DataFrame(columns=['FORECASTS'])
                # Add zeros infront so that we can read values columns wise
                add_shift_of_zeroes: pd.Series = pd.Series([0] * day)
                if day >= 1:
                    shifted_forecasts = pd.concat([add_shift_of_zeroes, raw_forecasts_df['FORECASTS']], ignore_index=True)
                    processed_forecast_df['FORECASTS'] = shifted_forecasts
                else:
                    processed_forecast_df['FORECASTS'] = raw_forecasts_df['FORECASTS']
                store_dataframes.append(processed_forecast_df)
                processed_forecast_df.to_csv(save_file_path, index=False)
            ''' For each pertubed model we have self.OUTPUT_SIZE dataframes '''
            
            logging.debug(f'Size of store dataframes: {len(store_dataframes)}')
            # Appending zeros logic is helpful here, and thus smoothly allow us to fetch all the prices for a given day
            for idx in range(self.OUTPUT_SIZE, 2 * self.OUTPUT_SIZE):
                day: int = idx % self.OUTPUT_SIZE
                ''' Since there is no shift in day 0 thus day is used and since there is shift in all other thus idx is used'''
                merged_forecasts_ranges[day].append(store_dataframes[0].iloc[day]['FORECASTS'])
                for day_j in range(day + 1, self.OUTPUT_SIZE):
                    merged_forecasts_ranges[day].append(store_dataframes[day_j].iloc[idx]['FORECASTS'])
        return merged_forecasts_ranges    
                
    '''
    @forecast_and_recommend(...)
    - This will call the  for each model 
    - And finally merge all the forecasts and generate the final recommendation using Prospect 
      theory
    - To ensure models are in-sync as well as to avoid any in-between break of codes, the data is 
      only persist once each model generate it's forecast and safely persist those to the disk 
    - If this part doesn't happen atomically, then we don't persist any thing and thus model 
      train/forecast the same thing as well save to the same location, thus code is idempotent
      (no matter how many time it crash, it is still safe to run)
    '''
    def forecast_and_recommend(self) -> None:
        logging.debug('Forecast and Recommendation is called!')
        ''' Setup default value at the beginning, later these will get update if already exist in persist dir'''
        delta: timedelta = datetime.strptime(self.recommendation_start_date, self.DATE_FORMAT) - datetime.strptime(self.training_start_date, self.DATE_FORMAT) 
        training_end_date = datetime.strptime(self.training_start_date, self.DATE_FORMAT) + timedelta(days=delta.days - self.OUTPUT_SIZE * self.DAY_JUMP)
        model_retrain_date = training_end_date
        ''' Load data from persist storage, this file will start into existence if model is run once '''
        last_run_info_file: str = os.path.join(self.persist_data_dir, self.MODEL_FILE_INFO)
        if (os.path.exists(path=last_run_info_file)):
            logging.debug(msg=f'Reading from the dump file')
            with open(file=last_run_info_file, mode='rb') as f:
                training_end_date, model_retrain_date= pickle.load(file=f)    
            f.close() 
            logging.debug(msg=f'''Training data till : {training_end_date}, Model retrain date: {model_retrain_date}''') 
            
        # If today is the retrain day, then first generated the dataset for the pertubed models
        retrain_day: bool = training_end_date == model_retrain_date
        #TODO: Uncomment below lines 
        if retrain_day == True:
            self.generate_pertubed_model_dataset()
        # All the 'missing dates' which are present in original imputed file but not in pertubed are copied
        self.fill_pertubed_dataset()
        ''' For each model generate the forecasts and persist those the disk'''
        for model_id in range(0, self.PERTUBED_MODEL_COUNT):
            learning_rate = self.MODEL_HYPERPARAMS.get('learning_rate')
            batch_size = self.MODEL_HYPERPARAMS.get('batch_size')
            merged_prices_df: pd.DataFrame = self.merge_target_and_surrogates_prices(model_id=model_id) # get the data frame for the pertubed model data
            merged_prices_df: pd.DataFrame = merged_prices_df[merged_prices_df.index >= self.training_start_date]    
            ''' Generate forecast through each model single original model + four pertubed models'''
            merged_prices_df = merged_prices_df[merged_prices_df.index < training_end_date] 
            merged_prices_df = merged_prices_df.reset_index(drop = True)
            ''' Generate and persist the forecast '''
            
            self.generate_forecasts_and_persist(model_id=model_id, 
                                                merged_prices_df=merged_prices_df, 
                                                retrain_day=retrain_day,
                                                learning_rate=learning_rate, 
                                                batch_size=batch_size)

        ''' A single common persistance store for all the models, so that it remain atomic, if any model
            fails in between we will not update the state, and thus next time training/forecast
            will be done for the same date, so that every model staty in sync with each other
        '''
        logging.debug('Updating the persist file for model info like retrain date, training up to which date')
        if training_end_date == model_retrain_date:
            model_retrain_date += timedelta(days = self.RETRAIN_FREQUENCY)
        training_end_date += timedelta(days=1) # only increment by +1, since cronjob is running daily with one iteration.
        logging.debug(f'Training end date: {training_end_date} and model_retrain date: {model_retrain_date} \n\n\n')
            
        ''' If there exists enough file, atleast for last self.OUTPUT_SIZE days, then start with the recommendations '''
        recommendation_date = training_end_date
        ready_to_recommend: bool = self.ready_for_recommend()
        if ready_to_recommend == True:
            logging.info(msg='Wait recommendations are getting generated...')
            ''' Merge forecast generated from all the models and finally generate recommendations'''
            merged_forecasts_ranges = self.merge_forecasts_of_models()
            ''' Store the range of forecasts '''
            forecast_range_file: str = f'recommendation_{datetime.strftime(recommendation_date, self.DATE_FORMAT)}.pkl'
            with open(os.path.join(self.forecast_ranges_dir, forecast_range_file), 'wb') as file:
                pickle.dump(obj=merged_forecasts_ranges, file=file)
            file.close()
            recommendation_date_str: str = datetime.strftime(recommendation_date, self.DATE_FORMAT)
            recommendation_file_name: str = f'recommendation_{recommendation_date_str}.csv'
            recommendation_df = generate_recommendations(merged_forecasts_ranges=merged_forecasts_ranges, 
                                                                    recommendation_date=recommendation_date, 
                                                                    forecast_type=self.forecast_type)
            recommendation_df.to_csv(os.path.join(self.archive_recommendation_dir, recommendation_file_name), index=False)
            #TODO: push to firebase code (duplicate push are already handled)
            push_recommendation(commodity=self.commodity_name, 
                                target_state=self.target_state, 
                                target_mandi=self.target_mandi, 
                                surrogate_state=self.surrogate_state, 
                                surrogate_mandi=self.surrogate_mandi, 
                                start_date=recommendation_date_str, 
                                end_date=recommendation_date_str,
                                rtype=recommendation_type.SHORT if self.forecast_type == 'shortterm_models' else recommendation_type.LONG)

            
        logging.critical(msg='Persist and file reorder is called after forecasts,  safely without any crash in between to ensure models are in-sync')
        ''' Store the updated data back to the persist storage, this data need to persist to survive system crash '''
        with open(os.path.join(last_run_info_file), mode='wb') as f:
            logging.debug(f'Training end date: {training_end_date} and model_retrain date: {model_retrain_date}')
            pickle.dump([training_end_date, model_retrain_date], file=f)
        f.close()
        ''' Setup forecast files by re-ordering files (or rotate) after forecasts get generated, so that next day 
            can safely generate forecasts '''
        self.setup_forecast_files_for_models() 
        
    #NOTE: Cronjob must run weekly basis for the long term forecasts 
    def run_experiment(self, start_date, end_date) -> None:    
        ''' clear all the previous run files for this given, like model data, forecasted data and recommendation data'''
        ''' check if pertubed dataset is already there for these mandi's till the end_date '''
        ''' create directory structure if this is the first iteration run for this set of parameters '''
        logging.debug(msg='Running experiment')
        self.recommendation_start_date = start_date 
        total_days_to_run = (datetime.strptime(end_date, self.DATE_FORMAT) \
                                - datetime.strptime(start_date, self.DATE_FORMAT)).days + self.OUTPUT_SIZE
        total_days_to_run = total_days_to_run // self.DAY_JUMP + 1
        for day in range(0, total_days_to_run):
            logging.debug(msg=f'Forecast and Recommendation called for day = {datetime.strftime(start_date + timedelta(days=day), self.DATE_FORMAT)}\n\n')
            self.forecast_and_recommend()
            gc.collect()
    
    '''
    @forecast(...)
    - This function will be called by the daily running cronjob
    - Possible failure scenario
        1. If model didn't run for few days, for that we are taking till_date.
           First step is to compare the till_date and the last date for which model
           had generated forecasts, whatever is the difference model get that much 
           iterations 
        2. If this is the first time iteration of model, we are running for additional self.OUTPUT_SIZE iterations
           so that we have sufficient files to generate recommendation from the provided day
    '''
    def forecast(self, till_date):
        till_date = datetime.strptime(till_date, self.DATE_FORMAT)
        delta: timedelta = datetime.strptime(self.recommendation_start_date, self.DATE_FORMAT) \
                          - datetime.strptime(self.training_start_date, self.DATE_FORMAT) 
        trained_till = datetime.strptime(self.training_start_date, self.DATE_FORMAT) \
                                        + timedelta(days=delta.days - self.OUTPUT_SIZE * self.DAY_JUMP)
        next_training_date = trained_till 
        previous_run_info_file: str = os.path.join(self.persist_data_dir, self.MODEL_FILE_INFO)

        if os.path.exists(previous_run_info_file):
            with open(file=previous_run_info_file, mode='rb') as f:
                trained_till, next_training_date = pickle.load(file=f)   
            f.close()
        else:
            # In case if this model is called for the first time, we need to run prior for self.OUTPUT_SIZE days, so that we have
            # sufficient files, only after this we will go uptill till_date
            # trained_date is default set to recommend_date - self.OUTPUT_SIZE * self.DAY_JUMP
            for day in range(0, self.OUTPUT_SIZE):
                self.forecast_and_recommend() 
                gc.collect()
        

        if (self.forecast_type == 'longterm_models' and trained_till < next_training_date):
            trained_till += timedelta(days=1)
            with open(file=previous_run_info_file, mode='wb') as f:
                pickle.dump([trained_till, next_training_date], file=f)
            return

        ''' Note that trained_till is updated and thus if till_date is same as trained_till, thus we have +1'''
        count_iterations = (((till_date - trained_till).days + 1) // self.DAY_JUMP)
        logging.debug(f'total number of iteration model will be called: {count_iterations} \n\n')
        logging.info(f'Model will start forecasting from {trained_till} and will goes up till {till_date}')
        for day in range(0, count_iterations):
            forecast_date: str = datetime.strftime(trained_till + timedelta(days=day), self.DATE_FORMAT)
            logging.info(msg=f'Generating forecasts and recommendations for {forecast_date} and will goes up till {till_date}')
            self.forecast_and_recommend()
            gc.collect()
            
        
if __name__ == '__main__':
    logging.info('This is an experiment and not run from the pipeline')
    logging.warning('Note this will override the existing model in pipeline too')
    

    tcn_model = TCNModel(target_state='rajasthan', 
                         target_mandi = 'kota',
                         surrogate_state='madhya pradesh', 
                         surrogate_mandi = 'mahidpur', 
                         lag_days=3, training_start_date='2006-01-01', 
                         recommendation_start_date='2008-10-01',
                         commodity='soyabean', 
                         forecast_type='shortterm')
    
    # tcn_model.run_experiment(start_date = '2006-06-01', end_date = '2007-03-01')
    tcn_model.forecast(till_date='2010-04-01')
