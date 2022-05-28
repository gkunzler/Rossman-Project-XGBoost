import pickle
import inflection
import pandas as pd
import numpy as np
import math
import datetime

#ROSSMAN CLASS
class Rossmann(object):
    def __init__ (self):
        self.home_path = ''
        self.competition_distance_scaler   =  pickle.load (open(self.home_path + 'parameter/competition_distance_scaler.pkl', 'rb'))
        self.competition_time_month_scaler =  pickle.load (open(self.home_path + 'parameter/competition_time_month_scaler.pkl', 'rb'))
        self.promo_time_weeks_scaler       =  pickle.load (open(self.home_path + 'parameter/promo_time_weeks_scaler.pkl', 'rb'))
        self.year_scaler                   =  pickle.load (open(self.home_path + 'parameter/year_scaler.pkl', 'rb'))
        self.store_type_scaler             =  pickle.load (open(self.home_path + 'parameter/store_type_scaler.pkl', 'rb'))

    
    def data_cleaning(self, df1):
        
        #rename columns
        cols_old = ['Store', 'DayOfWeek', 'Date', 'Open', 'Promo',
                    'StateHoliday', 'SchoolHoliday', 'StoreType', 'Assortment',
                    'CompetitionDistance', 'CompetitionOpenSinceMonth',
                    'CompetitionOpenSinceYear', 'Promo2', 'Promo2SinceWeek',
                    'Promo2SinceYear', 'PromoInterval']
    
        #build function 
        snakecase = lambda x: inflection.underscore(x)
        #apply function in cols_old
        cols_new = list(map(snakecase, cols_old))
        #rename
        df1.columns = cols_new



        #data types
        df1['date'] = pd.to_datetime(df1['date'], format = '%Y-%m-%d')

        #fillout na
        #competition_distance
        #it was considered that if it is na, it has no close competition. So distance was filled with a high value 
        df1['competition_distance'] = df1['competition_distance'].apply (lambda x: 200000.0 if math.isnan(x) else x)
        
        #competition_open_since_month
        #filled with sales month date - so it will be as the competition started in the same month as the sale
        df1['competition_open_since_month'] = df1.apply(lambda x: x['date'].month if math.isnan(x['competition_open_since_month']) else x['competition_open_since_month'], axis = 1)
        
        #competition_open_since_year
        #filled with sales year date - so it will be as the competition started in the same year as the sale
        df1['competition_open_since_year'] = df1.apply(lambda x: x['date'].year if math.isnan(x['competition_open_since_year']) else x['competition_open_since_year'], axis = 1)
        
        #promo2_since_week
        #filled with sales date - so it will be as the promo2 started in the same date as the sale
        df1['promo2_since_week'] = df1.apply(lambda x: x['date'].week if math.isnan(x['promo2_since_week']) else x['promo2_since_week'], axis = 1)
        
        #promo2_since_year
        #filled with sales date - so it will be as the promo2 started in the same date as the sale
        df1['promo2_since_year'] = df1.apply(lambda x: x['date'].year if math.isnan(x['promo2_since_year']) else x['promo2_since_year'], axis = 1)
        
        #promo_interval
        df1['promo_interval'].fillna(0, inplace= True)
        
        #promo now
        month_map = {1: 'Jan', 2: 'Feb', 3: 'Mar', 4:'Apr', 5:'May', 6:'Jun', 7:'Jul', 8:'Aug', 9:'Sept', 10:'Oct', 11:'Nov', 12: 'Dec'} 
        df1['month_map'] = df1['date'].dt.month.map(month_map)
        df1['promo_now'] = df1[['promo_interval','month_map']].apply (lambda x: 0 if x['promo_interval'] == 0 else 1 if x['month_map'] in x['promo_interval'].split(',') else 0, axis=1)
        
        #change types
        df1['competition_open_since_month'] = df1['competition_open_since_month'].astype(int)
        df1['competition_open_since_year'] =  df1['competition_open_since_year'].astype(int)      
        df1['promo2_since_week'] = df1['promo2_since_week'].astype(int)
        df1['promo2_since_year'] = df1['promo2_since_year'].astype(int)
        
        return df1
    
    
    
    def feature_engineering(self, df3):

        #year
        df3['year'] = df3['date'].dt.year
        
        #month
        df3['month'] = df3['date'].dt.month
        
        #day
        df3['day'] = df3['date'].dt.day
        
        #week of year
        df3['week_of_year'] = df3['date'].dt.isocalendar().week
        
        #year week
        df3['year_week'] = df3['date'].dt.strftime('%Y-%W')
        
        #competition since
        df3['competition_since'] = df3.apply(lambda x: str(x['competition_open_since_year']) + '-' + str(x['competition_open_since_month']) + '-' + '01', axis=1)
        df3['competition_since'] = pd.to_datetime(df3['competition_since'], format = '%Y-%m-%d')
        df3['competition_time_days'] = df3.apply(lambda x: x['date'] - x['competition_since'], axis=1)
        #transform days into months
        df3['competition_time_days'] = df3['competition_time_days'].astype('timedelta64[D]')
        df3['competition_time_month'] = df3['competition_time_days']/30
        df3['competition_time_month'] = df3['competition_time_month'].astype(int)
        
        #promo since
        df3['promo_since'] = df3['promo2_since_year'].astype(str) + '-' + df3['promo2_since_week'].astype(str)
        df3['promo_since'] = df3['promo_since'].apply(lambda x: datetime.datetime.strptime(x + '-1', '%Y-%W-%w') - datetime.timedelta(days=7))
        df3['promo_time_days'] = df3['date'] - df3['promo_since']
        #transform days into weeks
        df3['promo_time_days'] = df3['promo_time_days'].astype('timedelta64[D]')
        df3['promo_time_weeks'] = df3['promo_time_days']/7
        df3['promo_time_weeks'] = df3['promo_time_weeks'].astype(int)
        
        #assortment
        df3['assortment'] = df3['assortment'].apply(lambda x: 'basic' if x=='a' else
                                                               'extra' if x=='b' else
                                                                'extended')
        
        #holiday
        df3['state_holiday'] = df3['state_holiday'].apply(lambda x: 'public_holiday' if x=='a' else
                                                                    'easter_holiday' if x=='b' else
                                                                    'christmas' if x=='c' else 
                                                                    'regular_day' )
        
        
        #filter lines
        df3 = df3.loc[(df3['open'] != 0)] 
        
        #drop auxiliar columns
        cols_drop = ['open', 'month_map','competition_since','competition_time_days', 'promo_since', 'promo_time_days']
        df3 = df3.drop(cols_drop, axis=1)
        #customers - will be droped, since we don`t kow the number of customers that will be in the stores
        
        return df3
    
    def data_preparing(self, df6):
        
        #competition_distance
        df6['competition_distance'] = self.competition_distance_scaler.fit_transform(df6[['competition_distance']].values)
        
        
        #competition_time_month
        df6['competition_time_month'] = self.competition_time_month_scaler.fit_transform(df6[['competition_time_month']].values)
        
        
        #promo_time_weeks
        df6['promo_time_weeks'] = self.promo_time_weeks_scaler.fit_transform(df6[['promo_time_weeks']].values)
        
        
        #year
        df6['year'] = self.year_scaler.fit_transform(df6[['year']].values)
           
        #encoding
        #state_holiday
        #one hot encoding
        df6 = pd.get_dummies(df6, prefix=['state_holiday'], columns = ['state_holiday'])
        
        #store_type
        #label encoding
        df6['store_type'] = self.store_type_scaler.fit_transform(df6['store_type'])
        
        #assortment
        #ordinal encoding
        assortment_dict = {'basic':1, 'extra':2, 'extended':3}
        df6['assortment'] = df6['assortment'].map(assortment_dict)

        #ciclical features
        #day_of_week
        df6['day_of_week_sin'] = df6['day_of_week'].apply(lambda x: np.sin(x * (2. * np.pi/7)))
        df6['day_of_week_cos'] = df6['day_of_week'].apply(lambda x: np.cos(x * (2. * np.pi/7)))
        
        #month
        df6['month_sin'] = df6['month'].apply(lambda x: np.sin(x * (2. * np.pi/12)))
        df6['month_cos'] = df6['month'].apply(lambda x: np.cos(x * (2. * np.pi/12)))
        
        #day
        df6['day_sin'] = df6['day'].apply(lambda x: np.sin(x * (2. * np.pi/30)))
        df6['day_cos'] = df6['day'].apply(lambda x: np.cos(x * (2. * np.pi/30)))
        
        #week of year
        df6['week_of_year_sin'] = df6['week_of_year'].apply(lambda x: np.sin(x * (2. * np.pi/52)))
        df6['week_of_year_cos'] = df6['week_of_year'].apply(lambda x: np.cos(x * (2. * np.pi/52)))
        
        #columns_selected_boruta
        cols_selected = ['store','promo', 'store_type', 'assortment', 'competition_distance','competition_open_since_month','competition_open_since_year',
                                'promo2','promo2_since_week','promo2_since_year','competition_time_month','promo_time_weeks','day_of_week_sin','day_of_week_cos',
                                'month_sin','month_cos','day_sin','day_cos','week_of_year_sin','week_of_year_cos']
    
    
        return df6[cols_selected]
    
    def get_prediction (self, model, original_data, test_data):
        #prediction
        pred = model.predict (test_data)
        
        #join pred into the original data
        original_data['prediction'] = np.expm1(pred)
        
        return original_data.to_json(orient = 'records', date_format = 'iso')