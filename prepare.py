# judge: load data and calculate RMSE

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import os


#loading and splitting data into 80% training and 20% testing dataset
#create train and locked test csv's
def load_and_split_data():

    #load data
    df = pd.read_csv('data/train.csv')

    #split data, 20% of data is locked test set 
        #random_state: used fixed number so test set never changes everytime it runs 
        #stratify: make sure 'adoptionspeed' is split equally between train and test
    train_df, test_df = train_test_split(
        df, test_size = 0.20, random_state=42, stratify = df['AdoptionSpeed']
    )

    #save these to new files 
        #index =False, just save content, when we open the file next time it will automatically assign index
    train_df.to_csv('data/working_train.csv', index = False)
    test_df.to_csv('data/locked_test.csv', index = False)

    print('Data split complete.')


#return RMSE, what the agent is trying to optimize
def evaluate_model(y_true, y_pred):
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    return round(rmse, 4) #round to 4 decimal points 


#if prepare.py is called in terminal, only run load_and_split_data
if __name__ == '__main__':

    #if locked_test.csv doesnt exist, run load_and_split_data
    if not os.path.exists('data/locked_test.csv'):
        load_and_split_data()

    #if it does exist already, print already exists 
    else:
        print('Data already split and locked.')