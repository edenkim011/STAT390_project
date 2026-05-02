#baseline script: willl run first linear regression

import pandas as pd
import numpy as np
import time #measure runtime 
from sklearn.ensemble import RandomForestRegressor # create baseline model
from prepare import evaluate_model # model evaluator taken from prepare.py


#baseline model
def run_baseline():
    start_time = time.time() #start timer for runtime


    #load working_training data created by prepare.py
    #keep locked_test data hidden 
    train_df = pd.read_csv('data/working_train.csv')

    #select simple features for baseline
    features = ['Age', 'Fee', 'Type', 'Gender']
    target = 'AdoptionSpeed'

    #select from data 
    X = train_df[features]
    y = train_df[target]


    #train baseline model 
    model = RandomForestRegressor(random_state=42)
    model.fit(X, y)

    #generate predictions
    preds = model.predict(X)

    #evaluate model
    rmse = evaluate_model(y, preds)



    #get total run time 
    end_time = time.time() #stop timer
    duration = round(end_time - start_time, 4) #round seconds to 4 decimal points


    #print results 
    print(f"Baseline training RMSE: {rmse}")
    print(f'Runtime: {duration} seconds')

    return rmse #return to put in log history

#if train.py called in terminal, run run_baseline()
if __name__ == '__main__':
    run_baseline()


