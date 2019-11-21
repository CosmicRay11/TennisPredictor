# file containing the Model superclass from which the other models should inherit

import numpy as np
import random
from data_flow import DataManager
import datetime


# essentially a template for how models should operate
class Model(object):

    def __init__(self):
        self.path = "model//"
        self.columns = []
        self.dm = DataManager(self.path, self.columns)

    # get a dataframe object using a DataManager object
    def get_input_data(self, yearStart, yearEnd, saved=False):
        if not saved:
            df = self.dm.compile_clean_data(yearStart, yearEnd)
        else:
            df = self.dm.load_clean_data(saved)
        return df        

    # make a vector of predictions and outcomes to later input into an Evaluate object
    def make_prediction_set(self, yearStart, yearEnd, saveName, slamsOnly = False, testStart = None, trainName=None, inputCleanFile=False, inputProcessedFile=False):
        # if a file is supplied, use it, otherwise create a processed dataframe from scratch
        if inputProcessedFile:
            df = self.dm.load_clean_data(inputProcessedFile)
        else: 
            df = self.get_input_data(yearStart, yearEnd, saved=inputCleanFile)
            df = self.process_data(df, saveName + "-processed")

        # split the data into test and training sets based on date
        if testStart != None:
            date = datetime.datetime(testStart,1,1)
            df['tourney_date'] = df['tourney_date'].astype('datetime64[ns]')

            df1 = df[(df['tourney_date'] < date)].copy()
            df2 = df[(df['tourney_date'] >= date)].copy()
        else:
            df1 = df.copy()
            df2 = df.copy()

        if slamsOnly:
            df2 = df2[(df2['best_of'] == 5)]

        # drop unnecessary columns, that were necessary to create the test/train datasets
        if 'best_of' not in self.columns:
            df1.drop(columns = ['best_of'], inplace = True)
            df2.drop(columns = ['best_of'], inplace = True)
        
        if 'tourney_date' not in self.columns:
            df1.drop(columns = ['tourney_date'], inplace=True)
            df2.drop(columns = ['tourney_date'], inplace=True)

        df1.reset_index(drop=True, inplace=True)
        df2.reset_index(drop=True, inplace=True)

        # train to model, or load the training file for a model
        if trainName != None:
            self.load_trained_model(trainName)
        else:
            self.train(df1)

        # create the blank output vector
        output = np.zeros((len(df2), 2))

        # drop indexes and columns that shouldn't be given to models
        inputs = df2.copy()
        if "Unnamed: 0" in list(inputs):
            inputs.drop(columns = ['Unnamed: 0'], inplace = True)
        inputs.drop(columns = ['high_rank_won'], inplace=True)
        
        outcomes = df2['high_rank_won'].copy()

        # for each match in the dataframe, get a predicted probability and outcome, and add them to the output vector
        for index,row in inputs.iterrows():
            #array = row.values

            prob = self.predict(row)
            outcome = outcomes.loc[index] * 1

            output[index,0] = prob
            output[index,1] = outcome


        return output

    # subroutines that are common to all models, but act differently internally depending on the particular model (polymorphism)
    
    def process_data(self, df, saveName):
        self.dm.save_df(df, saveName)
        return df

    def train(self, df):
        pass

    def predict(self, predictors):
        pass

    def load_trained_model(self, trainName):
        pass
