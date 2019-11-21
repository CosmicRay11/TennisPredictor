#logistic regression model - iteration 9

import numpy as np
import random
import pandas as pd
import matplotlib.pyplot as plt

from data_flow import DataManager
from model import Model
from evaluate import Evaluate


# inherit from Model
class LogReg(Model):

    def __init__(self):
        self.path = "logreg//"
        # the columns needed for prediction
        self.columns = ['score', 'surface','best_of','draw_size','high_rank_won', "A_id", "A_seed", "A_entry", "A_hand", "A_ht", "A_age", "A_rank", "A_rank_points", "A_ace", "A_df", "A_svpt", "A_1stIn", "A_1stWon", "A_2ndWon", "A_SvGms", "A_bpSaved", "A_bpFaced", "B_id", "B_seed", "B_entry", "B_hand", "B_ht", "B_age", "B_rank", "B_rank_points", "B_ace", "B_df", "B_svpt", "B_1stIn", "B_1stWon", "B_2ndWon", "B_SvGms", "B_bpSaved", "B_bpFaced"]
        # columns that should be considered additively (i.e. the sum of aces over time rather than the number of sets in a match)
        self.cumColumns = ["A_ace", "A_df", "A_svpt", "A_1stIn", "A_1stWon", "A_2ndWon", "A_SvGms", "A_bpSaved", "A_bpFaced", "B_ace", "B_df", "B_svpt", "B_1stIn", "B_1stWon", "B_2ndWon", "B_SvGms", "B_bpSaved", "B_bpFaced"]
        self.dm = DataManager(self.path, self.columns)

        # set up hyperparameters
        self.learnRate = 0.01
        self.halfLife = 240
        self.surfaces = ['grass','hard','clay']        

    # process the basic features returned by the DataManager object to be more useful for prediction
    def process_data(self, df, saveName):
        # dictionary of players, with player ID as a key
        #                                                                                                                          -4               -3                   -2                        -1
        # value is a list of the form : [ a value for each index in self.cumColumns , bad loss, highest rank, matches played, last date]
        playerDict = {}

        # prefixes for new columns - c for cumulative, s for surface specific and sc for both
        options = ['c', 'sc', 's']

        # set up new columns for surface specific and cumulative varieties
        for option in options:
            for column in self.cumColumns:
                df[option + column] = pd.Series(np.zeros(len(df)), index=df.index)

        # set up new blank columns
        df['sA_matches_played'] = pd.Series(np.zeros(len(df)), index=df.index)
        df['sB_matches_played'] = pd.Series(np.zeros(len(df)), index=df.index)
        df['high_rank_won'] *= 1

        df['A_injury'] = pd.Series(np.zeros(len(df)), index=df.index)
        df['B_injury'] = pd.Series(np.zeros(len(df)), index=df.index)

        df['A_bad_loss'] = pd.Series(np.zeros(len(df)), index=df.index)
        df['B_bad_loss'] = pd.Series(np.zeros(len(df)), index=df.index)

        df['is_grass'] = pd.Series(np.zeros(len(df)), index=df.index)
        df['is_hard'] = pd.Series(np.zeros(len(df)), index=df.index)
        df['is_clay'] = pd.Series(np.zeros(len(df)), index=df.index)

    
        # set up three surface specific player dictionaries in a list
        # again, key is player id, and value is list of form :  [ a value for each index in self.cumColumns, highest rank, matches played, last date]
        surfaceStatList = [{}, {}, {}]

        # for each match
        for index,row in df.iterrows():

            # get the surface of the match, and update the dummy columns to reflect which surface is being considered
            surface = row['surface']
            if surface.lower() == 'grass':
                df.at[index, 'is_grass'] = 1
            elif surface.lower() == 'clay':
                df.at[index, 'is_clay'] = 1
            else:
                df.at[index, 'is_hard'] = 1

            # get the index of the surface of the match (from self.surfaces)
            if surface.lower() in self.surfaces:
                surfaceIndex = self.surfaces.index(surface.lower())
            else:
                surfaceIndex = 1  # (assume miscellaneous surfaces are hard court)

            # get the particular dictionary to update for the current surface
            statDict = surfaceStatList[surfaceIndex]                

            #####################

            # if a player is not in the dictionary, create blank entry for them

            if row['A_id'] not in statDict.keys():
                # -1 = last date, -2 = matches played, -3 = highest rank
                statDict[row['A_id']] = [0 for a in range(len(self.cumColumns)//2+3)]
                statDict[row['A_id']][-3] = 10000
            if row['B_id'] not in statDict.keys():
                statDict[row['B_id']] = [0 for a in range(len(self.cumColumns)//2+3)]
                statDict[row['B_id']][-3] = 10000


            if row['A_id'] not in playerDict.keys():
                # -1 = last date, -2 = matches played, -3 = highest rank, -4 = bad loss
                playerDict[row['A_id']] = [0 for a in range(len(self.cumColumns)//2+4)]
                playerDict[row['A_id']][-3] = 10000
            if row['B_id'] not in playerDict.keys():
                playerDict[row['B_id']] = [0 for a in range(len(self.cumColumns)//2+4)]
                playerDict[row['B_id']][-3] = 10000


            #######################

            
            # get the last date a match was played on the surface in question by each player
            if statDict[row['A_id']][-1] == 0:
                surfaceALastDate = row['tourney_date']
            else:
                surfaceALastDate = statDict[row['A_id']][-1]
            if statDict[row['B_id']][-1] == 0:
                surfaceBLastDate = row['tourney_date']
            else:
                surfaceBLastDate = statDict[row['B_id']][-1]

            # get the last date a match was played on any surface by each player
            if playerDict[row['A_id']][-1] == 0:
                ALastDate = row['tourney_date']
            else:
                ALastDate = playerDict[row['A_id']][-1]
            if playerDict[row['B_id']][-1] == 0:
                BLastDate = row['tourney_date']
            else:
                BLastDate = playerDict[row['B_id']][-1]

            # update each player's highest rank
            if playerDict[row['A_id']][-3] > row['A_rank']:
                playerDict[row['A_id']][-3] = row['A_rank']
                
            if playerDict[row['B_id']][-3] > row['B_rank']:
                playerDict[row['B_id']][-3] = row['B_rank']



            # get the number of days since a match was played for each player
            ADaysSinceMatch = (row['tourney_date'] - ALastDate).days
            BDaysSinceMatch = (row['tourney_date'] - BLastDate).days

            # if a player hasn't played for three months, mark them down as returning from injury
            if ADaysSinceMatch > 90:
                df.at[index, 'A_injury'] = 1
            if BDaysSinceMatch > 90:
                df.at[index, 'B_injury'] = 1        

            # get a scale factor for decreasing the weighting placed on past matches
            AExpFactor = np.exp(-np.log(2)*(ADaysSinceMatch/self.halfLife))
            BExpFactor = np.exp(-np.log(2)*(BDaysSinceMatch/self.halfLife))

            # update the dataframe and then the player dictionary for each feature in turn, for both players
            for col in range(len(self.cumColumns)//2):
                df.at[index, 'c' + self.cumColumns[col]] = playerDict[row['A_id']][col]
                playerDict[row['A_id']][col] *= AExpFactor                                           # weight past matches
                playerDict[row['A_id']][col] += row[self.cumColumns[col]]
            for col in range(len(self.cumColumns)//2, len(self.cumColumns)):
                df.at[index, 'c' + self.cumColumns[col]] = playerDict[row['B_id']][col-len(self.cumColumns)//2]
                playerDict[row['B_id']][col-len(self.cumColumns)//2] *= BExpFactor
                playerDict[row['B_id']][col-len(self.cumColumns)//2] += row[self.cumColumns[col]]


            #### repeat the same process with the surface specific statistics and dictionary
            
            surfaceADaysSinceMatch = (row['tourney_date'] - surfaceALastDate).days
            surfaceBDaysSinceMatch = (row['tourney_date'] - surfaceBLastDate).days
            
            surfaceAExpFactor = np.exp(-np.log(2)*(surfaceADaysSinceMatch/self.halfLife))
            surfaceBExpFactor = np.exp(-np.log(2)*(surfaceBDaysSinceMatch/self.halfLife))

            for col in range(len(self.cumColumns)//2):
                df.at[index, 'sc' + self.cumColumns[col]] = statDict[row['A_id']][col]
                statDict[row['A_id']][col] *= surfaceAExpFactor
                statDict[row['A_id']][col] += row[self.cumColumns[col]]
            for col in range(len(self.cumColumns)//2, len(self.cumColumns)):
                df.at[index, 'sc' + self.cumColumns[col]] = statDict[row['B_id']][col-len(self.cumColumns)//2]
                statDict[row['B_id']][col-len(self.cumColumns)//2] *= surfaceBExpFactor
                statDict[row['B_id']][col-len(self.cumColumns)//2] += row[self.cumColumns[col]]


            ################################

            
            # update the player dictionaries with the date of the last match they played (i.e. this one)
            playerDict[row['A_id']][-1] = row['tourney_date']
            playerDict[row['B_id']][-1] = row['tourney_date']            

            # update the dataframe with statistics from the player dictionaries and surface specific dictionaries
            df.at[index, 'A_bad_loss'] = playerDict[row['A_id']][-4]
            df.at[index, 'B_bad_loss'] = playerDict[row['B_id']][-4]
            df.at[index, 'A_highest_rank'] = playerDict[row['A_id']][-3]
            df.at[index, 'B_highest_rank'] = playerDict[row['B_id']][-3]
            df.at[index, 'A_matches_played'] = playerDict[row['A_id']][-2]
            df.at[index, 'B_matches_played'] = playerDict[row['B_id']][-2]
            
            df.at[index, 'sA_matches_played'] = statDict[row['A_id']][-2]
            df.at[index, 'sB_matches_played'] = statDict[row['B_id']][-2]

            
            # increment number of matches played
            playerDict[row['A_id']][-2] += 1
            playerDict[row['B_id']][-2] += 1

            statDict[row['A_id']][-2] += 1
            statDict[row['B_id']][-2] += 1
            
            # update the number of close losses for a player
            if abs(row['score'][0] - row['score'][1]) == 1:
                playerDict[row['A_id']][-4] += 1
                playerDict[row['B_id']][-4] += 1                

            # update the list of surface-specific dictionaries with the modified dictionary
            surfaceStatList[surfaceIndex] = statDict

            # end loop

        # create transformed features as ratios of features (for surface and non-surface specific features)
        for l in ['cA', 'cB', 'scA', 'scB']:
            df[l + '_ace%'] = df[l + '_ace'] / df[l + '_svpt']
            df[l + '_df%'] = df[l + '_df'] / df[l + '_svpt']
            df[l + '_1stIn%'] = df[l + '_1stIn'] / df[l + '_svpt']
            df[l + '_1stWon%'] = df[l + '_1stWon'] / df[l + '_1stIn']
            df[l + '_2ndWon%'] = df[l + '_2ndWon'] / (df[l + '_svpt'] - df[l + '_1stIn'])
            df[l + '_BpSave%'] = df[l + '_bpSaved'] / df[l + '_bpFaced']


        # create new columns for the log of ranks
        df['A_log_rank_high'] = np.log(df['A_highest_rank'])
        df['B_log_rank_high'] = np.log(df['B_highest_rank'])

        # create features for proportion of matches not lost by a large margin ############################################################################
        df['A_toughness'] = 1 - (df['A_bad_loss'] / df['A_matches_played'])
        df['B_toughness'] = 1 - (df['B_bad_loss'] / df['B_matches_played'])

        # make a list of columns to drop (mostly features that have been used to create new features already)
        simpleDrop = []
        for stat in ['ace%', 'df%', '1stIn%', '1stWon%', '2ndWon%', 'BpSave%', 'age', 'rank', 'seed', 'rank_points', 'ht', 'log_rank_high', 'highest_rank', 'log_rank', 'injury', 'matches_played', 'toughness']:
            if ('cA_' + stat) in list(df):
                df[stat] = df['cA_' + stat] - df['cB_' + stat]
                simpleDrop.append('cA_' + stat)
                simpleDrop.append('cB_' + stat)
            if ('A_' + stat) in list(df):
                df[stat] = df['A_' + stat] - df['B_' + stat]                
                simpleDrop.append('A_' + stat)
                simpleDrop.append('B_' + stat)
            if ('scA_' + stat) in list(df):
                df['s' + stat] = df['scA_' + stat] - df['scB_' + stat]                
                simpleDrop.append('scA_' + stat)
                simpleDrop.append('scB_' + stat)
            if ('sA_' + stat) in list(df):
                df['s' + stat] = df['sA_' + stat] - df['sB_' + stat]                
                simpleDrop.append('sA_' + stat)
                simpleDrop.append('sB_' + stat)

        df['surface'] = pd.Series(np.zeros(len(df)), index=df.index)

        # filter the dataset to only include matches where the players have played 10 matches on the surface and in general
        dfFiltered = df.query('A_matches_played>10 and B_matches_played>10 and sA_matches_played>10 and sB_matches_played>10')

        # drop unnecessary columns from the dataframe
        extraDrop = ['cA_ace', 'cA_df', 'cA_svpt', 'cA_1stIn', 'cA_1stWon', 'cA_2ndWon', 'cA_SvGms', 'cA_bpSaved', 'cA_bpFaced', 'cB_ace', 'cB_df', 'cB_svpt', 'cB_1stIn', 'cB_1stWon', 'cB_2ndWon', 'cB_SvGms', 'cB_bpSaved', 'cB_bpFaced']
        extraDrop = extraDrop + ['s' + a for a in extraDrop]
        cumColumns = self.cumColumns + ['s' + a for a in self.cumColumns]
        dropColumns = cumColumns + simpleDrop + extraDrop + ['surface', 'A_id', 'B_id', 'A_hand', 'B_hand', 'A_entry', 'B_entry', 'score', 'A_bad_loss', 'B_bad_loss']
        dfFiltered.drop(columns = dropColumns, inplace=True)
        df = dfFiltered.reset_index(drop=True)

        self.dm.save_df(df, saveName)
        return df

    # train the logistic regression model on historical data
    def train(self, df):
        inputDF = df.copy()
        # remove index and high_rank_won features from the dataframe if they are in it
        if 'Unnamed: 0' in list(inputDF):
            inputDF.drop(columns = ['Unnamed: 0'], inplace = True)
        inputDF.drop(columns = ['high_rank_won'], inplace=True)

        # calculate the mean and standard deviation for each feature
        # this is to normalise features for the purposes of model stability
        self.means = []
        self.std = []
        for col in list(inputDF):
            self.means.append(inputDF[col].mean())
            self.std.append(inputDF[col].std())
            # perform normalisation 
            inputDF[col] = (inputDF[col] - inputDF[col].mean())/inputDF[col].std()
            
            # replace N/A values with the mean across all players
            inputDF[col].fillna((inputDF[col].mean()), inplace=True)
            
        # convert the mean and standard deviation lists into numpy float arrays
        self.means = np.float32([np.array(self.means)])
        self.std = np.float32([np.array(self.std)])

        # get slices of the dataframe to list the outcomes of matches and how many sets each was best of 
        outputs = df['high_rank_won'].copy()
        bestOf  = df['best_of'].copy()

        # convert to numpy arrays
        inputs = np.float32(inputDF.values)
        outputs = np.float32(outputs.values)
        bestOf = np.float32(bestOf.values)

        # set up the actual logistic regression aspect of the model

        # initialise bias as zero
        self.bias = 0
        
        # set up coefficients for each input feature
        self.coef = np.matrix([0 for a in range(inputs[0].size)])

        # create a numpy array for the values to update coefficients by after each iteration
        update = np.ones(self.coef.size)
        it = 0














        # for 25 passes through the dataset (determined empirically)
        while it < 25*len(inputs):
            # get a random match to train the model with
            a = random.randint(0, len(inputs)-1)
            inp = inputs[a]
            out = outputs[a]

            # get the prediction for the match based on the current model
            predict = self.train_predict(inp)

            # set the value of the dynamic learning rate (intended to start high and decrease as more training occurs)
            learnRate = self.learnRate * (5**0.6) / (5+(10*it/len(inputs)))**0.6

            # get the values to update the coefficients and the bias by to drive the model towards a stronger prediction on this test case
            update = learnRate*(out - predict)*(predict)*(1-predict)*inp
            self.bias += learnRate*(out - predict)*(predict)*(1-predict)
            # update the coefficient array by the update values
            self.coef = self.coef + update
            it+=1























    # save the parameters that have been learnt by a model in a text file
    def save_model(self, saveName):
        # save the bias value and coefficients required to make a prediction
        # save the mean and standard deviation values required to normalise data
        with open(saveName + '.txt','w') as file:
            string = ''
            string += str(self.bias) + '\n'
            for a in range(self.coef.size):
                string += str(self.coef[0,a]) + '\n'
            for b in range(self.coef.size):
                string += str(self.means[0,b]) + '\n'
            for c in range(self.coef.size):
                string += str(self.std[0,c]) + '\n'
                
            string = string[:-2]
            file.write(string)

    # load the parameters required for a model to make predictions on unseen data
    def load_trained_model(self, saveName):
        # read in the bias, coefficients, means and standard deviation and save them as attributes of the object
        with open(saveName + '.txt','r') as file:
            lines = file.read().split('\n')
            self.bias = float(lines.pop(0))
            self.coef = np.float32([np.array(lines[:len(lines)//3])])
            self.means = np.float32([np.array(lines[len(lines)//3:2*len(lines)//3])])
            self.std = np.float32([np.array(lines[2*len(lines)//3:])])

    # make a fast prediction using numpy arrays for training purposes
    def train_predict(self, inp):
        inp = np.matrix([inp])
        # matrix multiply the coefficients and input array for the result (and add the bias)
        result = (self.coef * inp.T)[0,0] + self.bias

        # make the prediction based on the logistic function (hence logistic regression)
        prediction = 1.0/(1+np.exp(-result))

        # assert that the prediction is in usual bounds
        # (to alert the user if something has broken during the model's operation)
        assert prediction <= 1 and prediction >= 0
        return prediction        

    # make a prediction from a row of a dataframe
    def predict(self, row):
        # convert the dataframe slice to a row
        inp = list(row.values)

        # replace N/A values with the mean of that the column they belong to
        nans = np.isnan(inp)
        for el in range(len(inp)):
            if nans[el]:
                inp[el] = self.means[0,el]

        # convert the input list to a numpy matric and normalise it
        inp = np.matrix([inp])
        inp = (inp-self.means)/self.std

        # get a result and prediction using the logistic function
        result = (self.coef * inp.T)[0,0] + self.bias

        prediction = 1.0/(1+np.exp(-result))

        assert prediction <= 1 and prediction >= 0
        return prediction        


# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

if __name__ == "__main__":    
    m=LogReg()
    predictions = m.make_prediction_set(2000,2016, "2000-2016 (it9)", testStart = 2015, slamsOnly = False, trainName = 'LogReg_final', inputProcessedFile= "2000-2016 (it9)-processed")
    e = Evaluate(predictions)
    print('Final log reg model')
    e.display_summary()
