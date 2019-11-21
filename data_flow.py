# file containing the DataManager class

# this class extracts data from csv files by year, and concatenates these csv files into a Pandas dataframe
# it also creates some common features models will need, as well as cleaning the data
# it also converts data from being winner and loser statistics to being high-rank player and low-rank player statistics
# (so models don't just learn to predict a winner based on the order of statistics)


import numpy as np
import random
import pandas as pd
import matplotlib.pyplot as plt
from score_parser import parse_score
import datetime


class DataManager(object):

    def __init__(self, path, columns):

        # set up some useful file paths
        self.mainPath = ".\\Match_data\\tennis_atp-master\\"
        self.savePath  = self.mainPath + path
        self.inputPath = self.mainPath + "atp_matches_{}.csv"
        

        # manipulate lists of features to create a list of columns to drop from the dataset
        allColumns = ['tourney_id', 'tourney_name', 'winner_ioc', 'loser_ioc', 'tourney_level', 'surface','draw_size','tourney_date','match_num','winner_id','winner_seed','winner_entry','winner_name','winner_hand','winner_ht','winner_age','winner_rank','winner_rank_points','loser_id','loser_seed','loser_entry','loser_name','loser_hand','loser_ht','loser_age','loser_rank','loser_rank_points','score','best_of','round','minutes','w_ace','w_df','w_svpt','w_1stIn','w_1stWon','w_2ndWon','w_SvGms','w_bpSaved','w_bpFaced','l_ace','l_df','l_svpt','l_1stIn','l_1stWon','l_2ndWon','l_SvGms','l_bpSaved','l_bpFaced','high_rank_won', "A_id", "A_seed", "A_entry", "A_name", "A_hand", "A_ht", "A_age", "A_rank", "A_rank_points", "A_ace", "A_df", "A_svpt", "A_1stIn", "A_1stWon", "A_2ndWon", "A_SvGms", "A_bpSaved", "A_bpFaced", "B_id", "B_seed", "B_entry", "B_name", "B_hand", "B_ht", "B_age", "B_rank", "B_rank_points", "B_ace", "B_df", "B_svpt", "B_1stIn", "B_1stWon", "B_2ndWon", "B_SvGms", "B_bpSaved", "B_bpFaced"]
        self.alwaysDrop = ['tourney_id', 'tourney_name', 'winner_ioc', 'loser_ioc', 'tourney_level']

        self.winner = ['winner_id', 'winner_seed', 'winner_entry', 'winner_name', 'winner_hand', 'winner_ht', 'winner_age', 'winner_rank', 'winner_rank_points', 'w_ace', 'w_df', 'w_svpt', 'w_1stIn', 'w_1stWon', 'w_2ndWon', 'w_SvGms', 'w_bpSaved', 'w_bpFaced']
        self.loser = ['loser_id', 'loser_seed', 'loser_entry', 'loser_name', 'loser_hand', 'loser_ht', 'loser_age', 'loser_rank', 'loser_rank_points', 'l_ace', 'l_df', 'l_svpt', 'l_1stIn', 'l_1stWon', 'l_2ndWon', 'l_SvGms', 'l_bpSaved', 'l_bpFaced']

        self.dropColumns = list(set(allColumns) - set(columns)-set(self.alwaysDrop)-set(self.winner+self.loser)-set(['tourney_date','best_of']))

    # compile a dataframe from yearStart to yearEnd composed of clean data
    def compile_clean_data(self, yearStart, yearEnd):
        c = 0
        d = 0
        compiledData = []
        
        for year in range(yearStart, yearEnd+1):

            # get the next year of data into df
            filename = self.inputPath.format(str(year))
            df = pd.read_csv(filename, parse_dates = [5], na_values = [''])

            # replace the score field with a tuple of scores in sets, or None for W/O or RET
            # this means that walkovers and retired matches will be dropped from the data
            df['score'] = df.score.apply(parse_score)

            # drop some columns that are always useless, before data is filtered
            df.drop(columns = self.alwaysDrop, inplace=True)

            # replace the seed value for unseeded players with the total number of players in the draw
            # so that this feature isn't dropped from the dataframe for having invalid values
            df.winner_seed.fillna(df.draw_size, inplace=True)
            df.loser_seed.fillna(df.draw_size, inplace=True)

            # replace the entry code for standard entry players with "S" for standard
            df.winner_entry.fillna('S', inplace=True)
            df.loser_entry.fillna('S', inplace=True)

            # drop walkovers and matches where a player retired from the dataset, as well as matches lacking data
            df.dropna(inplace=True)

            # reorder the columns in the dataframe to read high-rank / lower-rank rather than winner / loser
            df = self.rank_order(df)

            # add this processed dataframe to the list of processed dataframes
            compiledData.append(df)

        # concatenate all processed dataframes together, and sort this new dataframe by date of match
        compiledData = pd.concat(compiledData)
        compiledData.sort_values(['tourney_date', 'match_num'], inplace=True)

        # drop irrelevant columns from the dataframe
        compiledData.drop(columns = self.dropColumns, inplace=True)
        compiledData.reset_index(drop=True, inplace=True)

        return compiledData



    #instead of ordering by winner/loser, order by high rank with an additional column for if they won
    def rank_order(self,df):
        # create a new columns, for if the high rank player won
        df['high_rank_won'] = df['winner_rank'] < df['loser_rank']

        # create a new dataframe to populate
        newDF = df.copy()

        # create new columns for each player using the self.compare function applied to each row of the dataframe
        newDF["A_id"], newDF["A_seed"], newDF["A_entry"], newDF["A_name"], newDF["A_hand"], newDF["A_ht"], newDF["A_age"], newDF["A_rank"], newDF["A_rank_points"], newDF["A_ace"], newDF["A_df"], newDF["A_svpt"], newDF["A_1stIn"], newDF["A_1stWon"], newDF["A_2ndWon"], newDF["A_SvGms"], newDF["A_bpSaved"], newDF["A_bpFaced"], newDF["B_id"], newDF["B_seed"], newDF["B_entry"], newDF["B_name"], newDF["B_hand"], newDF["B_ht"], newDF["B_age"], newDF["B_rank"], newDF["B_rank_points"], newDF["B_ace"], newDF["B_df"], newDF["B_svpt"], newDF["B_1stIn"], newDF["B_1stWon"], newDF["B_2ndWon"], newDF["B_SvGms"], newDF["B_bpSaved"], newDF["B_bpFaced"] = zip(*newDF.apply(self.compare, axis = 1))

        # drop the winner / loser columns from the dataframe
        dropColumns = self.winner+self.loser
        newDF.drop(columns = dropColumns, inplace=True,axis=1)
        return newDF

    # return a set of columns in the right order given a set of columns in winner / loser order
    def compare(self, x):
        tup1 = tuple(x[w] for w in self.winner)
        tup2 = tuple(x[l] for l in self.loser)
        if x['high_rank_won']:
            return tup1 + tup2
        else:
            return tup2 + tup1

    # load a clean data file given its path
    def load_clean_data(self, filename):
        df = pd.read_csv(self.savePath + filename + '.csv')

        # format date as a date data type if it is a column in the dataframe
        if "tourney_date" in list(df):
            df['tourney_date'] = df['tourney_date'].astype('datetime64[ns]')
        return df

    # save a dataframe as a .csv given a path
    def save_df(self, df, saveName):
        df.to_csv(self.savePath + saveName + '.csv')
