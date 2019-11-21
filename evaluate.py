#file containing the Evaluation class, that takes the output of models and evaluates their performance

import numpy as np
from prettytable import PrettyTable
import time

# input data is entered as a 2xN matrix of the format
#   [ prediction 1, outcome 1
#     prediction 2, outcome 2
#     ...                ,   ....
#     prediction N, outcome N]
# each prediction is the probability of the higher ranked player winning
# each outcome is Boolean, with 1 representing higher-ranked player winning and 0 representing an upset


class Evaluate(object):

    # take in input data
    def __init__(self, data):
        self.data = data

    # return the percentage of matches correctly predicted
    def get_perc_correct(self):
        total = 0.0
        correct = 0.0
        for match in self.data:
            prob = match[0]
            outcome = match[1]

            # if outcome = 0, prob should be less than 0.5 to reflect a correctly predicted upset
            # --> prob - outcome < 0.5
            # if outcome = 1, prob should be more than 0.5 to reflect a correctly predicted match
            # --> outcome - prob > 0.5
            # so "abs(outcome - prob) < 0.5" is a suitable constraint to reflect a correctly predicted match
            
            if abs(outcome - prob) < 0.5:
                correct += 1
            total += 1

        percCorrect = correct / total
        return round(100*percCorrect, 2)

    
    def get_discrimination(self):
        # for each correctly predicted match, add its probability to the probability sum and add one to the number of matches
        correctMatches = 0.0
        correctProbabilitySum = 0.0
        
        # repeat for each incorrectly predicted match,
        incorrectMatches = 0.0
        incorrectProbabilitySum = 0.0
        
        for match in self.data:
            prob = match[0]
            outcome = match[1]
            
            if abs(outcome - prob) < 0.5:
                correctMatches += 1
                correctProbabilitySum += prob
            else:
                incorrectMatches += 1
                incorrectProbabilitySum += prob

        # catching div by zero errors and calculating mean probabilities
        if correctMatches != 0:
            meanCorrect = correctProbabilitySum / correctMatches
        else:
            meanCorrect = 1
        if incorrectMatches != 0:
            meanIncorrect = incorrectProbabilitySum / incorrectMatches
        else:
            meanIncorrect = 0

        # return the discrimination 
        discrimination = meanCorrect - meanIncorrect
        return round(100*discrimination, 2)

    # use the definition of calibration ratio to return the statistic on calibration
    def get_calibration(self):
        probSum = 0.0
        highRankWins = 0.0
        for match in self.data:
            prob = match[0]
            outcome = match[1]
            
            probSum += prob
            if outcome == 1:
                highRankWins += 1

        calibration = probSum / highRankWins
        return round(100*calibration, 2)

    # get the mean squared error - i.e. square all the errors and find the average of these squares
    def get_brier_score(self):
        matches = 0.0
        sumOfSquares = 0.0
        for match in self.data:
            prob = match[0]
            outcome = match[1]

            sumOfSquares += (prob - outcome)*(prob - outcome)
            
            matches += 1

        BS = sumOfSquares / matches
            
        return round(100*BS, 2)

    # display a summary table of metrics and scores
    def display_summary(self):

        x = PrettyTable()
        x.field_names = ["Metric", "Score"]

        x.add_row(["Correct predictions", str(self.get_perc_correct()) + '%'])
        x.add_row(["Discrimination", str(self.get_discrimination()) + '%'])
        x.add_row(["Calibration", str(self.get_calibration()) + '%'])
        x.add_row(["Brier Score", str(self.get_brier_score()) + '%'])
        x.add_row(["Matches predicted", str(len(self.data))])

        print(x)
        input()


    def get_average_performance(self, modelClass, iterations, label):
        m=modelClass()

        # make an input file for testing and training models so it doesn't need to be done for each new iteration
        df = m.get_input_data(2000, 2016)
        df = m.process_data(df, "performance_data" + str(label))

        # repeatedly train and then evaluate a model to get an average score and standard deviation for each metric
        statList = [np.zeros(iterations), np.zeros(iterations), np.zeros(iterations), np.zeros(iterations)]        
        for it in range(iterations):
            m = modelClass()
            self.data = m.make_prediction_set(2000,2016, "performance_data" + str(label), inputProcessedFile = "performance_data" + str(label), testStart = 2015)
            statList[0][it] = self.get_perc_correct()
            statList[1][it] = self.get_discrimination()
            statList[2][it] = self.get_calibration()
            statList[3][it] = self.get_brier_score()

        # display the results
        x = PrettyTable()
        x.field_names = ["Metric", "Mean Score", "Standard Deviation"]

        x.add_row(["Correct predictions", str(round(statList[0].mean(), 2)) + '%', str(round(statList[0].std(), 2))])
        x.add_row(["Discrimination", str(round(statList[1].mean(), 2)) + '%', str(round(statList[1].std(), 2))])
        x.add_row(["Calibration", str(round(statList[2].mean(), 2)) + '%', str(round(statList[2].std(), 2))])
        x.add_row(["Brier Score", str(round(statList[3].mean(), 2)) + '%', str(round(statList[3].std(), 2))])

        print(x)
        input()


