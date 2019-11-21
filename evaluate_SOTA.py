#file containing an alternative Evaluation class, that uses state-of-the-art metrics to evaluate model performance


import numpy as np
from prettytable import PrettyTable
import time

# input data is entered as a 2xN matrix of the format
#   [ prediction 1, outcome 1
#     prediction 2, outcome 2
#     ...                ,   ....           ]
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

    # get a measure of discrimination defined differently to in my main EPQ
    # d = mean prediction for matches where the high-rank player won - mean prediction for matches where the low-rank player won 
    def get_discrimination(self):
        highMatches = 0.0
        highProbabilitySum = 0.0
        lowMatches = 0.0
        lowProbabilitySum = 0.0
        for match in self.data:
            prob = match[0]
            outcome = match[1]
            
            if outcome == 1:
                highProbabilitySum += prob
                highMatches += 1
            else:
                lowMatches += 1
                lowProbabilitySum += prob

        discrimination = (highProbabilitySum/highMatches) - (lowProbabilitySum/lowMatches)

        return round(100*discrimination, 2)

    # get calibration, the same as usual in Evaluate
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

    # get the Log Loss metric, defined as the average logarithm of predicted probabilities for the winning player
    def get_log_loss(self):
        matches = 0.0
        logSum = 0.0
        for match in self.data:
            prob = match[0]
            outcome = match[1]

            # if outcome == 1, this is the log of predicted probability for the winner, i.e. the high-rank player
            # if outcome == 0, this is the log of predicted probability for the winner, i.e. the low-rank player
            # therefore it is always the log of predicted probability for the winner
            logSum += (outcome *np.log(prob) + (1-outcome) *np.log(1-prob))
            
            matches += 1

        LL = -logSum / matches
            
        return round(LL,2)

    # display a neat tabular summary
    def display_summary(self):

        x = PrettyTable()
        x.field_names = ["Metric", "Score"]

        x.add_row(["Correct predictions", str(self.get_perc_correct()) + '%'])
        x.add_row(["Discrimination", str(self.get_discrimination()) + '%'])
        x.add_row(["Calibration", str(self.get_calibration()) + '%'])
        x.add_row(["Log Loss", str(self.get_log_loss())])
        x.add_row(["Matches predicted", str(len(self.data))])

        print(x)
        input()
