# TennisPredictor
Set of programs to process historical tennis data in order to make predictions on future matches.

Works using open source data compiled by Jeff Sackman (see https://github.com/JeffSackmann/tennis_atp)

log_reg_it9.py - the logistic regression model that maps the input parameters to an output probability of winning the match

data_flow.py   - unpacks data from the database using pandas

model.py - defines the Model superclass which is used to generalise how "models" make predictions

evaluate.py / evaluate_SOTA.py - evaluates the performance of models using various metrics (e.g. Brier score)

score_parser.py - parses strings containing match scores into useful numeric data
