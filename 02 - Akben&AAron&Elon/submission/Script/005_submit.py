#########################################################################
#########################################################################
# ANCHOR Libraries
import pathlib
from utils import EvalAI
import utils as U
import sklearn
import os
import sys
import re
import pandas as pd
import numpy as np
import tqdm
import json


#########################################################################
#########################################################################
# ANCHOR Functions

clc = U.clc

#########################################################################
#########################################################################
# ANCHOR Combine and submit

# Empathy size for test 100
# Interview size for test 45
# Clarity size for test 70
# Fairness size for test 58

res_empathy = pd.read_csv(r'Output/00_submit_empathy.csv')
res_empathy.columns = ['_id', 'output']
res_empathy['benchmark'] = "empathy"

res_interview = pd.read_csv(r'Output/00_submit_interview.csv')
res_interview = res_interview[['_id', 'output']]
res_interview['benchmark'] = "interview"


res_clarity = pd.read_csv(r'Output/00_submit_clarity.csv')
res_clarity.columns = ['_id', 'output']
res_clarity['benchmark'] = "clarity"

res_fairness = pd.read_csv(r'Output/00_submit_fairness.csv')
res_fairness = res_fairness[['_id', 'output']]
res_fairness['benchmark'] = "fairness"

# Order the columns benchmark, _id, output
res_empathy = res_empathy[['benchmark', '_id', 'output']]
res_interview = res_interview[['benchmark', '_id', 'output']]
res_clarity = res_clarity[['benchmark', '_id', 'output']]
res_fairness = res_fairness[['benchmark', '_id', 'output']]

min_s = res_clarity['output'].min()
max_s = res_clarity['output'].max()

res_clarity['output'] = (((res_clarity['output'] - min_s) / (max_s - min_s))*1)+4


df_output = pd.concat([res_empathy, res_interview,res_clarity, res_fairness], axis=0, ignore_index=True)

path = f'Output/SUBMISSION_TEST.csv'
df_output.to_csv(path, index=False)


#########################################################################
#########################################################################
# ANCHOR Eval AI Submit


# Create an instance of the EvalAI API
evalai = EvalAI()

# Set the API Token
token = r"EvalAITOken.json"

evalai.set_token(pathlib.Path(token))
evalai.get_team()
all_submit = evalai.get_all_submissions()

is_submitted = evalai.submit(csv_file_path=path, test=True)
if is_submitted:
    print("Submission is successful")
else:
    print("Submission is not successful")
    print("Please check the error message on the website")
    
