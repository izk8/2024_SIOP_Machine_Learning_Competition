#########################################################################
#########################################################################
# ANCHOR Libraries
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
# ANCHOR Load the Data

df_empathy, df_interview, df_clarity, df_fairness = U.loadData()

#########################################################################
#########################################################################
# ANCHOR Identifying Fairness Perceptions | Accuracy - Base Model


df = df_fairness[df_fairness['data'].isin(['train', 'dev', 'test'])].copy()

## Parameters ----------------
dev_or_test = "test"  # "dev" or "test"
N_sample = 5
# ----------------


unique_id = df[df['data'].isin([dev_or_test])]['_id'].values
sample_id = df[df['data'].isin(['train'])]['_id'].values

results = []
for i in range(len(unique_id)):
    id_i = unique_id[i]
    print(f"Processing {i+1} of {len(unique_id)}")

    test_i = df[df['_id'] == id_i]
    target_id = test_i['_id'].values[0]

    target1 = test_i['first_option'].values[0]
    target2 = test_i['second_option'].values[0]

    n_shot = 20

    main_sample = df_fairness[df_fairness['_id'].isin(sample_id)]
    # Ensure that the target_id is not in the train_n
    main_sample = main_sample[main_sample['_id'] != target_id]

    train_n = main_sample.sample(n_shot)

    self_consistency = []
    for each in range(N_sample):

        prompt = "Which policy statement is the much fairer option?\n# Option 1: {option1}\n# Option 2: {option2}\n # Answer: {answer}"

        prompt_in = ""

        for i in range(n_shot):
            prompt_in += "\n\nQuestion: " + \
                prompt.format(option1=train_n.iloc[i]['first_option'], option2=train_n.iloc[i]
                              ['second_option'], answer=train_n.iloc[i]['majority_vote'])

        test_prompt = "\n\nQuestion: " + "Which one is the much fairer option?\n# Option 1: {option1}\n# Option 2: {option2}\n # Answer: {answer}"
        test_prompt = test_prompt.format(
            option1=target1, option2=target2, answer="")
        test_prompt = test_prompt.strip()

        prompt_in = prompt_in.strip()
        prompt_in += "\n\n" + test_prompt

        messages = U.chat_input('user', prompt_in, [])
        response = U.chat_completion_openAI(messages, model="gpt-4-0125-preview", temperature=0.5, n=1)
        response = response.strip()
        self_consistency.append([id_i, response])

    results.append(pd.DataFrame(self_consistency, columns=['_id', 'output']))


# Count how many times there is `first` or `second` in the output
llm_res = pd.concat(results)
process = []
for each_id in llm_res['_id'].unique():
    temp = llm_res[llm_res['_id'] == each_id]
    first = temp[temp['output'] == "first"].shape[0]
    second = temp[temp['output'] == "second"].shape[0]
    process.append([each_id, first, second])
llm_response = pd.DataFrame(process, columns=['_id', 'first', 'second'])
llm_response['output'] = llm_response.apply(lambda x: "first" if x['first'] > x['second'] else "second", axis=1)

# Merge the data and save as an output
df = df.merge(llm_response, on='_id', how='right')
# Strip the leading and trailing spaces
df['output'] = df['output'].str.strip()
df['output'] = df['output'].apply(lambda x: "first" if x == "first" else "second")

os.makedirs('Output', exist_ok=True)
df.to_csv(r'Output/00_submit_fairness.csv', index=False)