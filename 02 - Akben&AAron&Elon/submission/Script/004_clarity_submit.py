#########################################################################
#########################################################################
### ANCHOR Libraries
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
### ANCHOR Functions

clc = U.clc

#########################################################################
#########################################################################
# ANCHOR Load the Data

df_empathy, df_interview, df_clarity, df_fairness = U.loadData()


#########################################################################
#########################################################################
### ANCHOR Encode clarity scores from 1 to 5

min_s = df_clarity['clarity'].min()
max_s = df_clarity['clarity'].max()

df_clarity['clarity'] = ((df_clarity['clarity']-min_s)/(max_s-min_s)*4)+1
df_clarity['org_clarity'] = df_clarity['clarity']
df_clarity['clarity'] = df_clarity['clarity'].round(2)

df_clarity['text'] = df_clarity['personality_item'].str.strip()
df_clarity['score'] = df_clarity['clarity']


#########################################################################
#########################################################################
### ANCHOR Prediction MASTERS

MASTER_PREDICTIONS = df_clarity.copy()
MASTER_PREDICTIONS.shape
MASTER_PREDICTIONS['predictions'] = 0


#########################################################################
#########################################################################
### ANCHOR Clarity Rule Based Model


df = df_clarity.copy()

# Parameters ----------------
dev_test = 'test' # dev or test
tempature = 0.20
# Select the model
model_type = "openai" # openai or antropic

if model_type == "openai":
    model = "gpt-3.5-turbo-0125"
    function = U.chat_completion_openAI
else:
    # IF ANTHROPIC
    # Select the function 
    model = U.model_anthropic[1] # Opus
    function = U.chat_completion_anthropic

prediction_names = f"Expert_{model}_{tempature}"
# Parameters ----------------


# Select the data
df = df[df.data.isin([dev_test])]

# Sort by clarity and make them a full sentence
df = df.sort_values('clarity')
df['text']  = df['personality_item'].str.strip()
df['text']  = df['text'].str.lower()
df['text'] = "I " + df['text']

unique_ids = df._id.unique()
results = []


prompt_mapping = {
        'passive': "Is this text written in a passive voice, such as the cake was baked by Emily, or I am considered by my peers to be talented, or etc? Return only Yes or No.\n",
        'negative': "Is this text negatively worded such as `not`, `none` etc.? Return only Yes or No.\n",
        'compound': "Does this text use a compound sentence? Return only Yes or No.\n",
        'complex': "Does this text use a complex sentence? Return only Yes or No.\n",
        'simple': "Does the text use simple, everyday language that a lay person can understand? Return only Yes or No.\n",
        'jargon': "Is the text free of jargon or technical terms? Return only Yes or No.\n",
        'focused': "Does the text express a single, focused thought or action? Return only Yes or No.\n",
        'vague': "Is the text free of vague or ambiguous wording? Return only Yes or No.\n",
        'active': "Does the text use strong, active verbs? Return only Yes or No.\n",
        'easy': "Is the text easy to read and understand on the first pass? Return only Yes or No.\n",
        'easy2': "Is the text easy to read and understand for a third grader? Return only Yes or No.\n",
        'unnecessary': "Is the text free of unnecessary words or phrases? Return only Yes or No.\n",
        'double': "Does the text contains `double negatives` or `convoluted phrasing` such as `I do not dislike` or so on? Return only Yes or No.\n",
        'vocabulary': "Is the vocabulary of the text appropriate for a high school educated reader? Return only Yes or No.\n",
        'clarity': "Is this text clear for a third-grader to read? Return only Yes or No.\n"
    }


def rate_text(prompt_type, text, model, temperature):
    prompt = prompt_mapping[prompt_type] + "# Text\n" + text
    # print(prompt)
    messages = U.chat_input("user", prompt, [])
    res = function(messages, model=model, temperature=temperature)
    return res.strip()


# Main loop for processing each ID
results = []
for each, id_i in enumerate(unique_ids):
    print("Processing", each, len(unique_ids), "ID:", id_i)
    
    target_df = df[df._id == id_i]
    target_text = target_df['text'].values[0]
    target_id = target_df['_id'].values[0]
    target_score = target_df['clarity'].values[0]
    
    ratings = {}
    aspects = prompt_mapping.keys()
    
    ratings['target'] = target_score
    ratings['id'] = target_id
    ratings['text'] = target_text
    
    for aspect in aspects:
        ratings[aspect] = rate_text(aspect, target_text, model, tempature)
    
    results.append(ratings)


LLM_res = pd.DataFrame(results)

rating_i = (LLM_res.loc[:,'passive':] == 'Yes')*1
rating_i['id'] = LLM_res['id']
rating_i['text'] = LLM_res['text']

rating_i['target'] = LLM_res['target']


negative_items =['passive','negative','compound','complex','double']
rating_i[negative_items] = 1-rating_i[negative_items]

# Total Score
rating_cols = rating_i.columns[~rating_i.columns.str.contains('id|text|target')]
# Combine scores
rating_i[f'total_{prediction_names}'] = rating_i[rating_cols].sum(axis=1)
rating_i[f'top_k_{prediction_names}'] = rating_i[['easy2','simple','vocabulary']].sum(axis=1)

min_s = rating_i[f'total_{prediction_names}'].min()
max_s = rating_i[f'total_{prediction_names}'].max()
rating_i[f'total_{prediction_names}'] = ((rating_i[f'total_{prediction_names}']-min_s)/(max_s-min_s)*4)+1

min_s = rating_i[f'top_k_{prediction_names}'].min()
max_s = rating_i[f'top_k_{prediction_names}'].max()
rating_i[f'top_k_{prediction_names}'] = ((rating_i[f'top_k_{prediction_names}']-min_s)/(max_s-min_s)*4)+1

# For training, you can check this correlation
# rating_i.select_dtypes(include=np.number).corr()['target'].sort_values(ascending=False)

# Save the predictions 
MASTER_PREDICTIONS = MASTER_PREDICTIONS.merge(rating_i[['id',f'total_{prediction_names}',f'top_k_{prediction_names}']], left_on='_id', right_on='id', how='left')


#########################################################################
#########################################################################
### ANCHOR Model 2: Expert Ratings


df = MASTER_PREDICTIONS.copy()
df['text'] = "I " + df['personality_item'].str.lower()

expert_cols = df.columns[df.columns.str.contains('top_k')]
# For each expert_cols reorder the names as expert1, expert2, ...
df.rename(columns={expert_cols[i]:f'expert{i+1}' for i in range(len(expert_cols))}, inplace=True)


# Parameters ----------------
dev_test = 'test' # dev or test
tempature = 0.30
# Select the model
model_type = "openai" # openai or antropic

if model_type == "openai":
    model = "gpt-4-0125-preview" # "gpt-3.5-turbo-0125"
    function = U.chat_completion_openAI
else:
    # IF ANTHROPIC
    # Select the function 
    model = U.model_anthropic[0] # Opus
    function = U.chat_completion_anthropic

prediction_names = f"FinModel_{model}_{tempature}"
# Parameters ----------------



unique_ids = df[df.data.isin([dev_test])]._id.unique()


prompt = U.read_write_txt("Prompts/00_clarity_expert_ratings.txt")

results = []
expert_cols = df.columns[df.columns.str.contains('expert')]


for i, id_i in enumerate(unique_ids):
    print("Processing", id_i, "out of", len(unique_ids))

    target_df = df[df._id == id_i]

    target_truth = target_df['score'].values[0]

    target_text = target_df['text'].values[0]

    expert_ratings = target_df[expert_cols].values[0]
    expert_rating_in = ""
    for i, ratings in enumerate(expert_ratings):
        expert_rating_in += f"Expert {i+1} Rating: {ratings}\n"
    expert_rating_in = expert_rating_in.strip()
    prompt_in = prompt.format(target_text=target_text,expert_ratings=expert_rating_in)

    messages = U.chat_input("user", prompt_in, [])
    LLM_response = function(messages, model=model, temperature=tempature)
    LLM_response = LLM_response.strip()

    try:
        score_extracted = re.findall(r'<rating>(.*?)</rating>', LLM_response, re.DOTALL)[0]
    except:
        score_extracted = ""

    results.append([id_i, LLM_response, score_extracted])

    print("Actual:", target_truth, "Predicted:", score_extracted)


res_df = pd.DataFrame(results, columns=['_id', 'LLMResponse', 'preds'])
res_df['preds'] = res_df['preds'].astype(float)

res_df = df.merge(res_df, left_on='_id', right_on='_id', how='right')
res_df = res_df[['_id','score','preds']+expert_cols.tolist()]
res_df['fin_predict'] = res_df[['preds']+expert_cols.tolist()].mean(axis=1)

# res_df.select_dtypes(include=np.number).corr()['score'].sort_values(ascending=False)
os.makedirs("Output", exist_ok=True)
res_df[['_id','fin_predict']].to_csv(r'Output/00_submit_clarity.csv', index=False)