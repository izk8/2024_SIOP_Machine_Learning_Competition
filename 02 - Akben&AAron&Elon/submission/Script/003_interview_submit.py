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
# ANCHOR Interview Prompt Task


df = df_interview.copy()

i = 0
for i in range(len(df)):
    df['questions_answers'].iloc[i] = "\nQuestion:".join(df['questions_answers'].iloc[i].split('Question:'))

# Strip the leading and trailing spaces
df['questions_answers'] = df['questions_answers'].apply(lambda x: x.strip())


# Parameters ----------------
dev_or_test = 'test' # dev or test or train
model = 'gpt-4-0125-preview' # | 'gpt-3.5-turbo'
temperature = 0.8
N_response = 5 # Set to 5
## -----------------------------


df = df[df['data'] == dev_or_test]
df['output'] = None
unique_ids = df['_id'].unique()


results = []



for i, id_i in enumerate(unique_ids):
    print("Processing: ", i+1, " of ", len(unique_ids))
    target_df = df[df['_id'] == id_i]

    # Clean the trailing or leading spaces
    task = target_df['questions_answers'].iloc[0].strip()

    
    # Split task by the question and response
    Questions = task.split("\nQuestion:")
    Responses = [x.split("\nResponse:") for x in Questions]
    # Unlist
    Responses = [item for sublist in Responses for item in sublist]
    # Remove empty strings
    Responses = [x for x in Responses if x]
    # Clean the leading and trailing spaces
    Responses = [x.strip() for x in Responses]
    Responses[0] = Responses[0].split("Question:")[1].strip()

    messages = []

    prompt = U.read_write_txt("Prompts/00_interview.txt")
    prompt = prompt.strip()

    last_question = ""
    conversation = ""
    for i in range(0, len(Responses), 2):
        if i == len(Responses)-1:
            last_question = "Question: " + Responses[i]
        else:
            question = "Question: " + Responses[i]
            conversation += question + "\n"
            # messages = U.chat_input("user",question + Responses[i], messages)
        try:
            response = "Response: "+Responses[i+1]
            conversation += response + "\n"
            # messages = U.chat_input("assistant",response, messages)
        except:
            pass
    conversation = conversation.strip()
    last_question = last_question.strip()

    simple_in = conversation+"\n"+last_question+"\nResponse:"
    
    prompt_in = prompt.format(conversation=conversation, final_question=last_question)
    messages = U.chat_input("user", prompt_in, [])

    response_list = []
    for i in range(N_response):
        resposnes = U.chat_completion_openAI(messages, model=model, temperature=temperature)
        response_list.append(resposnes)

    response_list = [x.strip() for x in response_list]
    response_list = [re.sub(r'\bDeliverable: \b', r'', text, re.DOTALL) for text in response_list]
    response_list = [re.sub(r'\bDeliverable:\n\b', r'', text, re.DOTALL) for text in response_list]
    response_list = [re.sub(r'\bDeliverable: \n\b', r'', text, re.DOTALL) for text in response_list]

    results.append([id_i]+response_list)


LLM_response = pd.DataFrame(results, columns=['_id'] + [f'predictions_{i}' for i in range(N_response)])
df = df.merge(LLM_response, on='_id', how='right')


## calculate cosine similarity between the last conversation and the predictions
df['previous_embeds'] = df['questions_answers'].apply(lambda x: U.embed_fn(x, model='transformer')[0])
for each in [f'predictions_{i}' for i in range(N_response)]:
    # Get Embeds
    df[f'{each}_embeds'] = df[each].apply(lambda x: U.embed_fn(x, model='transformer')[0])
    # Get Cosine
    df[f'cosine_{each}'] = df.apply(lambda x: np.dot(x['previous_embeds'], x[f'{each}_embeds']), axis=1)

# Select the highest cosine similarity in total
cosine_cols = df.columns[df.columns.str.contains('cosine')]
max_index = df[cosine_cols].mean(axis=0).sort_values(ascending=False).idxmax()

best_predict = "_".join(max_index.split('_')[1:])

# Combined Prediction 
all_preds = df[[f'predictions_{i}' for i in range(N_response)]]
all_preds = [" [CLS] ".join(x) for x in all_preds.values]

df['output'] = df[best_predict]
df['combined'] = all_preds

# For train, you can use the following code to calculate the cosine similarity between the last conversation and the output
# df['last_col_embeds'] = df.last_answer.apply(lambda x: U.embed_fn(x, model='transformer')[0])
# df['output_embeds'] = df.output.apply(lambda x: U.embed_fn(x, model='transformer')[0])
# df['cosine'] = df.apply(lambda x: np.dot(x['last_col_embeds'], x['output_embeds']), axis=1)
# df['cosine'].mean()

os.makedirs('Output', exist_ok=True)
df[['_id','output']].to_csv('Output/00_submit_interview.csv', index=False)