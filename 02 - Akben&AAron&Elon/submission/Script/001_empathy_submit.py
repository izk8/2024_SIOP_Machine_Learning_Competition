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

from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

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
### ANCHOR Set the working df

df = df_empathy.copy()
df_empathy['score'] = df_empathy['empathy']


#########################################################################
#########################################################################
### ANCHOR Model Prediction Dadtaframe


MASTER_PREDICTIONS = df_empathy.copy()
MASTER_PREDICTIONS.shape
MASTER_PREDICTIONS['predictions'] = 0


#########################################################################
#########################################################################
### ANCHOR Preprocessing: COT Generator for COT Model


RUN = False

if RUN:
    df = df_empathy.copy()
    unique_ids = df[df.data.isin(['train'])]['_id'].unique()

    prompt_path = "Prompts/00_COT_generator.txt"

    prompt = U.read_write_txt(prompt_path)

    feedback_target_dict = {0: "Self-Centric", 1: "Other-Centric"}
    
    class1 = feedback_target_dict[0]
    class2 = feedback_target_dict[1]    
    
    cot_examples = []

    for order, i in enumerate(unique_ids):
        print("-"*50)
        print(f"Processing {order} / {len(unique_ids)}")
        target_df = df[df._id == i]
        target_id = target_df['_id'].values[0]
        target_text = target_df['text'].values[0]
        target_score = target_df['score'].values[0]
        data_type = target_df['data'].values[0]
        
        target_text = target_text.strip()
        
        prompt_in = prompt.strip()
        prompt_in = prompt.format(text=target_text,
                                  feedback_target=feedback_target_dict[target_score],
                                  class1=class1,
                                  class2=class2)
        
        messages = U.chat_input("user",prompt_in, [])
        model = U.model_anthropic[0]
        LMM_response = U.chat_completion_anthropic(messages,model=model,temperature=0.8)
        
        # messages = U.chat_input("user",prompt_in, [], gemini=True)
        # model = "gemini2"
        # LMM_response = U.chat_completion_google(messages, model=model,temperature=0.8)
        print(LMM_response)
        cot_examples.append([target_id,data_type, target_text, LMM_response, target_score])
        print("-"*50)

    df = pd.DataFrame(cot_examples, columns=['_id','data','actual_text','cotExample','score'])
    df['cotExample'] = df['cotExample'].str.strip()

    df = df_empathy.merge(df[['_id','cotExample']], on='_id', how='left')

    path_to_extractive_summary = f"Data/combinedData/empathy_cot_self_other.csv"
    df.to_csv(path_to_extractive_summary, index=False)




#########################################################################
#########################################################################
### ANCHOR Preprocessing: Extract Only Feedback or Empathic Sentences


RUN = False

## Parameter----------------------
empath_or_feedback_extraction = 'empathy' # 
## Parameter----------------------

if RUN:
    df = df_empathy.copy()
    unique_ids = df._id.unique()

    if empath_or_feedback_extraction == 'empathy':
        prompt_path = "Prompts/00_empathyExtraction.txt"
    else:
        prompt_path = "Prompts/00_feedbackExtraction.txt"

    prompt = U.read_write_txt(prompt_path)

    main_feedback = []


    for order, i in enumerate(unique_ids):
        print("-"*50)
        print(f"Processing {order} / {len(unique_ids)}")
        target_df = df[df._id == i]
        target_id = target_df['_id'].values[0]
        target_text = target_df['text'].values[0]
        target_score = target_df['score'].values[0]
        data_type = target_df['data'].values[0]
        
        target_text = target_text.strip()
        
        prompt_in = prompt.strip()
        prompt_in = prompt_in.format(feedback=target_text)
        
        messages = U.chat_input("user",prompt_in, [])
        model = U.model_anthropic[1]
        LMM_response = U.chat_completion_anthropic(messages,model=model,temperature=0.3)
        print(LMM_response)
        try:
            response = re.findall(r'<response>(.*?)</response>', LMM_response, re.DOTALL)
            response = [i.strip() for i in response]
            response = " ".join(response)
        except:
            print("Error in response")
            response = [""]
        
        main_feedback.append([target_id,data_type, target_text, LMM_response, response, target_score])


    recode_text = "No empathetic feedback"

    df = pd.DataFrame(main_feedback, columns=['_id','data','actual_text','LLMResponse','text','score'])
    df['text'] = df['text'].str.strip()
    df['text'][df['text'].str.contains("No empathetic feedback")] = "This feedback does not contain any empathetic feedback."


    df['_id'].nunique()
    path_to_extractive_summary = f"Data/combinedData/FeedbackExtracted_{empath_or_feedback_extraction}.csv"
    df.to_csv(path_to_extractive_summary, index=False)

path_to_extractive_summary = f"Data/combinedData/FeedbackExtracted_{empath_or_feedback_extraction}.csv"


#########################################################################
#########################################################################
### ANCHOR Model 1
# 
# Model 1a: Text Completion Task: Context Learning with N-Shot Examples with Foundational Models
# Model 1b: Context-Learning with Self-Consistency N-Shot COTs using Chat LLM models


df = df_empathy.copy()
df['text'] = df['text'].str.strip()


### Set parameters ----------------------
dev_or_test = 'test' # dev set or "test"
# Use feedback without greetings and other flufs
feedback_only = True
## Self-Consisitency Parameter
n_repeat = 5
# N-Shot Examples
example_sample = 20
# Shuffle classes randomly
class_shuffle = True
# Temperature 
temperature = 0.40
# There is two option, chat or completion (model 1a and model 1b)
# Chat is roughly for RLHF models and will call COT n-shot 
# Completion is for foundational LLM models such as babbage, gpt3, palm, etc.
# If you go with the chat, you should change the model to gpt-4-0125-preview or other strong model
chat_or_completion = 'completion' 
# Select Model for the chat | completion
model = U.model_text_completion[2]
# Assign a prediction name to hold it on MASTER_PREDICTIONS
prediction_name = f"{model}_{chat_or_completion}_{temperature}_{n_repeat}_{example_sample}_{class_shuffle}_{feedback_only}" 
### --------------------------------------


path_to_extractive_summary = f"Data/combinedData/FeedbackExtracted_empathy.csv"


## Use only feedback without greeting and other parts
if feedback_only:
    df_extracted = pd.read_csv(path_to_extractive_summary)
    df_extracted.rename(columns={'text':'shorText'}, inplace=True)
    df_extracted = df_extracted[['_id','shorText']]
    df['longText'] = df['text']
    df = df.merge(df_extracted, on='_id', how='left')
    df['text'] = df['shorText'].str.strip()
    df['text'] = df['text'].str.replace("\n", " ")


# Class Encode
class_dict = {0:"Direct", 1:"Encouraging"}
# class_dict = {0:"<0>", 1:"<1>"} ## For the chat models, use this
df['class'] = df['score'].apply(lambda x: class_dict[1] if x==1 else class_dict[0])


## Unique Ids
if dev_or_test == 'dev':
    unique_ids = df[df.data.isin(['dev'])]['_id'].unique()
else:
    unique_ids = df[df.data.isin(['test'])]['_id'].unique()
sample_ids = df[df.data.isin(['train'])]['_id'].unique()


# Response Matrix
response_matrix = np.zeros((len(unique_ids), n_repeat))

# Main Prompt for the chat
prompt = U.read_write_txt("Prompts/00_Self_Consistency.txt")

# Save problematic ids
problematic_ids = []



for ith, id_target in enumerate(unique_ids):
    print(f"Processing {ith+1}/{len(unique_ids)}")
    # Select the target
    target = df.loc[df._id==id_target,:]
    
    # Select sample df
    rest = df.loc[df._id.isin(sample_ids),:]
    
    # select_ cols
    selected_cols = ['_id','text','class']
    
    target = target[selected_cols]
    rest = rest[selected_cols]
    
    classes = rest['class'].unique()
    class1 = rest[rest['class'] == classes[0]]
    class2 = rest[rest['class'] == classes[1]]
    
    max_attempts = 4  # Set max attempts for retries
    
    # Process each unique_id
    for jth in range(n_repeat):
        attempts = 0
        success = False
        
        while attempts < max_attempts and not success:
            attempts += 1
            try:
                sample1 = class1.sample(example_sample//2)
                sample2 = class2.sample(example_sample//2)
                
                # Text Input
                sample_in = pd.concat([sample1, sample2]).reset_index(drop=True)
                if class_shuffle:
                    sample_in = sample_in.sample(frac=1)
                rest_text = sample_in.to_dict('records')
                
                sample_text = ""
                for i,_ in enumerate(rest_text):
                    sample_text = sample_text + str(rest_text[i]) + "\n"
                
                # Delete all \' from the text
                sample_text = re.sub("\'", "", sample_text)        
                sample_text = re.sub(r"\"", "", sample_text)

                target_text = target.to_dict('records')[0]
                try:
                    actual_target = target_text.pop("class")
                    if dev_or_test in ['dev','test']:
                        actual_target = 'Test Data No Label'
                except:
                    actual_target = "test_dev_data"

                target_text = str(target_text)
                target_text = re.sub("\'", "", str(target_text))
                target_text = re.sub(r"\"", "", str(target_text))
                
                target_text = target_text[:-1]
                target_text = target_text + ", class:"
                
                prompt_in = sample_text + target_text
                
                
                # Call the LLM based on the mode
                if chat_or_completion != "chat":
                    ## LLM Completion Models (Like Gpt3 and Google Palm)
                    if model in U.model_text_completion:
                        LLM_response = U.text_completion_gpt3(prompt_in, model=model,temperature=temperature,stop=['\n'],max_tokens=30)
                    else:
                        LLM_response = U.text_completion_google(prompt_in,model=model,temperature=temperature,stop=['\n'],verbose=False)
                else:
                    # Chat Completion Models (with RLHF Models)
                    prompt_in = sample_text + "\n\n# Target Text:\n" + target_text[:-7] + "}"
                    system_message = prompt.format(class0=class_dict[0], class1=class_dict[1])
                    
                    if model in U.model_openai:
                        # OpenAI
                        in_text = []
                        in_text = U.chat_input('user',prompt_in,in_text)
                        in_text = U.chat_input('system',system_message,in_text)
                        LLM_response = U.chat_completion_openAI(in_text, model=model,temperature=temperature)
                        # print(LLM_response)
                    elif model in U.model_anthropic:
                        # Anthropic
                        in_text = []
                        in_text = U.chat_input('user',prompt_in,in_text)
                        LLM_response = U.chat_completion_anthropic(in_text,model=model,system=system_message,temperature=temperature)
                    
                    LLM_response = re.findall(f"<classification>(.*)</classification>", LLM_response.strip(),re.DOTALL)[0]

                # Delete all whitespaces
                LLM_response = re.sub(r"\s+", "", LLM_response)
                # Extract and process response
                response = re.findall(f"{class_dict[0]}|{class_dict[1]}", LLM_response)[0].strip()
                
                # Update the response matrix
                response_value = 1 if response == class_dict[1] else 0
                response_matrix[ith, jth] = response_value
                
                
                success = True  # Mark as success to exit loop
            except Exception as e:
                print(f"Attempt {attempts} failed due to error: {e}")
                if attempts < max_attempts:
                    print("Retrying...")
                else:
                    print("Max retry attempts reached. Moving to the next.")
                    print("Error in the ID:", id_target)
                    problematic_ids.append(id_target)



# Create a dataframe with the response matrix
df_response = pd.DataFrame(response_matrix,columns=[f"{prediction_name}_response_{i}" for i in range(n_repeat)])

# Count 0 and 1 by rows
count_0 = df_response.apply(lambda x: sum(x == 0), axis=1)
count_1 = df_response.apply(lambda x: sum(x == 1), axis=1)
regulized = ((count_1-1) >= count_0)*1

mean_name  = f'model_1_mean_pred_{prediction_name}'
mode_name  = f'model_1_mode_pred_{prediction_name}'
prod_name  = f'model_1_prod_pred_{prediction_name}'

df_response[mean_name] = df_response.mean(axis=1)
df_response[mode_name] = df_response.mode(axis=1)[0]
df_response[prod_name] = df_response.prod(axis=1)
df_response['_id'] = unique_ids

df_response['count_0'] = count_0
df_response['count_1'] = count_1
df_response['regulized'] = regulized

df_response[mean_name] = df_response[mean_name].apply(lambda x: U.sigmoid(x))
# df_response[mode_name] = df_response[mode_name].apply(lambda x: U.sigmoid(x))
df_response[prod_name] = df_response[prod_name].apply(lambda x: U.sigmoid(x))


MASTER_PREDICTIONS = MASTER_PREDICTIONS.merge(df_response, on='_id', how='left')


# For the train, check this 
# df_response = df_response.merge(df[['score','_id']], on='_id', how='left')
# df_response.corr().round(2)['score']
# print(classification_report(df_response['score'], df_response[mode_name]))
# print(classification_report(df_response['score'], df_response[mean_name] > df_response[mean_name].quantile(0.5)))
# print(classification_report(df_response['score'], df_response[prod_name] > df_response[prod_name].quantile(0.5)))


#########################################################################
#########################################################################
### ANCHOR Model 2: Elo Ratings with Nearest Neighbors


df = df_empathy.copy()
df['text'] = df['text'].str.strip()


# Get the embeddings on full text
embeddings = []
for i in tqdm.tqdm(range(len(df))):
    text_i = df_empathy.iloc[i]['text']
    embeds = U.embed_fn(text_i, model='transformer',verbose=False)
    embeddings.append(embeds[0])
df['embeddings'] = embeddings


### Set parameters ----------------------
dev_or_test = 'test' # dev set or test set
# Use feedback without greetings and other flufs
feedback_only = False
## Elo Game N
n_repeat = 600
# Temperature 
temperature = 0.40
# Select Model for the chat | completion
model = "gpt-4-0125-preview"
# Assign a prediction name to hold it on MASTER_PREDICTIONS
prediction_name = f"Elo_{model}_{temperature}_{n_repeat}_{feedback_only}"
### --------------------------------------



## Use only feedback without greeting and other parts
if feedback_only:
    # df_extracted = pd.read_csv("Data/combinedData/empathyFeedbackExtracted.csv")
    # df_extracted = pd.read_csv(path_to_extractive_summary)
    df_extracted = pd.read_csv(path_to_extractive_summary)
    df_extracted.rename(columns={'text':'shorText'}, inplace=True)
    df_extracted = df_extracted[['_id','shorText']]
    df['longText'] = df['text']
    df = df.merge(df_extracted, on='_id', how='left')
    df['text'] = df['shorText'].str.strip()
    # Delete all \n from the text " "
    df['text'] = df['text'].str.replace("\n", " ")



## Unique Ids
if dev_or_test == 'dev':
    unique_ids = df[df.data.isin(['dev'])]['_id'].unique()
else:
    unique_ids = df[df.data.isin(['test'])]['_id'].unique()

sample_ids = df[df.data.isin(['train','dev'])]['_id'].unique()


## Run the model for preferences and get Elo Ratings
results = []


## Run here
for _ in range(n_repeat):
    # print(f"Processing {_+1}/N")
    try:
        # Select a random target
        target_df = df[df._id.isin(unique_ids)].sample(1)
        target_text = target_df['text'].values[0]
        target_score = target_df['score'].values[0]
        target_id = target_df['_id'].values[0]
        
        # Select the sample df
        sample_df = df[df._id.isin(sample_ids)].copy()
        
        # Find cosine similarities
        target_embeds = target_df['embeddings'].values[0]
        example_embeds = np.array(sample_df['embeddings'].tolist())
        
        cosine_similarity = np.dot(target_embeds, example_embeds.T)
        
        sample_df['cosine_similarity'] = cosine_similarity
        # Select most different or similar examples
        sample_df.sort_values(by='cosine_similarity', ascending=True, inplace=True)
        sample_df.sort_values(by='cosine_similarity', ascending=False, inplace=True)
        
        # Sample 3 examples
        example = sample_df.head(10).sample(3)
        
        alternative = example.iloc[0]
        alternative_text  = alternative['text']
        alternative_score = alternative['score']
        alternative_id = alternative['_id']
        
        
        # Prompt ------------------------------------------------
        prompt_in = """<Instruction> Which feedack would you prefer? You will pick the feedback that makes you feel motivated to work harder and understood, rather than being judge. Return either `Text 1` OR `Text 2`</Instruction>
        <Task>
        <Text 1>\n{text1}\n</Text 1>\n
        
        <Text 2>\n{text2}\n</Text 2>\n</Task>"""
        prompt_in = prompt_in.strip()
        prompt_in = prompt_in.format(text1=target_text, text2=alternative_text)
        # print(prompt_in)
        # Prompt ------------------------------------------------
        
        ## Input the chat history
        messages = []
        system_prompt = "Carefully follow the given instructions from the user. You will pick a feedback that makes you feel motivated to work harder."
        
        if model in U.model_openai:
            messages = U.chat_input("system",system_prompt, messages)
            messages = U.chat_input("user",prompt_in, messages)
            LLM_response = U.chat_completion_openAI(messages, model=model,temperature=temperature)
        elif model in U.model_anthropic:
            messages = U.chat_input("user",prompt_in, messages)
            LLM_response = U.chat_completion_anthropic(messages,model=model,temperature=temperature,system=system_prompt)
        elif model in U.model_google:
            messages = U.chat_input("user",system_prompt+"\n"+prompt_in, messages,gemini=True)
            LLM_response = U.chat_completion_google(messages,temperature=temperature,model=model)
        
        LLM_response = LLM_response.strip()
        
        response = re.findall(r'Text 1', LLM_response)
        response = response[0] if len(response) > 0 else "Text 2"
        response = "Text 1" if response == "Text 1" else "Text 2"
        
        print("-"*50)
        print("LLM Response:", LLM_response)
        print("Extracted Response:", response)
        # print("Target:", target_score, "Alternative:", alternative_score)
        
        
        if response == 'Text 1':
            print("Correct" if target_score == 1 else "Incorrect")
            results.append((target_id, alternative_id, target_id))
        if response== 'Text 2':
            print("Correct" if alternative_score == 1 else "Incorrect")
            results.append((target_id, alternative_id, alternative_id))
        if response == 'EQUAL':
            print("Draw", target_score, alternative_score)
            results.append((target_id, alternative_id, 'draw'))
        else:
            pass
        print("-"*50)
    except Exception as e:
        print(e)
        pass


# Calcualte the ELO Ratings
ratings = U.calculate_elo_ratings(results)
ratings = pd.DataFrame(ratings.items(), columns=['_id', prediction_name])
ratings.isna().sum()
# Merge the ratings with the original dataframe
elo_results = df.merge(ratings, on='_id', how='left')
elo_results = elo_results[elo_results._id.isin(unique_ids)]

# Make it sure that all the predictions are filled, if not run the model again
elo_results[elo_results[prediction_name].isna()]


from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
elo_results[prediction_name] = scaler.fit_transform(elo_results[prediction_name].values.reshape(-1,1))
elo_results[prediction_name] = elo_results[prediction_name].apply(lambda x: U.sigmoid(x))


MASTER_PREDICTIONS = MASTER_PREDICTIONS.merge(elo_results[['_id',prediction_name]], on='_id', how='left')




#########################################################################
#########################################################################
### ANCHOR Model 3: Knowledge Distillation : When teachers fail  to teach, students fail to learn.

## !!!!! IMPORTANT !!!!!!!!
###########################
#  We did not use this model
###########################
## !!!!! IMPORTANT !!!!!!!!

df = pd.read_csv("Data/combinedData/empathy_cot.csv")
df['text'] = df['text'].str.strip()
df['cotExample'] = df['cotExample'].str.replace("\n\n", "\n")


### Set parameters ----------------------
dev_or_test = 'test' # dev set or test set
# Use feedback without greetings and other flufs
# Temperature 
temperature = 0.30
# Select Model for the chat | completion
model = 'gemini_1_5'
# Assign a prediction name to hold it on MASTER_PREDICTIONS
prediction_name = f"COT_synthetic_{model}_{temperature}.csv"
### --------------------------------------


# Get the embeddings on full text
embeddings = []
for i in tqdm.tqdm(range(len(df))):
    text_i = df_empathy.iloc[i]['text']
    embeds = U.embed_fn(text_i,verbose=False,model='transformer')
    embeddings.append(embeds[0])
df['embeddings'] = embeddings



## Unique Ids
if dev_or_test == 'dev':
    unique_ids = df[df.data.isin(['dev'])]['_id'].unique()
else:
    unique_ids = df[df.data.isin(['test'])]['_id'].unique()

sample_ids = df[df.data.isin(['train'])]['_id'].unique()


prompt = U.read_write_txt("Prompts/00_COT_analyzer.txt")
# print(prompt)

## Run the model for preferences and get Elo Ratings
results = []

failed_ids = []

feedback_target_dict = {0: "Not-Emphatic", 1: "Empathic"}
    
class1 = feedback_target_dict[0]
class2 = feedback_target_dict[1]    
import time

## Run here
for i, id_i in enumerate(unique_ids):
    print(f"Processing {i+1}/",len(unique_ids),"| ID:", id_i)
    try:
        # Select a random target
        target_df = df[df._id.isin([id_i])]
        target_text = target_df['text'].values[0]
        
        try:
            target_score = target_df['score'].values[0]
            if dev_or_test in ['dev','test']:
                target_score = "test_dev_data"
        except:
            target_score = "test_dev_data"
        
        target_id = target_df['_id'].values[0]
        
        
        # Select the sample df
        sample_df = df[df._id.isin(sample_ids)].copy()
        
        # Find cosine similarities
        target_embeds = target_df['embeddings'].values[0]
        example_embeds = np.array(sample_df['embeddings'].tolist())
        
        cosine_similarity = np.dot(target_embeds, example_embeds.T)
        
        sample_df['cosine_similarity'] = cosine_similarity
        # Select most different or similar examples
        sample_df.sort_values(by='cosine_similarity', ascending=True, inplace=True)
        # sample_df.sort_values(by='cosine_similarity', ascending=False, inplace=True)
        
        # Sample 3 examples
        example = sample_df.head(5).sample(3)
        
        example1_feedback = "<feedback>\n" + example.iloc[0]['text'] + "\n</feedback>\n"
        example2_feedback = "<feedback>\n" + example.iloc[1]['text'] + "\n</feedback>\n"
        example3_feedback = "<feedback>\n" + example.iloc[2]['text'] + "\n</feedback>\n"
        
        example1_result = "<classification>Empathic</classification>" if example.iloc[0]['score']==1 else "<classification>Not-Empathic</classification>"
        example2_result = "<classification>Empathic</classification>" if example.iloc[1]['score']==1 else "<classification>Not-Empathic</classification>"
        example3_result = "<classification>Empathic</classification>" if example.iloc[2]['score']==1 else "<classification>Not-Empathic</classification>"
        
        example1_COT = example.iloc[0]['cotExample'].strip() + "\n" + example1_result
        example2_COT = example.iloc[1]['cotExample'].strip() + "\n" + example2_result
        example3_COT = example.iloc[2]['cotExample'].strip() + "\n" + example3_result
        
        example1 = example1_feedback + "\n" + example1_COT
        example2 = example2_feedback + "\n" + example2_COT
        example3 = example3_feedback + "\n" + example3_COT                
        
        
        
        prompt_in = prompt.format(class1=class1,class2=class2,
                                  target_text=target_text, 
                                  example1=example1, 
                                  example2=example2, 
                                  example3=example3,
                                  )
        prompt_in = prompt_in + "\n\n<target_feedback>\n" + target_text + "\n</target_feedback>\n"
        
        
        
        ## Input the chat history
        
        messages = []
        system_prompt = "Carefully follow the given instructions to analyze feedback."
        
        if model in U.model_openai:
            print("Model : OpenAI")
            messages = U.chat_input("system",system_prompt, messages)
            messages = U.chat_input("user",prompt_in, messages)
            LLM_response = U.chat_completion_openAI(messages, model=model,temperature=temperature)
        elif model in U.model_anthropic:
            print("Model : Anthropic")
            messages = U.chat_input("user",prompt_in, messages)
            LLM_response = U.chat_completion_anthropic(messages,model=model,temperature=temperature,system=system_prompt)
        elif model in U.model_google:
            print("Model : Google")
            messages = U.chat_input("user",system_prompt+"\n"+prompt_in, messages,gemini=True)
            if model == "gemini_1_5":
                LLM_response = U.chat_completion_gemini_1_5(prompt_in)
        
        LLM_response = LLM_response.strip()
        
        try:
            response = re.findall(r'<classification>(.*?)</classification>', LLM_response, re.DOTALL)
            response = [i.strip() for i in response]
            response = response[0]
        except:
            response = ""
        
        print("-"*50)
        print("Actual:", target_score, "Predicted:", response)
        print("-"*50)
        
        results.append((target_id, LLM_response, response))
    except Exception as e:
        print(e)
        failed_ids.append(id_i)


df_res = pd.DataFrame(results, columns=['_id','LLMResponse','response'])
df_res['response'] = df_res['response'].apply(lambda x: 1 if x == 'Empathic' else 0)
df_res['response2'] = df_res['LLMResponse'].apply(lambda x: 0 if len(re.findall(r'Not-Empathic', x))>0 else 1)

df_res = df_empathy.merge(df_res[['response','response2','_id']], on='_id', how='left')



#########################################################################
#########################################################################
### ANCHOR Model 4: Long-Context Learning with Gemini 1.5-pro | Theory of Mind

import time

df = df_empathy.copy()
df['text'] = df['text'].str.strip()


### Set parameters ----------------------
dev_or_test = 'test' # dev set or test set
# Use feedback without greetings and other flufs
# Temperature 
temperature = 0.30
# Select Model for the chat | completion
model = 'gemini_1_5' # Gemini 1.5-pro model
# Assign a prediction name to hold it on MASTER_PREDICTIONS
prediction_name = f"Gemini1_5_pro_{model}_{temperature}.csv"
### --------------------------------------


## Unique Ids
if dev_or_test == 'dev':
    unique_ids = df[df.data.isin(['dev'])]['_id'].unique()
else:
    unique_ids = df[df.data.isin(['test'])]['_id'].unique()

sample_ids = df[df.data.isin(['train'])]['_id'].unique()


prompt = U.read_write_txt("Prompts/00_longContenxtTOM.txt")
# print(prompt)
feedback_target_dict = {0: "Class2", 1: "Class1"}


## Run the model for preferences and get Elo Ratings
results = []
failed_ids = []

## Run here

for i, id_i in enumerate(unique_ids):
    time.sleep(.8) # because of the rate limit of the API
    
    print(f"Processing {i+1}/",len(unique_ids),"| ID:", id_i)
    try:
        # Select a random target
        target_df = df[df._id.isin([id_i])]
        target_text = target_df['text'].values[0]
        
        try:
            target_score = target_df['score'].values[0]
        except:
            target_score = "test_dev_data"
        
        target_id = target_df['_id'].values[0]
        
        
        # Select the sample df
        sample_df = df[df._id.isin(sample_ids)].copy()
        
        sample_df = sample_df.sample(frac=1)
        
        # Generate Examples
        example_in = ""
        for each_row in range(len(sample_df)):
            sample_i = sample_df.iloc[each_row]
            text_i = sample_i['text']
            score_i = sample_i['score']            
            
            example_in += f"# Example {each_row}\n" + text_i + "\n# Classification Result\n" + feedback_target_dict[score_i]
            example_in += "\n\n"
        
        prompt_in = prompt.format(examples=example_in, target_text=target_text)
        # print(prompt_in)
        
        ## Input the chat history
        messages = []
        if model in U.model_openai:
            print("Model : OpenAI")
            messages = U.chat_input("user",prompt_in, messages)
            LLM_response = U.chat_completion_openAI(messages, model=model,temperature=temperature)
        elif model in U.model_anthropic:
            print("Model : Anthropic")
            LLM_response = U.chat_completion_anthropic(messages,model=model,temperature=temperature,system=system_prompt)
        elif model in U.model_google:
            print("Model : Google")
            if model == "gemini_1_5":
                LLM_response = U.chat_completion_gemini_1_5(prompt_in)
            else:
                messages = U.chat_input("user",system_prompt+"\n"+prompt_in, messages,gemini=True)
                LLM_response = U.chat_completion_google(messages,temperature=temperature,model=model)
        
        LLM_response = LLM_response.strip()
        
        try:
            # Response extraction
            extract = """You will extract the prediction results from the content below. Please read carefully the content and return final prediction and class distribution as JSON as {{"prediction", "class_distribution_1","class_distribution_2"}}\n# Content\n{text}"""
            extract_in = extract.format(text=LLM_response)
            response = U.chat_completion_anthropic(U.chat_input("user",extract_in, []))
            # Delete all whitespaces and \n
            response = re.sub(r"\s+", "", response)
            response = json.loads(response)
        except:
            response = ""
        
        print("-"*50)
        print("Actual:", target_score, "Predicted:", response)
        # print(LLM_response)
        print("-"*50)
        
        results.append((target_id, LLM_response, response))
    except Exception as e:
        print(e)
        failed_ids.append(id_i)


df_res = pd.DataFrame(results, columns=['_id','LLMResponse','response'])

extractedPreds = []
for each in df_res['response']:
    try:
        extractedPreds.append(each['prediction'])
    except:
        extractedPreds.append("Class2")

df_res['fin_preds'] = extractedPreds
df_res['fin_preds'] = df_res['fin_preds'].apply(lambda x: 1 if x in feedback_target_dict[1] else 0)

df_res = df_empathy.merge(df_res[['fin_preds','_id']], on='_id', how='right')
# df_res.select_dtypes(include=np.number).corr()


MASTER_PREDICTIONS = MASTER_PREDICTIONS.merge(df_res[['_id','fin_preds']], on='_id', how='left')



#########################################################################
#########################################################################
### ANCHOR Final: Ensemble Predictions for submission


from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()

dev_or_test = 'test'

submit_df = MASTER_PREDICTIONS[MASTER_PREDICTIONS.data.isin([dev_or_test])].copy()
submit_df.columns

# For Mean Predictions, omit the following
# omit1 = submit_df.columns.str.contains('mode')
# omit2 = submit_df.columns.str.contains('mean')
# omit3 = submit_df.columns.str.contains('prod')
# omit4 = submit_df.columns.str.contains('regu')
# omit = omit1 | omit2 | omit3 | omit4
# submit_df = submit_df.loc[:,~omit]

all_preds = submit_df.loc[:,"predictions":].columns.tolist()
all_preds = [i for i in all_preds if 'score' not in i]
submit_df[all_preds] = scaler.fit_transform(submit_df[all_preds])

# all_preds = [i for i in all_preds if 'mean' in i]
# Get the means and elo and fin_preds
all_preds = [i for i in all_preds if ('mean' in i) or ('Elo' in i) or ('fin_preds' in i)]
all_preds.remove('predictions')

submit_df['total'] = submit_df[all_preds].mean(axis=1)
submit_df['total'] = scaler.fit_transform(submit_df['total'].values.reshape(-1,1))
submit_df['total'] = U.sigmoid(submit_df['total'])

# Cut scores
cut_scores = .60 # Set to .45 to .60
submit_df['output'] = (submit_df['total'] > cut_scores)*1
os.makedirs('Output', exist_ok=True)
submit_df[['_id','output']].to_csv(r'Output/00_submit_empathy.csv', index=False)