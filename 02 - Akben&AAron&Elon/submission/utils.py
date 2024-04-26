# from bert_score import score as bert_score
# from nltk.translate.bleu_score import sentence_bleu
# from rouge import Rouge
# from sentence_transformers import SentenceTransformer, util as st_util
# import numpy as np
# import math
# from nltk import word_tokenize
# from sklearn.metrics.pairwise import cosine_similarity
# from sklearn.feature_extraction.text import CountVectorizer
# from transformers import GPT2LMHeadModel, GPT2Tokenizer
# import torch

# LLM Providers
import sklearn
import math
from sentence_transformers import SentenceTransformer
from openai import OpenAI
from mistralai.models.chat_completion import ChatMessage
from mistralai.client import MistralClient
import openai
import mistralai
import anthropic
import google.generativeai as genai
import cohere

# Other Libraries
import pathlib
import re
import json
import subprocess
import os
import sys
import numpy as np
import pandas as pd
import json

# Embedding Models
import numpy as np
import torch
device = "cuda" if torch.cuda.is_available() else "cpu"


#########################################################################
#########################################################################
### ANCHOR API Keys


# API Keys
genai.configure(api_key=os.getenv("GOOGLE_GENAI_API_KEY"))
Oclient = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
Mclient = MistralClient(api_key=os.getenv("MISTRAL_API_KEY"))
Aclient = anthropic.Client(api_key=os.getenv("ANTHROPIC_API_KEY"))
Cclient = co = cohere.Client(api_key=os.getenv("COHERE_API_KEY"))

# For vertex AI, please change the project ID to your own project ID 
# pip install --upgrade google-cloud-aiplatform

# Run the following to authenticate your account
# gcloud auth application-default login

VERTEX_AI_GOOGLE_PROJECT_ID = "geminipro-415023" # Gemini 1.5-Pro
                          

#########################################################################
#########################################################################
# ANCHOR Paths

data_path = os.path.join(os.getcwd(), 'Data')
data_path = pathlib.Path(data_path)


#########################################################################
#########################################################################
# ANCHOR Load the Data

# Load the Empathy Data
path = os.path.join(data_path, 'train', 'empathy_train.csv')
df_empathy_train = pd.read_csv(path)
df_empathy_train['data'] = 'train'


path = os.path.join(data_path, 'dev', 'empathy_val_public.csv')
df_empathy_dev = pd.read_csv(path)
df_empathy_dev['data'] = 'dev'

path = os.path.join(data_path, 'test', 'empathy_test_public.csv')
df_empathy_test = pd.read_csv(path)
df_empathy_test['data'] = 'test'


df_empathy = pd.concat([df_empathy_train, df_empathy_dev,df_empathy_test], axis=0, ignore_index=True)
df_empathy['benchmark'] = "empathy"


# Load the Interview Data
path = os.path.join(data_path, 'train', 'interview_train.csv')
df_interview_train = pd.read_csv(path)
df_interview_train['data'] = 'train'

path = os.path.join(data_path, 'dev', 'interview_val_public.csv')
df_interview_dev = pd.read_csv(path)
df_interview_dev['data'] = 'dev'

path = os.path.join(data_path, 'test', 'interview_test_public.csv')
df_interview_test = pd.read_csv(path)
df_interview_test['data'] = 'test'

df_interview = pd.concat([df_interview_train, df_interview_dev, df_interview_test], axis=0, ignore_index=True)
df_interview['benchmark'] = "interview"


# Load the Clarity Data
path = os.path.join(data_path, 'train', 'clarity_train.csv')
df_clarity_train = pd.read_csv(path)
df_clarity_train['data'] = 'train'

path = os.path.join(data_path, 'dev', 'clarity_val_public.csv')
df_clarity_dev = pd.read_csv(path)
df_clarity_dev['data'] = 'dev'

path = os.path.join(data_path, 'test', 'clarity_test_public.csv')
df_clarity_test = pd.read_csv(path)
df_clarity_test['data'] = 'test'

df_clarity = pd.concat([df_clarity_train, df_clarity_dev,
                       df_clarity_test], axis=0, ignore_index=True)
df_clarity['benchmark'] = "clarity"


# Load the Fairness Data
path = os.path.join(data_path, 'train', 'fairness_train.csv')
df_fairness_train = pd.read_csv(path)
df_fairness_train['data'] = 'train'

path = os.path.join(data_path, 'dev', 'fairness_val_public.csv')
df_fairness_dev = pd.read_csv(path)
df_fairness_dev['data'] = 'dev'

path = os.path.join(data_path, 'test', 'fairness_test_public.csv')
df_fairness_test = pd.read_csv(path)
df_fairness_test['data'] = 'test'


df_fairness = pd.concat([df_fairness_train, df_fairness_dev,
                        df_fairness_test], axis=0, ignore_index=True)
df_fairness['benchmark'] = "fairness"


def loadData():
    """
    Load the data and return the dataframes for empathy, interview, clarity, and fairness.

    Returns:
        df_empathy (pandas.DataFrame): DataFrame containing empathy data.
        df_interview (pandas.DataFrame): DataFrame containing interview data.
        df_clarity (pandas.DataFrame): DataFrame containing clarity data.
        df_fairness (pandas.DataFrame): DataFrame containing fairness data.
    """
    return df_empathy, df_interview, df_clarity, df_fairness


#########################################################################
#########################################################################
# ANCHOR Eval AI Submit
# pip install "evalai"
# evalai challenge 2207 phases || Test phase > 4392


class EvalAI:
    def __init__(self, json_path=True):
        self.json_path = json_path

    def get_team(self):
        # Get the team
        get_team_command = "evalai teams --participant"
        subprocess.run(get_team_command, shell=True)

    def set_token(self, token_path):
        # Set the token to class variable
        self.token_path = token_path
        # Open the token file and read the token
        with open(token_path, 'r') as f:
            token = f.read().strip()
            if self.json_path:
                token = json.loads(token)["token"]
        set_token_command = f"evalai set_token {token}"
        subprocess.run(set_token_command, shell=True)

    def get_challenge_phases(self):
        # Get the challenge phases
        get_phases_command = "evalai challenge 2207 phases"
        subprocess.run(get_phases_command, shell=True)

    def get_all_submissions(self, test=False, parse=True):
        dev_or_test = "4391"
        if test:
            dev_or_test = "4392"
        # Get all the submissions
        get_submissions_command = f"evalai challenge 2207 phase {dev_or_test} submissions"
        subprocess.run(get_submissions_command, shell=True)

        if parse:
            result = subprocess.run(
                get_submissions_command, shell=True, capture_output=True, text=True)
            output = result.stdout
            # Parse the output into a list of dictionaries
            lines = output.split('\n')
            headers = [h.strip() for h in lines[1].split('|')[1:-1]]
            data = []
            for line in lines[3:-2]:
                values = [v.strip() for v in line.split('|')[1:-1]]
                data.append(dict(zip(headers, values)))

            # Convert the list of dictionaries into a DataFrame
            df = pd.DataFrame(data)
            df = df[~df.ID.isna()]
            df = df[df.ID != ""]
            return df

    def get_submission(self, submission_id, test=False):
        dev_or_test = "4391"
        if test:
            dev_or_test = "4392"
        # Get the submission
        get_submission_command = f"evalai submission {submission_id} result"
        # Get this results to pyton
        result = subprocess.run(get_submission_command,
                                shell=True, capture_output=True)
        result = json.loads(result.stdout)
        return result

    def submit(self, csv_file_path, test=False):
        dev_or_test = "4391"
        if test:
            dev_or_test = "4392"

        # self.set_token(self.token_path)

        # Set the submission
        submit_command = f"evalai challenge 2207 phase {dev_or_test} submit --file {csv_file_path} --large --private"
        # Say N to the prompt and capture the output
        output = subprocess.run(
            submit_command, shell=True, input="N\n", capture_output=True, text=True)
        # Whether there is an error or not
        message_out = output.stdout
        message_out = message_out.lower()
        if "error" in output.stdout:
            print("\n!!! Submission Failed !!!\n")
            print(output.stdout)
            return False
        else:
            print("\n!!! Submission Complete !!!\n")
            print(output.stdout)
            submission_id = re.search(r"id (\d+)", output.stdout).group(1)
            print(submission_id)
            return submission_id

#########################################################################
#########################################################################
# ANCHOR LLM Functions and Other Functions



# Screen Cleaning
def clc():
    os.system('cls' if os.name == 'nt' else 'clear')


# Read and Write to a txt file
def read_write_txt(file_path, text_to_write=None, read=True):
    if read:
        with open(file_path, 'r', encoding='utf-8') as file:
            text = file.read()
        return text
    else:
        with open(file_path, 'w', encoding='utf-8') as file:
            file.write(text_to_write)


# ChatInput
def chat_input(role, text, message_history, mistral=False, gemini=False):
    if mistral:
        message_history.append(ChatMessage(role=role, content=text))
        return message_history
    elif gemini:
        message_history.append({"role": role, "parts": [text]})
        return message_history
    else:
        message_history.append({"role": role, "content": text})
        return message_history


def calculate_elo_ratings(matches, initial_rating=1500, k_factor=32):
    # Initialize player ratings
    ratings = {
        player: initial_rating for match in matches for player in match[:2]}

    def expected_score(ra, rb):
        return 1 / (1 + 10 ** ((rb - ra) / 400))

    for match in matches:
        player1, player2, result = match
        ra, rb = ratings[player1], ratings[player2]

        # Calculate expected scores
        ea = expected_score(ra, rb)
        eb = expected_score(rb, ra)

        # Determine actual outcomes
        if result == 'draw':
            sa, sb = 0.5, 0.5
        else:
            sa = 1 if result == player1 else 0
            sb = 1 if result == player2 else 0

        # Update ratings
        ratings[player1] += k_factor * (sa - ea)
        ratings[player2] += k_factor * (sb - eb)

    return ratings


def log2probs(log_probs):
    return math.exp(log_probs)


def sigmoid(x,sum=False):
    return 1 / (1 + np.exp(-x))


#########################################################################
#########################################################################
# ANCHOR LLM Functions

# Models list
try:
    model_openai = [model.id for model in Oclient.models.list().data]
except:
    pass

model_text_completion = ["davinci-002", "babbage-002","gpt-3.5-turbo-instruct", "gpt-3.5-turbo-instruct-0914"]

try:
    model_mistral = [model.id for model in Mclient.list_models().data]
except:
    pass

model_anthropic = ["claude-3-opus-20240229", "claude-3-sonnet-20240229","claude-3-haiku-20240307", "claude-2.1", "claude-2.0", "claude-instant-1.2"]

model_google = ["palm", "gemini1", "gemini2","gemini3", "gemini4", "gemini_1_5"]
try:
    for m in genai.list_models():
        if 'generateContent' in m.supported_generation_methods:
            model_google.extend([m.name])
except:
    pass



#########################################################################
#########################################################################
### ANCHOR Gemini 1.5 Pro


import base64
import vertexai
from vertexai.generative_models import GenerativeModel, Part, FinishReason,Content
import vertexai.preview.generative_models as generative_models


safety_settings = {
    generative_models.HarmCategory.HARM_CATEGORY_HATE_SPEECH: generative_models.HarmBlockThreshold.BLOCK_NONE,
    generative_models.HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: generative_models.HarmBlockThreshold.BLOCK_NONE,
    generative_models.HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: generative_models.HarmBlockThreshold.BLOCK_NONE,
    generative_models.HarmCategory.HARM_CATEGORY_HARASSMENT: generative_models.HarmBlockThreshold.BLOCK_NONE,
}


def chat_input_gemini(role, text, messages):
    if role == "assistant":
        role = 'model'
    messages.append(Content(role=role, parts=[Part.from_text(text)]))
    return messages 


def chat_completion_gemini_1_5(messages, temperature=0.4,max_output_tokens=2048, top_p=0.4, top_k=32,safety_settings=safety_settings):
    vertexai.init(project=VERTEX_AI_GOOGLE_PROJECT_ID, location="us-central1")
    model = GenerativeModel("gemini-experimental")
    
    generation_config = {
        "max_output_tokens": max_output_tokens,
        "temperature": temperature,
        "top_p": top_p,
        "top_k": top_k,
        }

    responses = model.generate_content(
        messages,
        generation_config=generation_config,
        safety_settings=safety_settings,
        stream=False,
        )
    return responses.text



#########################################################################
#########################################################################
# ANCHOR Embedding Functions


sent_transformer = SentenceTransformer(
    'sentence-transformers/all-MiniLM-L6-v2', device=device)


def embed_fn(text,  model="text-embedding-3-large",verbose=True):
    if "transformer" in model:
        if verbose:
            print("Using the Transformer Model")
        return sent_transformer.encode([text])
    else:
        if verbose:
            print("Using the OpenAI Model")
        embeds = Oclient.embeddings.create(input=text, model=model)
        return np.array(embeds.data[0].embedding).reshape(1, -1)


def reranker(query, docs, top_n=3, return_documents=False):
    results = co.rerank(model="rerank-english-v2.0",
                        documents=docs,
                        query=query,
                        return_documents=return_documents,
                        top_n=top_n)
    relevance_scores = [i.relevance_score for i in results.results]
    idx = [i.index for i in results.results]
    return [idx, relevance_scores]



#########################################################################
#########################################################################
# ANCHOR Chat Functions


def chat_completion_openAI(messages,
                           model="gpt-3.5-turbo-0125",
                           temperature=0.8,
                           return_text=True,
                           **kwargs):
    response = Oclient.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,
        **kwargs
    )
    if return_text:
        return response.choices[0].message.content
    else:
        return response


def chat_completion_mistral(messages,
                            model="mistral-tiny",
                            temperature=0.8,
                            return_text=True,
                            **kwargs
                            ):

    response = Mclient.chat(
        messages=messages,
        model=model,
        temperature=temperature,
        **kwargs
    )
    if return_text:
        return response.choices[0].message.content
    else:
        return response


def chat_completion_anthropic(
    messages,
    model="claude-3-haiku-20240307",
    temperature=0.8,
    return_text=True,
    max_tokens=2024,
    system="",
    **kwargs,
):
    response = Aclient.messages.create(
        system=system,
        model=model,
        temperature=temperature,
        max_tokens=max_tokens,
        messages=messages,
        **kwargs
    )
    if return_text:
        return response.content[0].text
    else:
        return response


def chat_completion_google(
    messages,
    model="palm",
    temperature=0.80,
    return_text=True,
    max_tokens=1000,
    top_k=40,
    top_p=0.95,
    candidate_count=1,
    system_message=None,
    examples=None
):

    if 'palm' in model:
        # Adjust model name based on the input
        if model == "palm":
            model = "models/chat-bison-001"

        # Default parameters for the request
        defaults = {
            'model': model,
            'temperature': temperature,
            'candidate_count': candidate_count,
            'top_k': top_k,
            'top_p': top_p,
        }

        # Append a marker for the next request in the messages list
        # Generate the response using the genai chat function
        # Assuming genai.chat is a callable function available in this context
        response = genai.chat(
            **defaults,
            context=system_message,
            examples=examples,
            messages=messages
        )

        if return_text:
            return response.last
        else:
            return response

    if "gemini" in model:
        safety_settings = [{"category": "HARM_CATEGORY_HARASSMENT", "threshold": "block_none"}, {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "block_none"}, {
            "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "block_none"}, {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "block_none"}, ]

        switcher = {
            "gemini1": "models/gemini-1.0-pro",
            "gemini2": "models/gemini-1.0-pro-001",
            "gemini3": "models/gemini-1.0-pro-latest",
            }
        model = switcher[model]

        # Process Messages for the models
        # Get the last user_input from message_history
        last_user_input = [message["parts"][0] for message in messages if message["role"] == "user"][-1]
        # Find the last user input idx
        where = [i for i, message in enumerate(messages) if message["role"] == "user"][-1]
        # Pop the last user input
        messages.pop(where)

        # Set the generation config
        generation_config = {
            "temperature": temperature,
            "top_p": top_p,
            "top_k": top_k,
            "max_output_tokens": max_tokens}

        model = genai.GenerativeModel(model_name=model, generation_config=generation_config, safety_settings=safety_settings)
        convo = model.start_chat(history=messages)
        convo.send_message(last_user_input)

        # Add back to the message to messages
        chat_input("user", last_user_input, messages, gemini=True)
        if return_text:
            return convo.last.text
        else:
            return convo



#########################################################################
#########################################################################
# ANCHOR Text Completion Functions


def text_completion_google(prompt,
                           model='palm',
                           temperature=0.7,
                           candidate_count=1,
                           top_k=40,
                           top_p=0.95,
                           max_output_tokens=1000,
                           stop=[],
                           verbose=True

                           ):

    if "palm" in model:
        if verbose:
            print("Using the PALM Model")
        model = 'models/text-bison-001'
        settings = {
            'model': str(model),
            'temperature': temperature,
            'candidate_count': int(candidate_count),
            'top_k': int(top_k),
            'top_p': top_p,
            'max_output_tokens': int(max_output_tokens),
            'stop_sequences': stop,
            'safety_settings': [{"category": "HARM_CATEGORY_DEROGATORY", "threshold": "block_none"}, {"category": "HARM_CATEGORY_TOXICITY", "threshold": "block_none"}, {"category": "HARM_CATEGORY_VIOLENCE", "threshold": "block_none"}, {"category": "HARM_CATEGORY_SEXUAL", "threshold": "block_none"}, {"category": "HARM_CATEGORY_MEDICAL", "threshold": "block_none"}, {"category": "HARM_CATEGORY_DANGEROUS", "threshold": "block_none"}],
        }
        response = genai.generate_text(**settings, prompt=prompt)
        if candidate_count == 1:
            return response.result
        else:
            return [r['output'] for r in response.candidates]

    if "gemini" in model:
        if verbose:
            print("Using the Gemini Model")
        # safety_settings = [ { "category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_ONLY_HIGH" }, { "category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_ONLY_HIGH" }, { "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_ONLY_HIGH" }, { "category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_ONLY_HIGH" }, ]
        safety_settings = [{"category": "HARM_CATEGORY_HARASSMENT", "threshold": "block_none"}, {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "block_none"}, {
            "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "block_none"}, {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "block_none"}, ]

        switcher = {
            "gemini1": "gemini-1.0-pro",
            "gemini2": "gemini-1.0-pro-001",
            "gemini3": "models/gemini-1.0-pro-latest",

        }

        model = switcher[model]

        generation_config = {
            "temperature": temperature,
            "top_p": top_p,
            "top_k": top_k,
            "max_output_tokens": max_output_tokens,
            "candidate_count": candidate_count,
            "stop_sequences": stop,
        }

        model = genai.GenerativeModel(
            model_name=model, generation_config=generation_config, safety_settings=safety_settings)
        prompt_parts = [prompt]

        response = model.generate_content(prompt_parts)
        return response.text

    if (("palm" in model) or ("gemini" in model)) == False:
        print("Model not found")
        return None


def text_completion_gpt3(
    prompt,
    model="babbage-002",
    temperature=0.1,
    stop=None,
    max_tokens=100,
    logprobs=None,
    return_text=True,
    **kwargs,
):
    
    response = Oclient.completions.create(
        model=model,
        prompt=prompt,
        temperature=temperature,
        stop=stop,
        logprobs=logprobs,
        max_tokens=max_tokens,
        **kwargs
    )
    if return_text:
        return response.choices[0].text
    else:
        return response


def log_probs_gpt3(response_in):

    if isinstance(response_in, openai.types.completion.Completion):
        top_logs = response_in.choices[0].logprobs.top_logprobs

    if isinstance(response_in, list):
        top_logs = response_in

    probs = []
    for i, log_prob_out in enumerate(top_logs):
        temp_i = pd.DataFrame([log_prob_out]).T
        temp_i.reset_index(inplace=True)
        temp_i.columns = ["token", "log_prob"]
        temp_i['prob'] = temp_i['log_prob'].apply(lambda x: log2probs(x))
        temp_i['order'] = i
        probs.append(temp_i)

    probs = pd.concat(probs, ignore_index=True)
    return probs


def text_completion_chatGpt(messages,
                            model="gpt-3.5-turbo-0125",
                            temperature=0.8,
                            max_tokens=None,
                            logprobs=False,
                            top_logprobs=None,
                            stop=[],
                            return_text=True,
                            **kwargs
                            ):

    completion = Oclient.chat.completions.create(
        messages=messages,
        model=model,
        temperature=temperature,
        logprobs=logprobs,
        top_logprobs=top_logprobs,
        max_tokens=max_tokens,
        stop=stop,
        **kwargs
    )
    if return_text:
        return completion.choices[0].message.content
    else:
        return completion


def log_probs_chatgpt(response):
    if isinstance(response, openai.types.chat.chat_completion.ChatCompletion):
        top_logs = response.choices[0].logprobs
    else:
        raise ValueError(
            "Response is not a valid response. Use ChatGPT completion directly.")

    results = []
    for each_log in top_logs:
        for i, each_idx in enumerate(each_log[1]):
            print(i)
            for each in each_idx.top_logprobs:
                each = each.json()
                # Read as json
                each = json.loads(each)
                # Omits bytes
                each.pop('bytes', None)
                each['order'] = i
                results.append(pd.DataFrame(each, index=[0]))

    results = pd.concat(results, ignore_index=True)
    results['probs'] = results['logprob'].apply(lambda x: log2probs(x))
    results.probs = results.probs.round(4)
    return results


#########################################################################
#########################################################################
### ANCHOR Interview Stats

# model = SentenceTransformer('all-MiniLM-L6-v2')

# # Function to calculate cosine similarity
# def calculate_cosine_similarity(text1, text2):
#     embeddings1 = model.encode(text1)
#     embeddings2 = model.encode(text2)
#     cosine_sim = cosine_similarity([embeddings1], [embeddings2])
#     return cosine_sim[0][0]

# # Function to calculate BERTScore
# def calculate_bert_score(candidate, reference):
#     P, R, F1 = bert_score([candidate], [reference], lang="en")
#     return np.mean(F1.numpy())

# # Function for BLEU Score
# def calculate_bleu_score(reference, candidate):
#     reference_tokens = word_tokenize(reference)
#     candidate_tokens = word_tokenize(candidate)
#     score = sentence_bleu([reference_tokens], candidate_tokens)
#     return score

# # Function for ROUGE Score
# def calculate_rouge_scores(reference, candidate):
#     rouge = Rouge()
#     scores = rouge.get_scores(candidate, reference)[0]
#     return scores


# def calculate_perplexity(text, model_name='gpt2'):
#     """
#     Calculate the perplexity of a text using GPT-2.
    
#     :param text: The input text for which to calculate perplexity.
#     :param model_name: The model variant of GPT-2. Default is 'gpt2'.
#     :return: The perplexity score.
#     """
#     # Load pre-trained model tokenizer (vocabulary) and model
#     tokenizer = GPT2Tokenizer.from_pretrained(model_name)
#     model = GPT2LMHeadModel.from_pretrained(model_name)
#     model.eval()

#     # Encode text inputs
#     encodings = tokenizer(text, return_tensors='pt')

#     # Calculate log likelihood and keep track of total token count
#     with torch.no_grad():  # No need to calculate gradients
#         outputs = model(**encodings, labels=encodings["input_ids"])
#         log_likelihood = -outputs.loss * encodings["input_ids"].shape[1]

#     # Calculate perplexity
#     perplexity = torch.exp(log_likelihood / encodings["input_ids"].shape[1]).item()

#     return perplexity


# def generation_metrics(reference, candidate):
#     """
#     Calculate and return the BERTScore, BLEU Score, ROUGE-1, ROUGE-2, ROUGE-L, Cosine Similarity, and Perplexity of the candidate text.

#     Parameters:
#     reference (str): The reference text.
#     candidate (str): The candidate text.

#     Returns:
#     tuple: A tuple containing the BERTScore, BLEU Score, ROUGE-1, ROUGE-2, ROUGE-L, Cosine Similarity, and Perplexity.

#     """
#     bert_score = calculate_bert_score(candidate, reference)
#     bleu_score = calculate_bleu_score(reference, candidate)
#     rouge_scores = calculate_rouge_scores(reference, candidate)
#     rouge1 = rouge_scores['rouge-1']['f']
#     rouge2 = rouge_scores['rouge-2']['f']
#     rougeL = rouge_scores['rouge-l']['f']
#     cosine_similarity = calculate_cosine_similarity(candidate, reference)
#     perplexity = calculate_perplexity(reference+candidate)
#     return bert_score, bleu_score, rouge1, rouge2, rougeL, cosine_similarity, perplexity

    