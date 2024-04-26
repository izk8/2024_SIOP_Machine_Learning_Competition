from openai import OpenAI
import pandas as pd
import json
import os


# Main class for the Interview prediction generation.
class InterviewPredictor:
    def __init__(self, model_params, message_params, secret_key, file_paths):
        self.client = self._initialize_client(secret_key)
        self.model_params = model_params
        self.personality_dict = self._load_interpretations(
            file_paths['interpretation_fp'])
        self.system_message = message_params['system_message']
        self.task_explanation = message_params['task_explanation']
        self.predictions_path = file_paths['results_fp']
        self.test_data = self._merge_datasets(file_paths)

    # Initialize the OpenAI client using the input key.
    # Inputs
    #   secret_key: str
    # Outputs:
    #   client: OpenAI
    def _initialize_client(self, secret_key):
        client = OpenAI(api_key=secret_key)

        return client

    # Loads personality interpretations.
    # Inputs
    #   interpretation_fp: str
    # Outputs
    #   personality_dict: dict
    def _load_interpretations(self, interpretation_fp):
        with open(interpretation_fp, 'r') as f:
            personality_dict = json.load(f)
            f.close()

        return personality_dict

    # Format question and answers to use as examples for the API call.
    # Inputs
    #   text: str
    # Outputs
    #   merged: str
    def _format_questions_answers(self, text):
        # format questions
        temp = text.split('Question: ')[1:-1]
        questions = [t.split('\r\nResponse: ')[0] for t in temp]
        formatted_questions = [f'###Question: {q}\n' for q in questions]
        last_question = text.split('Question: ')[-1]
        formatted_questions.append(f'###Question: {last_question}\n')

        # format answers
        temp = text.split('Question: ')[1:-1]
        answers = [t.split('\r\nResponse: ')[1] for t in temp]
        formatted_answers = [f'Response: {a}###\n' for a in answers]
        last_answer = f'Response: ###'
        formatted_answers.append(last_answer)

        # merge together
        merged = ''.join(
            [f'{q}{a}' for q, a in zip(formatted_questions, formatted_answers)])
        return merged

    # Return the bucket version of the final score.
    # Inputs
    #   score: int
    # Outputs
    #   str
    def _classify_personality_bucket(self, score):
        if score < 3:
            return 'low'
        elif score > 5:
            return 'high'
        else:
            return 'moderate'

    # Process the interview data.
    # Inputs
    #   interview_fp: str
    # Outputs
    #   interview: pandas.DataFrame
    def _process_interview(self, interview_fp):
        interview = pd.read_csv(interview_fp)
        interview['reference'] = interview['questions_answers'].apply(
            lambda x: self._format_questions_answers(x))

        return interview

    # Process the personality data.
    # Inputs
    #   personality_fp: str
    # Outputs
    #   personality: pandas.DataFrame
    def _process_personality(self, personality_fp):
        personality = pd.read_csv(personality_fp)
        personality['extraversion_bucket'] = personality['extraversion'].apply(
            lambda x: self._classify_personality_bucket(x))
        personality['agreeableness_bucket'] = personality[
            'agreeableness'].apply(
            lambda x: self._classify_personality_bucket(x))
        personality['conscientiousness_bucket'] = personality[
            'conscientiousness'].apply(
            lambda x: self._classify_personality_bucket(x))
        personality['emotional_stability_bucket'] = personality[
            'emotional_stability'].apply(
            lambda x: self._classify_personality_bucket(x))
        personality['openness_bucket'] = personality['openness'].apply(
            lambda x: self._classify_personality_bucket(x))

        return personality

    # Load, format, and merge the interview and personality datasets.
    # Inputs
    #   file_paths: dict
    # Outputs
    #   merged_df: pandas.DataFrame
    def _merge_datasets(self, file_paths):
        interview = self._process_interview(file_paths['interview_fp'])
        personality = self._process_personality(file_paths['personality_fp'])
        merged_df = interview.merge(personality, on='_id', how='left')

        return merged_df

    # Generates a list of messages for use in the OpenAI API call.
    # Inputs
    #   df: pandas.DataFrame
    # Outputs
    #   messages: list[dict]
    def _generate_messages(self, df):
        messages = []
        messages.append({"role": "system", "content": self.system_message})

        # generate scores plus buckets
        extraversion = f"{df['extraversion'].iloc[0]} out of 7, which is {df['extraversion_bucket'].iloc[0]}"
        agreeableness = f"{df['agreeableness'].iloc[0]} out of 7, which is {df['agreeableness_bucket'].iloc[0]}"
        conscientiousness = f"{df['conscientiousness'].iloc[0]} out of 7, which is {df['conscientiousness_bucket'].iloc[0]}"
        emotional_stability = f"{df['emotional_stability'].iloc[0]} out of 7, which is {df['emotional_stability_bucket'].iloc[0]}"
        openness = f"{df['openness'].iloc[0]} out of 7, which is {df['openness_bucket'].iloc[0]}"

        # generate score and bucket explanations
        extraversion_explanation = f"This means that {self.personality_dict['extraversion'][df['extraversion_bucket'].iloc[0]]}"
        agreeableness_explanation = f"This means that {self.personality_dict['agreeableness'][df['agreeableness_bucket'].iloc[0]]}"
        conscientiousness_explanation = f"This means that {self.personality_dict['conscientiousness'][df['conscientiousness_bucket'].iloc[0]]}"
        emotional_stability_explanation = f"This means that {self.personality_dict['emotional_stability'][df['emotional_stability_bucket'].iloc[0]]}"
        openness_explanation = f"This means that {self.personality_dict['openness'][df['openness_bucket'].iloc[0]]}"

        # add scores, buckets, and explanations to prompt
        prompt = 'You were tested on your personality using the Five Factor model. Your results are as follows:\n'
        prompt += f'- You scored {extraversion} in Extraversion. {extraversion_explanation}\n'
        prompt += f'- You scored {agreeableness} in Agreeableness. {agreeableness_explanation}\n'
        prompt += f'- You scored {conscientiousness} in Conscientiousness. {conscientiousness_explanation}\n'
        prompt += f'- You scored {emotional_stability} in Emotional Stability. {emotional_stability_explanation}\n'
        prompt += f'- You scored {openness} in Openness. {openness_explanation}\n'
        prompt += f'{self.task_explanation}\n'

        prompt += df['reference'].iloc[0]

        messages.append({"role": "user", "content": prompt})

        return messages

    # Parse and format the information from ChatGPT response.
    # Inputs
    #   responses_df: pandas.DataFrame
    # Outputs
    #   clean_df: pandas.DataFrame
    def _parse_responses(self, responses_df):
        responses_df = responses_df.rename(columns={'index': '_id'})
        responses_df['benchmark'] = 'interview'
        responses_df['output'] = responses_df['response'].apply(
            lambda x: x.choices[0].message.content)
        clean_df = responses_df[
            ['_id', 'benchmark', 'output']].reset_index(drop=True)

        return clean_df

    # Generate the GPT model responses using the OpenAI APIs and clean the
    # response.
    # Inputs
    # Outputs
    #   clean_df: pandas.DataFrame
    def _generate_responses(self):
        # dictionary for collecting responses
        responses = {}
        for i, _id in enumerate(self.test_data['_id'].unique()):
            current_data = self.test_data[self.test_data['_id'] == _id]
            gpt_message = self._generate_messages(current_data)
            response = self.client.chat.completions.create(
                **self.model_params,
                messages=gpt_message
            )

            # record responses in dictionary
            responses[_id] = {'response': response}

        # create df from dictionary
        responses_df = pd.DataFrame.from_dict(
            responses, orient='index').reset_index()
        clean_df = self._parse_responses(responses_df)

        return clean_df

    # Save the input prediction dataframe.
    # Inputs
    #   prediction_df: pandas.DataFrame
    # Outputs
    def _save_prediction(self, prediction_df):
        if not os.path.exists(self.predictions_path):
            os.makedirs(self.predictions_path)
        prediction_df.to_csv('{}/interview.csv'.format(self.predictions_path),
                             index=False)

    # Create and save the predictions for the input data.
    # Inputs
    # Outputs
    def predict_and_save(self):
        print('Predicting Interview')
        prediction = self._generate_responses()
        self._save_prediction(prediction)
        print('- End of interview prediction')
