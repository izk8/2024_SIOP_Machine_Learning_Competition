from openai import OpenAI
import pandas as pd
import os


# Main class for the Farirness prediction generation.
class FairnessPredictor:
    def __init__(self, model_params, message_params, secret_key, file_paths):
        self.client = self._initialize_client(secret_key)
        self.model_params = model_params
        self.system_message = message_params['system_message']
        self.task_explanation = message_params['task_explanation']
        self.predictions_path = file_paths['predictions_path']
        self.test_df, self.train_df = self._load_data(file_paths)

    # Initialize the OpenAI client using the input key.
    # Inputs
    #   secret_key: str
    # Outputs:
    #   client: OpenAI
    def _initialize_client(self, secret_key):
        client = OpenAI(api_key=secret_key)

        return client

    # Merge the two text from the input data.
    # Inputs
    #   data: pandas.DataFrame
    # Outputs
    #   data: pandas.DataFrame
    def _merge_options(self, data):
        data['text'] = data[['first_option', 'second_option']].apply(
            lambda x: f'first = {x[0]}\nsecond = {x[1]}', axis=1)

        return data

    # Loads and prepares train and test datasets
    # for generating input messages
    def _load_data(self, file_paths):
        test_df = pd.read_csv(file_paths['test_data_fp'])
        train_df = pd.read_csv(file_paths['train_data_fp'])
        train_df = train_df.rename(columns={'majority_vote': 'res'})
        train_df = self._merge_options(train_df)
        test_df = self._merge_options(test_df)

        return test_df, train_df

    # Generates a list of examples from train dataset in a format that is
    # ready to use in an API call.
    # Inputs
    # Outputs
    #   examples: list[dict]
    def _generate_examples(self):
        examples = []
        for text, res in zip(self.train_df['text'], self.train_df['res']):
            examples.append({"role": "user", "content": text})
            examples.append({"role": "assistant", "content": str(res)})

        return examples

    # Generates a list of messages to use for the API call.
    # Inputs
    #   curren_data: pandas.DataFrame
    # Outputs
    #   messages: list[dict]
    def _generate_messages(self, current_data):
        messages = []
        messages.append({"role": "system", "content": self.system_message})
        messages.append({"role": "user", "content": self.task_explanation})
        examples = self._generate_examples()
        messages.extend(examples)
        messages.append({"role": "user", "content": current_data['text'].iloc[0]})

        return messages

    # Parse and format the information from ChatGPT response
    # and applies post-processing cleaning.
    # Inputs
    #   responses_df: pandas.DataFrame
    # Outputs
    #   clean_df: pandas.DataFrame
    def _parse_responses(self, responses_df):
        responses_df = responses_df.rename(columns={'index': '_id'})
        responses_df['benchmark'] = 'fairness'
        responses_df['output'] = responses_df['response'].apply(
            lambda x: x.choices[0].message.content
            if (
                    x.choices[0].message.content == 'first' or
                    x.choices[0].message.content == 'second')
            else 'second'
        )
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
        for i, _id in enumerate(self.test_df['_id'].unique()):
            current_data = self.test_df[self.test_df['_id'] == _id]
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
        prediction_df.to_csv('{}/fairness.csv'.format(self.predictions_path),
                             index=False)

    # Create and save the predictions for the input data.
    # Inputs
    # Outputs
    def predict_and_save(self):
        print('Predicting fairness')
        prediction = self._generate_responses()
        self._save_prediction(prediction)
        print('- End of fairness prediction')
