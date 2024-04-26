from datasets import load_dataset, Dataset, Features, Value
import pandas as pd
import numpy as np
import copy
import os

from clarity.model import Model


# Main class for the Clarity prediction generation.
class ClarityPredictor:
    def __init__(self, device, models_params_dict, question, file_paths):
        self.device = device
        self.predictions_path = file_paths['predictions_path']
        self.test_dataset = self._load_test_data(file_paths, question)
        self.models_dict = self._init_models(models_params_dict)

    # Load the test data from the local path.
    # Inputs
    #   file_paths: dict
    #   question: str
    # Outputs
    #   test_dataset: Dataset
    def _load_test_data(self, file_paths, question):
        test_df = pd.read_csv(file_paths['test_data_fp'])
        test_df['question'] = question
        cols = ['_id', 'personality_item', 'question']
        features = Features(
            {'_id': Value('int16'),
             'personality_item': Value('string'),
             'question': Value('string')})
        test_dataset = Dataset.from_pandas(test_df[cols], features=features)
        
        return test_dataset

    # Initialize the Model classes using the input dictionary of parameters.
    # Inputs
    #   models_params_dict: dict
    # Outputs
    #   models_dict: dict
    def _init_models(self, models_params_dict):
        models_dict = {}
        for model_name, model_params in models_params_dict.items():
            models_dict[model_name] = Model(self.device, model_params,
                                            copy.deepcopy(self.test_dataset))

        return models_dict

    # Merge the predictions of the models stored on the input dictionary.
    # Inputs
    #   predictions_dict: dict
    # Outputs
    #   clarity: pandas.DataFrame
    def _merge_predictions(self, predictions_dict):
        model1_predictions = predictions_dict['model1']
        model2_predictions = predictions_dict['model2']
        clarity = model1_predictions[['_id', 'benchmark', 'model1']].merge(
            model2_predictions[['_id', 'model2']], on='_id', how='left')
        clarity['output'] = clarity[['model1', 'model2']].apply(
            lambda x: np.mean([x.iloc[0], x.iloc[1], x.iloc[1]]), axis=1)
        clarity = clarity[['_id', 'benchmark', 'output']]

        return clarity

    # Use the models to get the predictions which are finally merged.
    # Inputs
    # Outputs
    #   clarity: pandas.DataFrame
    def _predict(self):
        predictions_dict = {}
        for model_name, model in self.models_dict.items():
            predictions = model.predict()
            predictions = predictions.rename(columns={'label': model_name})
            predictions_dict[model_name] = predictions
        clarity = self._merge_predictions(predictions_dict)

        return clarity

    # Save the input prediction dataframe.
    # Inputs
    #   prediction_df: pandas.DataFrame
    # Outputs
    def _save_prediction(self, prediction_df):
        if not os.path.exists(self.predictions_path):
            os.makedirs(self.predictions_path)
        prediction_df.to_csv('{}/clarity.csv'.format(self.predictions_path),
                             index=False)

    # Create and save the predictions for the input data.
    # Inputs
    # Outputs
    def predict_and_save(self):
        print('Predicting Clarity')
        prediction = self._predict()
        self._save_prediction(prediction)
        print('- End of clarity prediction')
