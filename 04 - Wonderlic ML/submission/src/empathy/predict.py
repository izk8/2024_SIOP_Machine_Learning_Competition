from setfit import SetFitModel
import pandas as pd
import os


# Main class for the Empathy prediction generation.
class EmpathyPredictor:
    def __init__(self, model_path, test_data_path, predictions_path):
        self.model = self._load_model(model_path)
        self.test_data = self._load_data(test_data_path)
        self.predictions_path = predictions_path

    # Load the SetFit model from the HuggingFace hub.
    # Inputs
    #   model_path: str
    # Outputs
    #   model: SetFitModel
    def _load_model(self, model_path):
        model = SetFitModel.from_pretrained(model_path)

        return model

    # Load the test data from the local path.
    # Inputs
    #   test_data_path: str
    # Outputs
    #   test_data: pandas.DataFrame
    def _load_data(self, test_data_path):
        test_data = pd.read_csv(test_data_path)

        return test_data

    # Save the input prediction dataframe.
    # Inputs
    #   prediction_df: pandas.DataFrame
    # Outputs
    def _save_prediction(self, prediction_df):
        if not os.path.exists(self.predictions_path):
            os.makedirs(self.predictions_path)
        prediction_df.to_csv('{}/empathy.csv'.format(self.predictions_path),
                             index=False)

    # Use the model to get the predictions.
    # Inputs
    # Outputs
    #   empathy_df: pandas.DataFrame
    def _predict(self):
        res = self.model(self.test_data['text'].values)
        res = res.detach().numpy()
        empathy = pd.DataFrame(res, columns=['output'])
        empathy_df = pd.concat([self.test_data['_id'], empathy], axis=1)
        empathy_df['benchmark'] = 'empathy'
        empathy_df = empathy_df[['_id', 'benchmark', 'output']]

        return empathy_df

    # Create and save the predictions for the input data.
    # Inputs
    # Outputs
    def predict_and_save(self):
        print('Predicting Empathy')
        prediction = self._predict()
        self._save_prediction(prediction)
        print('- End of empathy prediction')
