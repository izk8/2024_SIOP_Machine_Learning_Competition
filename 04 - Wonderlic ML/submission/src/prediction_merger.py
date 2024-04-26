import pandas
import os


class PredictionsMerger:
    def __init__(self, predictions_folder, merged_prediction_name):
        self.predictions_folder = predictions_folder
        self.merged_prediction_name = merged_prediction_name

    def _merge_predictions(self):
        pred_file_paths = ['{}/{}'.format(self.predictions_folder, file_name)
                           for file_name in os.listdir(self.predictions_folder)
                           if file_name.endswith('.csv')]
        predictions_df = pandas.DataFrame(
            columns=['_id', 'benchmark', 'output'])
        for file_path in pred_file_paths:
            current_pred = pandas.read_csv(file_path)
            predictions_df = pandas.concat([predictions_df, current_pred],
                                           ignore_index=True)

        return predictions_df

    def _save_predictions(self, predictions_df):
        merged_prediction_path = '{}/{}'.format(self.predictions_folder,
                                                self.merged_prediction_name)
        predictions_df.to_csv(merged_prediction_path, index=False)

    def merge_and_save(self):
        predictions_df = self._merge_predictions()
        self._save_predictions(predictions_df)
