import json
from empathy.predict import EmpathyPredictor
from interview.predict import InterviewPredictor
from fairness.predict import FairnessPredictor
from clarity.predict import ClarityPredictor
from prediction_merger import PredictionsMerger

CONFIG_PATH = 'config/predict.json'
PREDICTORS_MAPPING = {
    'empathy': EmpathyPredictor,
    'interview': InterviewPredictor,
    'fairness': FairnessPredictor,
    'clarity': ClarityPredictor

}


def load_config(config_path):
    with open(config_path, 'r') as config_file:
        return json.load(config_file)


def init_predictors(config):
    predictors = {}
    for benchmark_name, predictor_params in config['benchmarks'].items():
        predictors[benchmark_name] = PREDICTORS_MAPPING[benchmark_name](
            **predictor_params)

    return predictors


def init_merger(config):
    merger = PredictionsMerger(**config['merger_params'])

    return merger


def predict_and_merge(predictors, predictions_merger):
    for _, predictor in predictors.items():
        predictor.predict_and_save()
    predictions_merger.merge_and_save()


if __name__=="__main__":
    config_data = load_config(CONFIG_PATH)
    predictors = init_predictors(config_data)
    merger = init_merger(config_data)
    predict_and_merge(predictors, merger)