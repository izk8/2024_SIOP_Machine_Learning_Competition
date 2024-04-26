from transformers import AutoModelForCausalLM, AutoTokenizer, default_data_collator
from peft import PeftModel, PeftConfig
import numpy as np
import pandas as pd
import torch
import gc


# Main class for the Clarity sub-challenge Models
class Model:
    def __init__(self, device, model_params, test_dataset):
        self.device = device
        self.model, self.tokenizer = self._init(model_params)
        self.test_dataset = self._process_dataset(test_dataset)

    # Initialize the tokenizer using the input parameters.
    # Inputs
    #   model_params: dict
    # Outputs
    #   tokenizer: AutoTokenizer
    def _init_tokenizer(self, model_params):
        tokenizer = AutoTokenizer.from_pretrained(model_params['model_id'])
        tokenizer.pad_token_id = tokenizer.eos_token_id
        self.max_length = model_params['max_length']
        self.max_tokens = model_params['max_tokens']
        self.eos_token = model_params['eos_token']

        return tokenizer

    # Initialize the model using the input parameters.
    # Inputs
    #   model_params: dict
    # Outputs
    #   model: AutoModelForCausalLM
    def _init_model(self, model_params):
        config = PeftConfig.from_pretrained(
            model_params['peft_id'])
        model = AutoModelForCausalLM.from_pretrained(
            model_params['model_id'])
        model = PeftModel.from_pretrained(model,
                                          model_params['peft_id'])
        model = model.to(self.device)

        return model

    # Initialize model and tokenizer calling the separate initialization
    # functions.
    # Inputs
    #   model_params: dict
    # Outputs
    #   model: AutoModelForCausalLM
    #   tokenizer: AutoTokenizer
    def _init(self, model_params):
        model = self._init_model(model_params)
        tokenizer = self._init_tokenizer(model_params)

        return model, tokenizer

    # Preprocess the input data in the form accepted by the model.
    # Inputs:
    #   examples: pandas.DataFrame
    # Outputs:
    #   model_inputs: dict
    def _test_preprocess_function(self, examples):
        batch_size = len(examples['_id'])
        inputs = [f"Personality Item: {x} - Question: {y} - Answer: " for x, y
                  in zip(examples['personality_item'], examples['question'])]
        model_inputs = self.tokenizer(inputs)
        for i in range(batch_size):
            sample_input_ids = model_inputs["input_ids"][i]
            model_inputs["input_ids"][i] = [self.tokenizer.pad_token_id] * (
                    self.max_length - len(
                sample_input_ids)) + sample_input_ids
            model_inputs["attention_mask"][i] = [0] * (
                    self.max_length - len(sample_input_ids)) + \
                                                model_inputs["attention_mask"][
                                                    i]
            model_inputs["input_ids"][i] = torch.tensor(
                model_inputs["input_ids"][i][-self.max_length:])
            model_inputs["attention_mask"][i] = torch.tensor(
                model_inputs["attention_mask"][i][-self.max_length:])

        return model_inputs

    # Process and tokenize dataset using the utility function.
    # Inputs
    #   test_dataset: pandas.DataFrame
    # Outputs
    #   processed_dataset: dict
    def _process_dataset(self, test_dataset):
        processed_dataset = test_dataset.map(
            self._test_preprocess_function,
            batched=True,
            num_proc=1,
            load_from_cache_file=False
        )

        return processed_dataset

    # Use the model to create the predictions of the test dataset.
    # Inputs
    # Outputs
    #   results: pandas.DataFrame
    def predict(self):
        res_dict = {'_id': [], 'benchmark': [], 'label': []}
        self.model.eval()
        with torch.no_grad():
            for i in self.test_dataset:
                outputs = self.model.generate(
                    input_ids=torch.tensor(
                        np.array(i["input_ids"]).reshape(1, -1)).to(self.device),
                    attention_mask=torch.tensor(
                        np.array(i["attention_mask"]).reshape(1, -1)).to(self.device),
                    max_new_tokens=self.max_tokens,
                    eos_token_id=self.eos_token)
                res = self.tokenizer.batch_decode(
                    outputs.detach().cpu().numpy(),
                    skip_special_tokens=True)
                label = float(res[0].partition('Answer:')[-1].strip())
                res_dict['_id'].append(i['_id'])
                res_dict['benchmark'].append('clarity')
                res_dict['label'].append(label)

        results = pd.DataFrame.from_dict(res_dict)

        # delete model
        del self.model
        gc.collect()
        torch.cuda.empty_cache()

        return results
