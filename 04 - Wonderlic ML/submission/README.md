# Wonderlic ML Team SIOP 2024 Competition

This repo contains the code to generate the results on the Test data of the 
SIOP ML competition. 

An in-depth description of the competition can be found at [SIOP ML 
Competition 2024](https://eval.ai/web/challenges/challenge-page/2207/overview).

## Getting Started

### Dependencies
This code has been tested on an Amazon EC2 with the following settings
* **Instance name**: g5.4xlarge
* **Ubuntu 20.04**
* **Deep Learning AMI**
* **Python 3.10**

**NOTE:** To run the code you must have access to the **OpenAI APIs** from 
which you can get a secret key as explained [here](https://help.openai.com/en/articles/4936850-where-do-i-find-my-openai-api-key).

### Requirements
| Library | Version |
| ------- | ------- |
| datasets | 2.18.0 |
| numpy | 1.26.4 |
| openai | 1.16.2 |
| pandas | 2.2.1 |
| peft | 0.10.0 |
| sentence_transformers | 2.6.1 |
| setfit | 1.0.3 |
| torch | 2.2.0 |
| tqdm | 4.66.2 |
| transformers | 4.39.3 |

### Executing the program on the test data
* Copy your secret key on the `config/predict.json` file where `secret_key` 
is specified
* Download and install [Anaconda](https://anaconda.org). 
* [Create](https://conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html) a new anaconda environment with `python 3.10`
* Activate the environment 
* Install all the dependencies from the command line using the following 
command 
    ```
    pip install -r requirements.txt
    ```
* Run the `main_predict.py` file
  ```
  python main_predict.py
  ```
  This step will produce the files used for the submission into `results/test`.

## Authors
Our team is the Data Science and Engineering team at [Wonderlic Inc.](https://wonderlic.com)

Please, contact us for any comment, bug or help.

**Guglielmo Menchetti - Senior Machine Learning Engineer**
* guglielmo.menchetti@wonderlic.com

**Lea Cleary - Machine Learning Engineer**
* lea.cleary@wonderlic.com

**Annie Brinza - Manager of Data Science and Engineering**
* annie.brinza@wonderlic.com
