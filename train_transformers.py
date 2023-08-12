import pandas as pd
import os
from transformers import AutoTokenizer, AutoModelForSequenceClassification, DataCollatorWithPadding
from transformers import  TrainingArguments, Trainer, set_seed, EarlyStoppingCallback, IntervalStrategy
import numpy as np
import torch
from sklearn.metrics import classification_report
import random
import wandb
from datasets import Dataset
import platform

#### Ensure reproducibility
os.environ['PYTHONHASHSEED']= "123"
set_seed(123)

#### check whether x86 or arm64 is used, should be the latter
print(platform.platform()) 

"""
If its x86, type the following into the console:

$ CONDA_SUBDIR=osx-arm64 conda create -n env_name -c conda-forge
$ conda activate env_name
$ conda config --env --set subdir osx-arm64
$ conda install python=3.9
$ pip install -r requirements.txt

"""

#### Start a new wandb run to track this script
wandb.init(
    project="hate-speech-classifier"
)

#### This ensures that the current MacOS version is at least 12.3+
print(torch.backends.mps.is_available())
#### This ensures that the current PyTorch installation was built with MPS activated.
print(torch.backends.mps.is_built())

#### Set device (only for Mac!)
device = torch.device("mps")

#### Set paths
WORKING_DIR = os.path.dirname(__file__)
DATA_DIR = "data"
MODEL_DIR = os.path.join(WORKING_DIR, "models")
full_path = os.path.join(WORKING_DIR, DATA_DIR)

#### Load datasets
train = pd.read_csv(os.path.join(full_path,"binary_train_v2_dataset.csv"))
test = pd.read_csv(os.path.join(full_path,"binary_test_v2_dataset.csv"))

#### Label mappings
labels_to_hs_ids = {'offensive': 0, 'other': 1}
ids_to_hs_labels = {0: 'offensive', 1: 'other'}
# labels_to_hs_ids = {'abuse': 0, 'explicit': 1, 'implicit': 2, 'insult': 3, 'other': 4, 'profanity': 5}
# ids_to_hs_labels = {0: 'abuse', 1: 'explicit', 2: 'implicit', 3: 'insult', 4: 'other', 5: 'profanity'}


#### Load tokenizer and model

tokenizer = AutoTokenizer.from_pretrained("dbmdz/bert-base-german-uncased")

#### Transform to HF datasets
train_dataset = Dataset.from_pandas(train)
test_dataset = Dataset.from_pandas(test)


#### Create the DataLoaders for our training and validation sets.

def preprocess_function(examples):
    text = examples["texts"]
    return tokenizer(text, padding=True, truncation=True, return_tensors="pt").to(device)

tokenized_data_train = train_dataset.map(preprocess_function, batched=True, remove_columns="texts")
tokenized_data_test = test_dataset.map(preprocess_function, batched=True, remove_columns="texts")

data_collator = DataCollatorWithPadding(tokenizer=tokenizer)


def compute_metrics(eval_preds):
    labels = eval_preds.label_ids
    preds = eval_preds.predictions.argmax(-1)
    report = classification_report(labels, preds, target_names=list(labels_to_hs_ids.keys()) , zero_division=0, output_dict=True)
    df_report = pd.DataFrame(report).transpose()
    print(df_report.to_latex())
    macro_f1 = report["macro avg"]["f1-score"]
    weight_f1 = report["weighted avg"]["f1-score"]
    score_dict = {"macro_f1": macro_f1, "weighted_f1": weight_f1}
    return score_dict


#### Set WANDB 

os.environ["WANDB_PROJECT"]="hate-speech-classifier"
os.environ["WANDB_LOG_MODEL"]="true"
os.environ["WANDB_WATCH"]="false"

def model_init():
    return AutoModelForSequenceClassification.from_pretrained("dbmdz/bert-base-german-uncased",
    num_labels = 2, # The number of output labels
    id2label=ids_to_hs_labels,
    label2id=labels_to_hs_ids).to(device)

training_args = TrainingArguments(
    output_dir=os.path.join(MODEL_DIR, "model_ubv2"),
    learning_rate=2e-5,
    optim="adamw_torch",
    auto_find_batch_size=True,
    save_total_limit=1,
    num_train_epochs=10,
    warmup_ratio=0.1,
    weight_decay=0.01,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="weighted_f1",
    data_seed=123,
    seed=123,
    full_determinism=True,
    use_mps_device=True
)

trainer = Trainer(
    model=model_init(),
    args=training_args,
    train_dataset=tokenized_data_train,
    eval_dataset=tokenized_data_test,
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics
)


#### Start training and evaluate

trainer.train()
print(trainer.evaluate())

#### After training, access the path of the best checkpoint like this
best_ckpt_path = trainer.state.best_model_checkpoint
print(best_ckpt_path)
wandb.finish()
