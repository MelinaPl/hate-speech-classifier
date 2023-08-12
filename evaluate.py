import pandas as pd
import os
from transformers import AutoTokenizer, AutoModelForSequenceClassification, DataCollatorWithPadding
from transformers import  TrainingArguments, Trainer, set_seed
import numpy as np
import torch
from sklearn.metrics import classification_report

from datasets import Dataset

#### Set device
device = torch.device("mps")

#### Set paths
WORKING_DIR = os.path.dirname(__file__)
DATA_DIR = "data"
MODEL_DIR = os.path.join(WORKING_DIR, "models")
full_path = os.path.join(WORKING_DIR, DATA_DIR)
tokenizer = AutoTokenizer.from_pretrained("dbmdz/bert-base-german-cased")

#### Define methods for preprocessing and evaluation metrics
def preprocess_function(examples):
    text = examples["texts"]
    return tokenizer(text, padding=True, truncation=True, return_tensors="pt").to(device)

def compute_metrics(eval_preds):
    labels = eval_preds.label_ids
    preds = eval_preds.predictions.argmax(-1)
    report = classification_report(labels, preds, target_names=list(labels_to_hs_ids.keys()) , zero_division=0, output_dict=True)
    df_report = pd.DataFrame(report).transpose()
    print(df_report.to_latex())
    return report

#### Create label mappings (comment out the ones that are not needed)
# labels_to_hs_ids = {'offensive': 0, 'other': 1}
# ids_to_hs_labels = {0: 'offensive', 1: 'other'}
labels_to_hs_ids = {'abuse': 0, 'explicit': 1, 'implicit': 2, 'insult': 3, 'other': 4, 'profanity': 5}
ids_to_hs_labels = {0: 'abuse', 1: 'explicit', 2: 'implicit', 3: 'insult', 4: 'other', 5: 'profanity'}


#### Read in data
test = pd.read_csv(os.path.join(full_path,"multi_test_v1_dataset.csv"))
train = pd.read_csv(os.path.join(full_path,"multi_train_v1_dataset.csv"))
train_dataset = Dataset.from_pandas(train)
test_dataset = Dataset.from_pandas(test)
tokenized_data_test = test_dataset.map(preprocess_function, batched=True, remove_columns="texts")
tokenized_data_train = train_dataset.map(preprocess_function, batched=True, remove_columns="texts")

#### Load model and show a test prediction
model = AutoModelForSequenceClassification.from_pretrained(os.path.join(MODEL_DIR, "model_cmv1/checkpoint-300/"), local_files_only=True)
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
input_text = "@SPIEGEL_Politik[SEP]Warum veröffentlicht ihr keine Bilder von linken Anarchisten?[SEP]Die gibt es zuhauf![SEP]Aber ihr seit nur ein Unterstützer dieser Horden![SEP]DIRECTIVE[SEP]DIRECTIVE[SEP]ASSERTIVE[SEP]ASSERTIVE"
inputs = tokenizer(input_text, return_tensors="pt")
with torch.no_grad():
    logits = model(**inputs).logits
predicted_class_id = logits.argmax().item()
print(model.config.id2label[predicted_class_id])


training_args = TrainingArguments(
    output_dir=os.path.join(MODEL_DIR, "model_cmv1"),
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
    model=model,
    args=training_args,
    train_dataset=tokenized_data_train,
    eval_dataset=tokenized_data_test,
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics
)

#### Evaluate
print(trainer.evaluate())