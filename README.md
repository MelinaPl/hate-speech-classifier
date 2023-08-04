# Building a Hate Speech Classifier with Speech Act Features

Based on my previous work (https://github.com/MelinaPl/speech-act-classifier), the main aim of this project is to build a hate speech classifier and test whether including speech acts as features improves the performance of the classifier.

This repository was created for a term paper at Humboldt University Berlin.

## Contents

### Data Statistics

- Notebook `data_statistics.ipynb`

### Data Preprocessing

- Python script `preprocess.py`

### Training 

- Python script `train_transformers.py`

### Evaluation

- Python script `evaluate.py`

### Error Analysis

- Folder `pictures` containing confusion matrixes of every model
- Notebook `error_analysis.ipynb` containing the corresponding scripts

## Notes

**Models**

In total, 12 models were trained. The experiments had three factors following a 2 x 2 x 3 design: 

- Casing: Uncased, Cased
- Labels: Binary, fine-grained
- Speech act features: none, v1, v2

The model names followed the schema:

- `model_` +
- `c` for cased or `u` for uncased +
- `b` for binary or `m` for multiple labels (fine-grained) +
- `n` for no feature inclusion, `v1` for feature inclusion (version 1) or `v2` for feature inclusion (version 2)

**Dataset**

- The dataset can be retrieved from the repository: https://github.com/MelinaPl/speech-act-analysis
- More specifically, the file `annotations_with_text.json` was used within this project

