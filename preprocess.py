import os
import re
import pandas as pd
from sklearn.model_selection import train_test_split


speech_acts = ['ASSERTIVE', 'COMOTH', 'DIRECTIVE','EXPRESSIVE', 'UNSURE']

#### Write method to create multiclass hs dataset
def create_dataset(df, tweet_names, write_file=False):
    """
    Takes the df created from the file "annotations_with_text.json" as input as well as 
    a list of the corresponding tweet_names and returns a df (with option to write to file)
    """
    all_texts, all_sa_labels, all_hs_labels = [], [], []
    for tweet_name in tweet_names:
        hate_label = re.findall(pattern=pattern, string=tweet_name)
        hate_label = hate_label[0].strip(".xml").replace("_", "")
        tweet = df[tweet_name] 
        texts, sa_labels = [], []
        for value in tweet:
            sentences = value['sentences']
            for sentence in sentences:
                text = sentences[sentence]['text']
                texts.append(text)
                if sentences[sentence]['coarse'] == "COMMISSIVE" or sentences[sentence]['coarse'] == "OTHER":
                    coarse = "COMOTH"
                else:
                    coarse = sentences[sentence]['coarse']
                sa_labels.append(coarse)
        all_hs_labels.append(hate_label)
        all_texts.append(texts)
        all_sa_labels.append(sa_labels)
    final_df = pd.DataFrame({"texts": all_texts, "labels": all_sa_labels, "hs_labels": all_hs_labels})
    if write_file:
        final_df.to_csv('dataset.csv', index=False)
    return final_df

def transform_to_binary(train_df, test_df):
    """
    Converts the train and test split into splits with binary hs labels
    """
    train_labels, test_labels = [],[]
    for train in train_df["labels"]:
        if train == 4:
            train_labels.append(1) # other
        else:
            train_labels.append(0) # offensive
    new_train_df = pd.DataFrame({"texts": train_df["texts"].values.tolist(),"labels": train_labels})
    for test in test_df["labels"]:
        if test == 4:
            test_labels.append(1) # other
        else:
            test_labels.append(0) # offensive
    new_test_df = pd.DataFrame({"texts": test_df["texts"].values.tolist(),"labels": test_labels})
    return new_train_df, new_test_df

def erase_labels(train_df, test_df):
    """
    Erases the speech acts labels from the train and test split
    """
    train_texts, test_texts = [],[]
    for text in train_df["texts"]:
        new_text = []
        splitted = text.split("[SEP]")
        for sentence in splitted:
            if sentence.replace("[SEP]","") in speech_acts:
                break
            else:
                new_text.append(sentence)
        train_texts.append("[SEP]".join(new_text))
    new_train_df = pd.DataFrame({"texts": train_texts,"labels": train_df["labels"].values.tolist()})
    for text in test_df["texts"]:
        new_text = []
        splitted = text.split("[SEP]")
        for sentence in splitted:
            print(sentence)
            if sentence.replace("[SEP]", "") in speech_acts:
                break
            else:
                new_text.append(sentence)
        test_texts.append("[SEP]".join(new_text))
    new_test_df = pd.DataFrame({"texts": test_texts,"labels": test_df["labels"].values.tolist()})
    return new_train_df, new_test_df

def transform_features(train_df, test_df):
    """
    Transforms how the speech act features are included in the train and test split
    """
    train_texts, test_texts = [],[]
    for text in train_df["texts"]:
        sa_occurences = set()
        new_text = []
        splitted = text.split("[SEP]")
        for sentence in splitted:
            if sentence.replace("[SEP]","") in speech_acts:
                sa_occurences.add(sentence.replace("[SEP]",""))
            else:
                new_text.append(sentence)
        new_text.extend(list(sorted(sa_occurences)))
        train_texts.append("[SEP]".join(new_text))
    new_train_df = pd.DataFrame({"texts": train_texts,"labels": train_df["labels"].values.tolist()})
    for text in test_df["texts"]:
        sa_occurences = set()
        new_text = []
        splitted = text.split("[SEP]")
        for sentence in splitted:
            if sentence.replace("[SEP]","") in speech_acts:
                sa_occurences.add(sentence.replace("[SEP]",""))
            else:
                new_text.append(sentence)
        new_text.extend(list(sorted(sa_occurences)))
        test_texts.append("[SEP]".join(new_text))
    new_test_df = pd.DataFrame({"texts": test_texts,"labels": test_df["labels"].values.tolist()})
    return new_train_df, new_test_df

if __name__ == "__main__":

    #### Set paths
    WORKING_DIR = os.path.dirname(__file__)
    DATA_DIR = "data"
    full_path = os.path.join(WORKING_DIR, DATA_DIR)
    df = pd.read_json(full_path + "/annotations_with_text.json")
    tweet_names = list(df.columns)
    pattern = "_[a-z]+.xml"

    #### Transform speech act labels + hate speech labels to numbers
    final_df = create_dataset(df, tweet_names) 

    ### Transform hate speech

    hs_labels = [i for i in final_df['hs_labels'].values.tolist()]
    unique_labels = set()

    for lb in hs_labels:
        if lb not in unique_labels:
            unique_labels.add(lb)
    hs_labels_to_ids = {k: v for v, k in enumerate(sorted(unique_labels))}
    ids_to_hs_labels = {v: k for v, k in enumerate(sorted(unique_labels))}

    ### Transform speech acts

    sa_labels = []
    for sa in final_df['labels'].values.tolist():
        sa_labels.extend(sa)
    unique_labels = set()

    for lb in sa_labels:
        if lb not in unique_labels:
            unique_labels.add(lb)
    sa_labels_to_ids = {k: v for v, k in enumerate(sorted(unique_labels))}
    ids_to_sa_labels = {v: k for v, k in enumerate(sorted(unique_labels))}

    ### Prints all label mappings
    print(ids_to_hs_labels)
    print(ids_to_sa_labels)
    print(sa_labels_to_ids)
    print(hs_labels_to_ids)

    #### Concatenate speech act labels to text and separating them using [SEP]
    texts_with_features = []
    labels = []
    for text, sa_label, hs_label in zip(final_df['texts'].values.tolist(), final_df['labels'].values.tolist(), final_df['hs_labels'].values.tolist()):
        text.extend(sa_label)
        joined = "[SEP]".join(text) # still maintain sentence boundaries via [SEP]
        texts_with_features.append(joined)

    #### Create dataframe with speech act features and write to csv 
    labels = [hs_labels_to_ids[i] for i in final_df['hs_labels'].values.tolist()]
    transformed_df = pd.DataFrame({"texts": texts_with_features, "labels" : labels})
    transformed_df.to_csv("data/transformed_dataset.csv", index=False)

    #### Split data into training and test set (80/20)
    train_df, test_df = train_test_split(transformed_df, test_size=0.2, random_state=200, shuffle=True, stratify=final_df["labels"])
    train_df.to_csv('data/multi_train_v1_dataset.csv', index=False) 
    test_df.to_csv('data/multi_test_v1_dataset.csv', index=False)

    #### Transform data to binary hs labels and write to file
    binary_train, binary_test = transform_to_binary(train_df, test_df)
    binary_train.to_csv('data/binary_train_v1_dataset.csv', index=False) 
    binary_test.to_csv('data/binary_test_v1_dataset.csv', index=False)

    #### BINARY: Erase speech act features ('nofeat')
    train, test = pd.read_csv("data/binary_train_v1_dataset.csv"), pd.read_csv("data/binary_test_v1_dataset.csv")
    binary_nofeat_train, binary_nofeat_test = erase_labels(train, test)
    binary_nofeat_train.to_csv('data/binary_train_nofeat_dataset.csv', index=False) 
    binary_nofeat_test.to_csv('data/binary_test_nofeat_dataset.csv', index=False)

    #### BINARY: Convert to speech act feature version 'v2'
    train, test = pd.read_csv("data/binary_train_v1_dataset.csv"), pd.read_csv("data/binary_test_v1_dataset.csv")
    binary_v2_train, binary_v2_test = transform_features(train, test)
    binary_v2_train.to_csv('data/binary_train_v2_dataset.csv', index=False) 
    binary_v2_test.to_csv('data/binary_test_v2_dataset.csv', index=False)

    ##### MULTI: Erase speech act features ('nofeat')
    train, test = pd.read_csv("data/multi_train_v1_dataset.csv"), pd.read_csv("data/multi_test_v1_dataset.csv")
    multi_train_nofeat, multi_test_nofeat = erase_labels(train, test)
    multi_train_nofeat.to_csv('data/multi_train_nofeat_dataset.csv', index=False) 
    multi_test_nofeat.to_csv('data/multi_test_nofeat_dataset.csv', index=False)

    #### MULTI: Convert to speech act feature version 'v2'
    train, test = pd.read_csv("data/multi_train_v1_dataset.csv"), pd.read_csv("data/multi_test_v1_dataset.csv")
    multi_train_v2, multi_test_v2 = transform_features(train, test)
    multi_train_v2.to_csv('data/multi_train_v2_dataset.csv', index=False) 
    multi_test_v2.to_csv('data/multi_test_v2_dataset.csv', index=False)