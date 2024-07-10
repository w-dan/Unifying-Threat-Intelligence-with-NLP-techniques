import pandas as pd
import torch
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from pymongo import MongoClient
import pandas as pd
import torch
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import nltk
from pymongo import MongoClient
from sklearn.model_selection import train_test_split
from transformers import PreTrainedTokenizer
from typing import List, Tuple
import numpy as np
from typing import List, Any
import torch
from transformers import Trainer, TrainingArguments, AdamW, get_linear_schedule_with_warmup, TrainerCallback
from torch.optim.lr_scheduler import ReduceLROnPlateau

class CustomTrainerCallback(TrainerCallback):
    def __init__(self, scheduler):
        self.scheduler = scheduler

    def on_evaluate(self, args, state, control, **kwargs):
        metrics = kwargs.get('metrics', {})
        eval_metric = metrics['eval_loss']
        self.scheduler.step(eval_metric, state.global_step)

class CustomReduceLROnPlateau(ReduceLROnPlateau):
    def __init__(self, optimizer, mode='min', factor=0.5, patience=10, threshold=0.01, threshold_mode='rel', cooldown=0, min_lr=0, eps=1e-08, verbose=False):
        super(CustomReduceLROnPlateau, self).__init__(optimizer, mode, factor, patience, threshold, threshold_mode, cooldown, min_lr, eps, verbose)

    def step(self, metrics, epoch=None):
        current = float(metrics)
        if epoch is None:
            epoch = self.last_epoch + 1
        self.last_epoch = epoch

        if self.is_better(current, self.best):
            self.best = current
            self.num_bad_epochs = 0
        else:
            self.num_bad_epochs += 1

        if self.num_bad_epochs > self.patience:
            self._reduce_lr(epoch)
            self.num_bad_epochs = 0

    def _reduce_lr(self, epoch):
        for i, param_group in enumerate(self.optimizer.param_groups):
            old_lr = float(param_group['lr'])
            new_lr = max(old_lr * self.factor, self.min_lrs[i])
            if old_lr - new_lr > self.eps:
                param_group['lr'] = new_lr
                if self.verbose:
                    print('Epoch {:5d}: reducing learning rate to {:.4e}.'.format(epoch, new_lr))


class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx], dtype=torch.float)
        return item

    def __len__(self):
        return len(self.labels)

def download_nltk_data():
    nltk.download('punkt')
    nltk.download('stopwords')


def preprocess_corpus(text):
    """
    Preprocesses a given text by removing stopwords and setting lowercase.

    Args:
    text: The text to be preprocessed.

    Returns:
    The preprocessed text.
    """

    tokens = word_tokenize(text.lower())
    stop_words = set(stopwords.words('english'))
    filtered_tokens = [token for token in tokens if token not in stop_words]

    preprocessed_text = ' '.join(filtered_tokens)

    return preprocessed_text


def fetch_and_preprocess_data(db_name: str, collection_name: str, connection_string: str, preprocess: bool, field_to_get: str, include_tactics: bool) -> pd.DataFrame:
    """
    Fetch documents from the MongoDB collection, preprocess the corpus, and return a DataFrame.
    
    Args:
    db_name (str): The name of the MongoDB database.
    collection_name (str): The name of the MongoDB collection.
    connection_string (str): The MongoDB connection string.
    preprocess (bool): Flag indicating whether to preprocess the corpus.
    field_to_get (str): The specific field to fetch from the documents.
    include_tactics (bool): Flag indicating whether to include tactics in the DataFrame.
    
    Returns:
    pd.DataFrame: A DataFrame containing the preprocessed data.
    """
    download_nltk_data()

    client = MongoClient(connection_string)
    db = client[db_name]
    collection = db[collection_name]
    
    documents = collection.find()
    
    data = []
    for doc in documents:
        original_corpus = doc.get("corpus", "")
        tactics = doc.get(field_to_get, [])
        
        if preprocess:
            preprocessed_corpus = preprocess_corpus(original_corpus)
        else:
            preprocessed_corpus = original_corpus

        data_entry = {
            "corpus": preprocessed_corpus
        }
        if include_tactics:
            data_entry["tactics"] = tactics

        data.append(data_entry)

    df = pd.DataFrame(data)

    print(f"[+] Shape: {df.shape}")
    print(df.head())

    return df

def get_onehot_data(database_name, collection_name, client_uri='mongodb://localhost:27017/'):
    client = MongoClient(client_uri)
    db = client[database_name]
    collection = db[collection_name]

    documents = collection.find()
    for doc in documents:
        print(doc.keys())
        break

    documents = collection.find()

    data = []
    for doc in documents:
        corpus = doc.get("corpus", "")
        tactics = doc.get("tactics", [])
        
        data.append({
            "corpus": preprocess_corpus(corpus),
            "tactics": tactics
        })

    df = pd.DataFrame(data)

    print(f"[+] Shape: {df.shape}")
    #print(df.head())

    df['tactics_length'] = df['tactics'].apply(len)
   #  print(df['tactics_length'].value_counts())

    tactics = df['tactics'].apply(pd.Series).stack().reset_index(drop=True).unique()
    one_hot_df = pd.get_dummies(df['tactics'].apply(pd.Series).stack()).groupby(level=0).sum()
    one_hot_df = one_hot_df.reindex(columns=tactics, fill_value=0)

    df_one_hot_encoded = df.drop(columns=['tactics']).join(one_hot_df)

    print(f"[+] One-Hot Encoded DataFrame Shape: {df_one_hot_encoded.shape}")
    # print(df_one_hot_encoded.head())

    return df_one_hot_encoded

def process_tactics(df: pd.DataFrame) -> pd.DataFrame:
    """
    Process the 'tactics' column in the DataFrame to create a one-hot encoded DataFrame.

    Args:
    df (pd.DataFrame): The input DataFrame containing a 'tactics' column.

    Returns:
    pd.DataFrame: The DataFrame with one-hot encoded 'tactics' and a new column 'tactics_length'.
    """
    df['tactics_length'] = df['tactics'].apply(len)
    print(df['tactics_length'].value_counts())

    # unique tactics
    tactics = df['tactics'].apply(pd.Series).stack().reset_index(drop=True).unique()
    
    # one-hot encoding
    one_hot_df = pd.get_dummies(df['tactics'].apply(pd.Series).stack()).groupby(level=0).sum()
    one_hot_df = one_hot_df.reindex(columns=tactics, fill_value=0)
    df_one_hot_encoded = df.drop(columns=['tactics']).join(one_hot_df)

    return df_one_hot_encoded


def convert_to_float32(x):
    if isinstance(x, list) or isinstance(x, np.ndarray):
        return np.array(x, dtype=np.float32)
    else:
        return np.float32(x)


def prepare_datasets(texts: List[str], labels: List[int], tokenizer: PreTrainedTokenizer, test_size: float = 0.3, val_test_size: float = 0.5, random_state: int = 42) -> Tuple[CustomDataset, CustomDataset, CustomDataset]:
    """
    Prepares the training, validation, and test datasets.

    Args:
    texts (List[str]): The input texts.
    labels (List[int]): The corresponding labels.
    tokenizer (PreTrainedTokenizer): The tokenizer to use.
    test_size (float): The proportion of the dataset to include in the test split (default is 0.3).
    val_test_size (float): The proportion of the temporary dataset to include in the validation split (default is 0.5).
    random_state (int): The seed used by the random number generator (default is 42).

    Returns:
    Tuple[CustomDataset, CustomDataset, CustomDataset]: The training, validation, and test datasets.
    """
    
    # 1. split into training and temporary (which will be divided in 2) datasets
    train_texts, temp_texts, train_labels, temp_labels = train_test_split(
        texts, labels, test_size=test_size, random_state=random_state
    )

    # 2. split into validation and test (15% of total each)
    val_texts, test_texts, val_labels, test_labels = train_test_split(
        temp_texts, temp_labels, test_size=val_test_size, random_state=random_state
    )

    # 3. tokenize
    train_encodings = tokenizer(list(train_texts), truncation=True, padding=True)
    val_encodings = tokenizer(list(val_texts), truncation=True, padding=True)
    test_encodings = tokenizer(list(test_texts), truncation=True, padding=True)

    # 4. create the datasets
    train_dataset = CustomDataset(train_encodings, train_labels)
    val_dataset = CustomDataset(val_encodings, val_labels)
    test_dataset = CustomDataset(test_encodings, test_labels)

    return train_dataset, val_dataset, test_dataset


def plot_training_history(train_logs, eval_results, trial_number):
    epochs = range(1, len(train_logs['metrics']['train_loss']) + 1)
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, train_logs['metrics']['train_loss'], 'b-', label='Training Loss')
    plt.plot(epochs, eval_results['eval_loss'], 'r-', label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title(f'Training vs Validation Loss (Trial {trial_number})')
    plt.legend()
    plt.show()

def plot_confusion_matrix(y_true, y_pred, labels, title):
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title(title)
    plt.show()


def predict(val_dataset):
    predictions = trainer.predict(val_dataset)
    pred_probs = torch.sigmoid(torch.tensor(predictions.predictions))
    pred_labels = (pred_probs >= 0.5).int().numpy()
    return pred_labels


def evaluate_predictions(true_labels, pred_labels):
    target_names = [f'Class {i}' for i in range(true_labels.shape[1])]

    print(classification_report(true_labels, pred_labels, target_names=target_names))
    cm = multilabel_confusion_matrix(true_labels, pred_labels)

    fig, axes = plt.subplots(nrows=cm.shape[0], figsize=(10, cm.shape[0] * 4))
    for i, ax in enumerate(axes):
        sns.heatmap(cm[i], annot=True, fmt='d', xticklabels=['Not ' + target_names[i], target_names[i]],
                    yticklabels=['Not ' + target_names[i], target_names[i]], ax=ax)
        ax.set_xlabel('Predicted')
        ax.set_ylabel('True')
        ax.set_title(f'Confusion Matrix for {target_names[i]}')

    plt.tight_layout()
    plt.show()


def perform_inference(val_dataset):
    model.eval()
    dataloader = DataLoader(val_dataset, batch_size=8)
    pred_probs_list = []

    with torch.no_grad():
        for batch in dataloader:
            inputs = {key: val.to(device) for key, val in batch.items() if key != 'labels'}
            outputs = model(**inputs)
            logits = outputs.logits
            pred_probs = torch.sigmoid(logits)
            pred_probs_list.append(pred_probs.cpu())

    pred_probs = torch.cat(pred_probs_list, dim=0)
    pred_labels = (pred_probs >= 0.5).int().numpy()
    return pred_labels

def evaluate_inference(true_labels, pred_labels):
    target_names = one_hot_df.columns

    print(classification_report(true_labels, pred_labels, target_names=target_names))
    cm = multilabel_confusion_matrix(true_labels, pred_labels)

    fig, axes = plt.subplots(nrows=cm.shape[0], figsize=(10, cm.shape[0] * 4))
    for i, ax in enumerate(axes):
        sns.heatmap(cm[i], annot=True, fmt='d', xticklabels=['Not ' + target_names[i], target_names[i]],
                    yticklabels=['Not ' + target_names[i], target_names[i]], ax=ax)
        ax.set_xlabel('Predicted')
        ax.set_ylabel('True')
        ax.set_title(f'Confusion Matrix for {target_names[i]}')

    plt.tight_layout()
    plt.show()
