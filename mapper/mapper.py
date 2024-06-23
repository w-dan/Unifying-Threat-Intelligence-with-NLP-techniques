import pdfplumber
import re
import json
from transformers import BertTokenizer, BertForSequenceClassification
import torch
from typing import List, Dict
from utils import preprocess_corpus
import stix2
from stix2 import Indicator, Bundle, AttackPattern
import uuid
import datetime

from constants import MITRE_DESCRIPTIONS

# regex for IOCs
ip_regex = re.compile(r'\b(?:[0-9]{1,3}\.){3}[0-9]{1,3}(?::\d{1,5})?\b')
domain_regex = re.compile(r'\b(?:[a-zA-Z0-9-]+\.)+[a-zA-Z]{2,6}\b')
url_regex = re.compile(r'https?://(?:[-\w.]|(?:%[\da-fA-F]{2}))+')
file_regex = re.compile(r'\b(?:[a-zA-Z]:\\|/)?(?:[\w\s]+\\|/)*[\w\s]+\.\w+\b')


# labels for mapping
labels = [
    "Execution", "Persistence", "Credential Access", "Collection", "Defense Evasion", 
    "Exploitation", "Privilege Escalation", "Exfiltration", "Command and Control", 
    "Discovery", "Initial Access", "Lateral Movement", "Impact", "Reconnaissance", 
    "Delivery", "Resource Development"
]


def pdf_to_text(pdf_path: str) -> str:
    """
    Extracts text from a PDF file using pdfplumber.

    Args:
    pdf_path (str): Path to the PDF file.

    Returns:
    str: Extracted text from the PDF.
    """
    with pdfplumber.open(pdf_path) as pdf:
        text = ''
        for page in pdf.pages:
            text += page.extract_text()
    return text


# BERT and tokenizer functions
def load_model_and_tokenizer(model_path: str):
    """
    Loads the BERT model and tokenizer from a specified path.

    Args:
    model_path (str): Path to the directory containing the model and tokenizer files.

    Returns:
    Tuple: Loaded BERT model and tokenizer.
    """
    tokenizer = BertTokenizer.from_pretrained(model_path)
    model = BertForSequenceClassification.from_pretrained(model_path)
    return tokenizer, model


def map_onehot_to_labels(onehot_array: List[int]) -> List[str]:
    """
    Maps a one-hot list to a list of tactic IDs.

    Args:
    onehot_array (List[int]): List of one-hot values.

    Returns:
    List[str]: List of tactic IDs corresponding to the one-hot values.
    """
    tactic_ids = list(MITRE_DESCRIPTIONS.keys())
    return [tactic_ids[idx] for idx, value in enumerate(onehot_array) if value == 1]


def infer_tactics(preprocessed_corpus: str, tokenizer: BertTokenizer, model: BertForSequenceClassification, threshold: float = 0.5) -> List[str]:
    """
    Performs multi-label inference on the preprocessed corpus using the BERT model.

    Args:
    preprocessed_corpus (str): Preprocessed text corpus.
    tokenizer (BertTokenizer): Tokenizer for the BERT model.
    model (BertForSequenceClassification): Loaded BERT model.
    threshold (float): Threshold for deciding if a label is active.

    Returns:
    List[str]: List of predicted labels.
    """
    # obtaining output
    inputs = tokenizer(preprocessed_corpus, return_tensors='pt', truncation=True, padding=True)
    outputs = model(**inputs)
    logits = outputs.logits
    # securing binary output
    probabilities = torch.sigmoid(logits)
    predictions = (probabilities > threshold).int()
    
    # converting to list and mapping
    prediction_list = predictions.squeeze().tolist()

    return map_onehot_to_labels(prediction_list)


def main(pdf_path: str) -> None:
    """
    Main function to process the PDF, extract and preprocess text, find IOCs, perform model inference, 
    and print the resulting JSON.

    Args:
    pdf_path (str): Path to the PDF file.
    """
    # processing text
    print("[📝] Processing text")
    raw_text = pdf_to_text(pdf_path)
    preprocessed_corpus = preprocess_corpus(raw_text)
    
    ip_addresses = ip_regex.findall(preprocessed_corpus)
    domains = domain_regex.findall(preprocessed_corpus)
    urls = url_regex.findall(preprocessed_corpus)
    files = file_regex.findall(preprocessed_corpus)
    
    # model and inference
    print("[🤖] Loading model and performing inference")
    model_path = '../models/bert_multitag'
    tokenizer, model = load_model_and_tokenizer(model_path)
    mitre_tactics_ids = infer_tactics(preprocessed_corpus, tokenizer, model)
    
    # stix2 indicators
    print("[🔍] Generating STIX 2.0 Indicators")
    indicators = []

    for ip in ip_addresses:
        indicators.append(Indicator(name="IP Address", pattern=f"[ipv4-addr:value = '{ip}']", pattern_type="stix"))

    for domain in domains:
        indicators.append(Indicator(name="Domain", pattern=f"[domain-name:value = '{domain}']", pattern_type="stix"))

    for url in urls:
        indicators.append(Indicator(name="URL", pattern=f"[url:value = '{url}']", pattern_type="stix"))

    for file in files:
        indicators.append(Indicator(name="File", pattern=f"[file:name = '{file}']", pattern_type="stix"))

    # AttackPattern objects for MITRE tactics
    for tactic_id in mitre_tactics_ids:
        tactic_info = MITRE_DESCRIPTIONS[tactic_id]
        attack_pattern = AttackPattern(
            name=tactic_info["name"],
            description=tactic_info["description"],
            external_references=[
                {
                    "source_name": "mitre-attack",
                    "url": f"https://attack.mitre.org/tactics/{tactic_id}/"
                }
            ],
            created_by_ref="identity--d39689c2-1e26-474c-b2b2-12d37a6dfe4f",
            id="attack-pattern--" + str(uuid.uuid4()),
            created=datetime.datetime.now().isoformat() + "Z"
        )
        indicators.append(attack_pattern)

    bundle = Bundle(objects=indicators)

    # print STIX 2.0 JSON
    print(bundle.serialize(pretty=True))



if __name__ == "__main__":
    pdf_path = "../samples/sample.pdf"
    main(pdf_path)
