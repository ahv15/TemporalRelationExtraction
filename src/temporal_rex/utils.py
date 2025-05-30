"""
Utility functions for temporal relation extraction.

This module contains helper functions for data processing, model saving/loading,
graph generation, and other utility operations.
"""

import torch
import os
import spacy
import dgl
import networkx as nx
import numpy as np
import re
from nltk.tokenize.treebank import TreebankWordDetokenizer
from nltk.tokenize import sent_tokenize, word_tokenize
import nltk


# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download("punkt")


def clean_str(text):
    """
    Clean and preprocess text strings.
    
    Args:
        text (str): Input text to clean
        
    Returns:
        str: Cleaned text
    """
    text = text.lower()
    return text.strip()


def save(model, optimizer, output_path="/content/gdrive/MyDrive/model_train.pth"):
    """
    Save model and optimizer state dictionaries.
    
    Args:
        model: PyTorch model to save
        optimizer: Optimizer to save
        output_path (str): Path to save the model
    """
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
        },
        output_path,
    )
    print(f"Model saved to {output_path}")


def load_model(model, optimizer, model_path, device="cpu"):
    """
    Load model and optimizer state dictionaries.
    
    Args:
        model: PyTorch model to load state into
        optimizer: Optimizer to load state into
        model_path (str): Path to the saved model
        device (str): Device to load the model on
        
    Returns:
        tuple: (loaded_model, loaded_optimizer)
    """
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    print(f"Model loaded from {model_path}")
    return model, optimizer


def generateDependencyGraph(edge_dict, train_text, count, total_train_graphs):
    """
    Generate dependency graphs from text using spaCy.
    
    Args:
        edge_dict (dict): Dictionary mapping edge types to IDs
        train_text (list): List of text strings
        count (int): Current count for edge type assignment
        total_train_graphs (list): List to append generated graphs
        
    Returns:
        list: Updated list of dependency graphs
    """
    nlp = spacy.load("en_core_web_sm")
    
    for text in train_text:
        edge_types = []
        node_features = []
        doc = nlp(str(text))
        g = dgl.DGLGraph()
        
        for token in doc:
            g.add_nodes(1)
            node_features.append(token.vector)
            
        for token in doc:
            for child in token.children:
                edge_type = child.dep_.lower()
                if edge_type not in edge_dict.keys():
                    edge_dict[edge_type] = count
                    count = count + 1
                edge_types.append(edge_dict[edge_type])
                g.add_edges(token.i, child.i)
                
        one_hot = torch.nn.functional.one_hot(torch.tensor(edge_types), len(edge_dict))
        g.edata["rel"] = one_hot
        g.ndata["node_feat"] = torch.tensor(node_features)
        total_train_graphs.append(g)
        
    return total_train_graphs


def process_temporal_data(sentences):
    """
    Process temporal data by combining temporal expressions.
    
    Args:
        sentences (list): List of tokenized sentences
        
    Returns:
        list: Processed sentences with combined temporal expressions
    """
    for sentence in sentences:
        i = 0
        while i < len(sentence):
            pat = re.compile(r"^tmx[0-9]*-[0-9]*-")
            temporal = pat.search(sentence[i])
            if temporal:
                tempVal = temporal.group()
                while i + 1 < len(sentence) and sentence[i + 1].startswith(tempVal):
                    sentence[i] += " " + sentence[i + 1].replace(tempVal, "")
                    sentence.remove(sentence[i + 1])
            i = i + 1
    
    return sentences


def extract_entities_and_relations(sentences, relations_):
    """
    Extract entities and relations from processed sentences.
    
    Args:
        sentences (list): List of processed sentences
        relations_ (list): List of relation tuples
        
    Returns:
        list: Processed data with entity markers
    """
    data = []
    sentence1 = {}
    sentence2 = {}
    
    for i, relation in enumerate(relations_):
        check1 = True
        check2 = True
        if relation[0] == relation[1]:
            relations_.remove(relation)
            continue
            
        for ind, sentence in enumerate(sentences):
            for word in sentence:
                if word.startswith(relation[0]) and check1:
                    sentence1[relation[0] + relation[1]] = ind
                    check1 = False
                if word.startswith(relation[1]) and check2:
                    sentence2[relation[0] + relation[1]] = ind
                    check2 = False
        check1 = True
        check2 = True

    count = 0
    for i in range(len(sentence1)):
        try:
            ind1 = min(
                sentence1[relations_[i][0] + relations_[i][1]],
                sentence2[relations_[i][0] + relations_[i][1]],
            )
            ind2 = max(
                sentence1[relations_[i][0] + relations_[i][1]],
                sentence2[relations_[i][0] + relations_[i][1]],
            )
        except:
            continue
            
        count = count + 1
        combined_sentence = []
        for ind in range(ind1, ind2 + 1):
            combined_sentence.extend(sentences[ind])
            
        for index, word in enumerate(combined_sentence):
            if word.startswith(relations_[i][0]):
                combined_sentence[index] = (
                    combined_sentence[index].replace(relations_[i][0], "<e1>") + "</e1>"
                )
            if word.startswith(relations_[i][1]):
                combined_sentence[index] = (
                    combined_sentence[index].replace(relations_[i][1], "<e2>") + "</e2>"
                )
            combined_sentence[index] = re.sub(
                "^e[0-9]*-[0-9]*-", "", combined_sentence[index]
            )
            combined_sentence[index] = re.sub(
                "^tmx[0-9]*-[0-9]*-", "", combined_sentence[index]
            )
            
        data.append(
            [TreebankWordDetokenizer().detokenize(combined_sentence), relations_[i][2]]
        )
    
    return data


def predict_relations(text, model=None, tokenizer=None):
    """
    Predict temporal relations for input text.
    
    Args:
        text (str): Input text with temporal entity markers
        model: Trained temporal relation model  
        tokenizer: BERT tokenizer
        
    Returns:
        str: Predicted temporal relation label
    """
    if model is None or tokenizer is None:
        print("Model and tokenizer required for prediction")
        return "UNKNOWN"
    
    # Preprocess text
    processed_text = clean_str(text)
    
    # Add special tokens if not present
    if "[CLS]" not in processed_text:
        processed_text = "[CLS] " + processed_text + " [SEP]"
    
    # Tokenize
    tokens = tokenizer.tokenize(processed_text)
    input_ids = tokenizer.convert_tokens_to_ids(tokens)
    
    # Create attention mask
    attention_mask = [1] * len(input_ids)
    
    # Pad to max length if needed
    max_len = 512
    while len(input_ids) < max_len:
        input_ids.append(tokenizer.pad_token_id)
        attention_mask.append(0)
    
    # Convert to tensors
    input_ids = torch.tensor([input_ids])
    attention_mask = torch.tensor([attention_mask])
    
    # Find entity positions (simplified)
    e1_pos = [i for i, token in enumerate(tokens) if token in ["$", "<e1>"]]
    e2_pos = [i for i, token in enumerate(tokens) if token in ["#", "<e2>"]]
    
    if len(e1_pos) >= 2 and len(e2_pos) >= 2:
        event_ix = torch.tensor([[e1_pos[0], e2_pos[0]]])
    else:
        event_ix = torch.tensor([[1, 2]])  # Default positions
    
    # Predict
    model.eval()
    with torch.no_grad():
        # Note: This is a simplified prediction - full implementation would 
        # require graph construction and proper preprocessing
        outputs = model(input_ids, attention_mask, event_ix, [])
        logits = outputs[1] if len(outputs) > 1 else outputs[0]
        predicted_class = torch.argmax(logits, dim=1).item()
    
    # Map prediction to label
    label_map = {0: "BEFORE", 1: "AFTER", 2: "EQUAL", 3: "VAGUE"}
    return label_map.get(predicted_class, "UNKNOWN")


def create_edge_dict():
    """
    Create default edge dictionary for dependency parsing.
    
    Returns:
        dict: Dictionary mapping dependency relation types to IDs
    """
    return {
        "dep": 0,
        "nummod": 1,
        "punct": 2,
        "appos": 3,
        "nmod": 4,
        "advmod": 5,
        "advcl": 6,
        "compound": 7,
        "aux": 8,
        "prep": 9,
        "pobj": 10,
        "npadvmod": 11,
        "det": 12,
        "amod": 13,
        "dobj": 14,
        "relcl": 15,
        "nsubj": 16,
        "nsubjpass": 17,
        "auxpass": 18,
        "xcomp": 19,
        "conj": 20,
        "cc": 21,
        "case": 22,
        "poss": 23,
        "ccomp": 24,
        "attr": 25,
        "prt": 26,
        "mark": 27,
        "pcomp": 28,
        "quantmod": 29,
        "acl": 30,
        "oprd": 31,
        "predet": 32,
        "neg": 33,
        "expl": 34,
        "agent": 35,
        "acomp": 36,
        "meta": 37,
        "preconj": 38,
        "intj": 39,
        "csubj": 40,
        "dative": 41,
        "csubjpass": 42,
        "parataxis": 43,
    }


def setup_special_tokens(tokenizer):
    """
    Setup special tokens for entity marking.
    
    Args:
        tokenizer: BERT tokenizer to add special tokens to
        
    Returns:
        tuple: (cls_id, sep_id, pad_id, e1_id, e2_id)
    """
    special_tokens_dict = {"additional_special_tokens": ["$", "#"]}
    tokenizer.add_special_tokens(special_tokens_dict)
    
    cls_id = tokenizer.cls_token_id
    sep_id = tokenizer.sep_token_id
    pad_id = tokenizer.pad_token_id
    e1_id, e2_id = tokenizer.additional_special_tokens_ids
    
    return cls_id, sep_id, pad_id, e1_id, e2_id


def normalize_graph_nodes(graphs, max_nodes):
    """
    Normalize graph nodes to have the same number of nodes.
    
    Args:
        graphs (list): List of DGL graphs
        max_nodes (int): Maximum number of nodes to normalize to
        
    Returns:
        list: List of normalized graphs
    """
    for g in graphs:
        num_to_add = max_nodes - g.num_nodes()
        if num_to_add > 0:
            g.add_nodes(num_to_add)
            g.ndata["node_feat"][-num_to_add:] = 0
    return graphs


def compute_class_weights(labels):
    """
    Compute class weights for balanced training.
    
    Args:
        labels (list): List of class labels
        
    Returns:
        torch.Tensor: Class weights tensor
    """
    from sklearn.utils import class_weight
    
    unique_labels = np.unique(labels)
    class_weights = class_weight.compute_class_weight(
        class_weight="balanced",
        classes=unique_labels,
        y=labels
    )
    return torch.tensor(class_weights, dtype=torch.float)


def format_training_stats(epoch, train_loss, eval_metrics):
    """
    Format training statistics for display.
    
    Args:
        epoch (int): Current epoch number
        train_loss (float): Training loss
        eval_metrics (dict): Evaluation metrics
        
    Returns:
        str: Formatted statistics string
    """
    stats = f"Epoch {epoch}:\n"
    stats += f"  Train Loss: {train_loss:.4f}\n"
    
    if eval_metrics:
        for metric_name, value in eval_metrics.items():
            stats += f"  {metric_name.capitalize()}: {value:.4f}\n"
    
    return stats
