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


def prepareData(val_text, val_label, tokenizer, cls_id, sep_id, pad_id, e1_id, e2_id):
    """
    Prepare data for model input with special token handling.
    
    Args:
        val_text (list): List of text strings
        val_label (list): List of labels (can be None for prediction)
        tokenizer: BERT tokenizer
        cls_id (int): CLS token ID
        sep_id (int): SEP token ID  
        pad_id (int): PAD token ID
        e1_id (int): Entity 1 token ID
        e2_id (int): Entity 2 token ID
        
    Returns:
        tuple: (x_token, x_mark_index_all, labels)
    """
    labels = []
    x_token = []
    x_mark_index_all = []
    
    for cnt, item in enumerate(val_text):
        temp = tokenizer.encode(
            item,
            add_special_tokens=False,
            max_length=512,
            truncation=True,
            padding="max_length",
        )
        while len(temp) < 512:
            temp.append(pad_id)
            
        temp_cup = list(enumerate(temp))
        cls_index = [index for index, value in temp_cup if value == cls_id]
        cls_index.append(0)
        e1_index = [index for index, value in temp_cup if value == e1_id]
        e2_index = [index for index, value in temp_cup if value == e2_id]
        sep_index = [index for index, value in temp_cup if value == sep_id]
        
        if len(e1_index) != 2 or len(e2_index) != 2:
            continue
            
        sep_index.append(0)
        x_mark_index = []
        x_mark_index.append(cls_index)
        x_mark_index.append(e1_index)
        x_mark_index.append(e2_index)
        x_mark_index.append(sep_index)
        x_mark_index_all.append(x_mark_index)
        x_token.append(temp)
        
        if val_label is not None:
            labels.append(val_label[cnt])
            
    return (x_token, x_mark_index_all, labels)


def predict_relations(text, model, tokenizer, label2class=None):
    """
    Predict temporal relations for input text using the actual working prediction code.
    
    Args:
        text (str): Input text with temporal entity markers ($ and #)
        model: Trained temporal relation model  
        tokenizer: BERT tokenizer with special tokens configured
        label2class (dict): Mapping from class indices to label names
        
    Returns:
        str: Predicted temporal relation label
    """
    if model is None or tokenizer is None:
        print("Model and tokenizer required for prediction")
        return "UNKNOWN"
    
    # Default label mapping if not provided
    if label2class is None:
        label2class = {
            0: "BEFORE",
            1: "AFTER", 
            2: "EQUAL",
            3: "VAGUE"
        }
    
    # Get special token IDs
    cls_id, sep_id, pad_id, e1_id, e2_id = setup_special_tokens(tokenizer)
    
    # Prepare the sentence as a list (matching original format)
    sentence = [text]
    
    try:
        # Use the original prepareData function
        tx_token, tx_mark_index_all, tlabels1 = prepareData(
            sentence, None, tokenizer, cls_id, sep_id, pad_id, e1_id, e2_id
        )
        
        if not tx_token:  # If prepareData returns empty (entity markers not found properly)
            return "UNKNOWN"
        
        # Convert to numpy and torch tensors (matching original code exactly)
        tx_token = np.vstack(tx_token).astype(float)
        tx_token = np.array(tx_token)
        tx_token = torch.from_numpy(tx_token)
        
        # Set model to evaluation mode
        model.eval()
        
        # Make prediction (matching original code exactly)
        with torch.no_grad():
            out = model(tx_token.clone().detach().requires_grad_(True).long(), tx_mark_index_all)
            
        # Get predicted class and convert to label (matching original code)
        predicted_class = torch.max(out[1], 1)[1].item()
        predicted_label = label2class[predicted_class]
        
        return predicted_label
        
    except Exception as e:
        print(f"Error during prediction: {e}")
        return "UNKNOWN"


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
