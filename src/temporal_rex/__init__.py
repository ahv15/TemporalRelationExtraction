"""
Temporal Relation Extraction Package

A deep learning framework for extracting temporal relations from text using
BERT-based models with Graph Attention Networks.
"""

__version__ = "0.1.0"
__author__ = "Harshit"

from .model import TempRelModel, TempRel2Model, GAT, CustomTrainer
from .data import temprel_ee, temprel_set, TempDataset, LabelType
from .eval import calc_f1, compute_metrics
from .utils import clean_str, load_data_and_labels, save, predict_relations

__all__ = [
    "TempRelModel",
    "TempRel2Model", 
    "GAT",
    "CustomTrainer",
    "temprel_ee",
    "temprel_set",
    "TempDataset",
    "LabelType",
    "calc_f1",
    "compute_metrics",
    "clean_str",
    "load_data_and_labels",
    "save",
    "predict_relations"
]
