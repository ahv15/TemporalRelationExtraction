#!/usr/bin/env python3
"""
Example usage of the Temporal Relation Extraction framework.

This script demonstrates how to use the refactored temporal_rex package
for predicting temporal relations between events in text.
"""

import sys
import os

# Add the src directory to the Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from temporal_rex import predict_relations
from temporal_rex.data import LabelType, class2label, label2class
from temporal_rex.utils import clean_str, setup_special_tokens
from transformers import AutoTokenizer


def main():
    """Main example function."""
    print("Temporal Relation Extraction - Example Usage")
    print("=" * 50)
    
    # Initialize tokenizer
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
    
    # Setup special tokens for entity marking
    cls_id, sep_id, pad_id, e1_id, e2_id = setup_special_tokens(tokenizer)
    print(f"Special tokens configured: CLS={cls_id}, SEP={sep_id}, E1={e1_id}, E2={e2_id}")
    
    # Example texts with temporal entities marked
    examples = [
        {
            "text": "The patient went into a $ coma $ after he had a # heart attack #.",
            "expected": "AFTER",
            "description": "Medical event sequence"
        },
        {
            "text": "The $ meeting $ was scheduled before the # deadline #.",
            "expected": "BEFORE", 
            "description": "Business event ordering"
        },
        {
            "text": "The $ earthquake $ and the # tsunami # occurred simultaneously.",
            "expected": "EQUAL",
            "description": "Simultaneous natural events"
        },
        {
            "text": "The $ announcement $ was made around the time of the # election #.",
            "expected": "VAGUE",
            "description": "Vague temporal relation"
        }
    ]
    
    print("\nRunning examples...")
    print("-" * 30)
    
    for i, example in enumerate(examples, 1):
        print(f"\nExample {i}: {example['description']}")
        print(f"Text: {example['text']}")
        print(f"Expected: {example['expected']}")
        
        # Clean the text
        cleaned_text = clean_str(example['text'])
        print(f"Cleaned: {cleaned_text}")
        
        # Predict relation (using placeholder function since we don't have a trained model)
        predicted = predict_relations(example['text'])
        print(f"Predicted: {predicted}")
        
        # Check if prediction matches expectation
        match_status = "✓" if predicted == example['expected'] else "✗"
        print(f"Status: {match_status}")
    
    # Demonstrate label mapping
    print("\n" + "=" * 50)
    print("Label Mappings:")
    print("-" * 20)
    print("Class to Label:")
    for class_id, label in label2class.items():
        print(f"  {class_id}: {label}")
    
    print("\nLabel to Class:")
    for label, class_id in class2label.items():
        print(f"  {label}: {class_id}")
    
    # Demonstrate LabelType enum
    print("\nLabelType Enum:")
    for label_type in LabelType:
        print(f"  {label_type.name}: {label_type.value}")
    
    print("\n" + "=" * 50)
    print("Example completed!")
    print("\nTo train your own model:")
    print("1. Prepare your dataset in XML format")
    print("2. Use the model.py module to define and train models")
    print("3. Use eval.py to evaluate performance")
    print("4. Use utils.py for preprocessing and graph generation")


if __name__ == "__main__":
    main()
