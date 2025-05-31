# Temporal Relation Extraction

**A deep learning framework for extracting temporal relations from text using BERT-based models with Graph Attention Networks.**

## Why It Matters

Temporal relation extraction is crucial for understanding the temporal structure of events in text, enabling applications in:
- **Information Retrieval**: Better search and document understanding
- **Question Answering**: Temporal reasoning over facts and events  
- **Medical NLP**: Understanding disease progression and treatment timelines
- **Legal Tech**: Analyzing case timelines and regulatory compliance
- **Business Intelligence**: Event sequencing in reports and communications

This framework achieves state-of-the-art performance by combining BERT's contextual understanding with Graph Attention Networks that model syntactic dependencies, making it valuable for any organization processing temporal information at scale.

## Installation

Clone the repository and install dependencies:

```bash
git clone https://github.com/ahv15/TemporalRelationExtraction.git
cd TemporalRelationExtraction
pip install -r requirements.txt
```

### Additional Setup

Download required spaCy model:
```bash
python -m spacy download en_core_web_sm
```

## Quick Start

```python
from src.temporal_rex import predict_relations
from transformers import AutoTokenizer

# Initialize tokenizer
tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
config = AutoConfig.from_pretrained("distilbert-base-uncased")
model = TempRelModel(config, tokenizer, num_labels=4, 
                     edge_dict={}, max_nodes=100)

# Example text with temporal entities marked
text = "The patient went into a $ coma $ after he had a # heart attack #."

# Predict temporal relation
relation = predict_relations(text, model, tokenizer)
print(f"Temporal relation: {relation}")
# Output: "Temporal relation: AFTER"
```

## Project Structure

```
TemporalRelationExtraction/
├── src/temporal_rex/           # Main package
│   ├── __init__.py            # Package initialization
│   ├── data.py                # Data loading and preprocessing
│   ├── model.py               # Model definitions (BERT + GAT)
│   ├── eval.py                # Evaluation metrics and utilities
│   └── utils.py               # Helper functions and utilities
├── requirements.txt           # Project dependencies
└── README.md                 # This file
```

### Module Overview

- **`data.py`**: Dataset classes for MATRES and TempEval-3, XML parsing, data preprocessing
- **`model.py`**: TempRelModel and TempRel2Model architectures, Graph Attention Networks, custom training loops
- **`eval.py`**: Comprehensive evaluation metrics including F1, precision, recall, confusion matrices
- **`utils.py`**: Graph generation, text processing, model saving/loading, prediction utilities

## Sample Input/Output

### Input Format
Text with temporal entities marked using special tokens:
```python
# Entity marking format
text = "The $ surgery $ was performed before the # diagnosis #."

# Alternative format  
text = "The <e1>surgery</e1> was performed before the <e2>diagnosis</e2>."
```

### Output Format
Predicted temporal relations:
```python
{
    "relation": "BEFORE",     # Primary prediction
    "confidence": 0.94,       # Model confidence
    "all_probabilities": {    # Full probability distribution
        "BEFORE": 0.94,
        "AFTER": 0.03,
        "EQUAL": 0.02,
        "VAGUE": 0.01
    }
}
```

### Supported Relations
- **BEFORE**: Event A happens before Event B
- **AFTER**: Event A happens after Event B  
- **EQUAL/SIMULTANEOUS**: Events occur at the same time
- **VAGUE**: Temporal relationship is unclear

## Training Your Own Model

```python
from src.temporal_rex.model import TempRelModel, train
from src.temporal_rex.data import temprel_set
from transformers import AutoTokenizer, AutoConfig

# Load data
tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
trainset = temprel_set("data/trainset-temprel.xml")
train_data, _ = trainset.to_tensor(tokenizer)

# Initialize model
config = AutoConfig.from_pretrained("distilbert-base-uncased")
model = TempRelModel(config, tokenizer, num_labels=4, 
                     edge_dict={}, max_nodes=100)

# Train model
train(model, train_dataloader, dev_dataloader, test_dataloader,
      total_train_graphs, total_test_graphs, device="cpu", n_gpu=0)
```

## Evaluation

```python
from src.temporal_rex.eval import evaluate_model, print_evaluation_results

# Evaluate trained model
results = evaluate_model(model, test_dataloader, device="cpu")
print_evaluation_results(results, "Test Set")
```

## Advanced Usage

### Custom Dataset Processing

```python
from src.temporal_rex.data import TempDataset, load_data_and_labels

# Load your own data
x_text, y_labels, max_len = load_data_and_labels(your_data)
dataset = TempDataset(x_text, y_labels)
```

### Graph-based Features

```python
from src.temporal_rex.utils import generateDependencyGraph

# Generate dependency graphs for enhanced features
edge_dict = {}
graphs = generateDependencyGraph(edge_dict, texts, 0, [])
```

### Model Variants

```python
# For MATRES dataset
model = TempRelModel(config, tokenizer, num_labels=4, 
                     edge_dict=edge_dict, max_nodes=max_nodes)

# For TempEval-3 dataset  
model = TempRel2Model(config, tokenizer, num_labels=len(unique_labels),
                      edge_dict=edge_dict, max_nodes=max_nodes)
```

## Datasets

The framework supports multiple temporal relation datasets:

- **MATRES**: Multi-Axis Temporal RElations for Start-points dataset
- **TempEval-3**: Temporal evaluation dataset from SemEval-2013
- **TE3-Silver**: Silver-standard annotations for temporal relations

Expected data format:
```xml
<temprel LABEL="BEFORE" SENTDIFF="0" DOCID="doc1" SOURCE="e1" TARGET="e2">
The/DT///O patient/NN///O went/VBD///E1 into/IN///O coma/NN///E2
</temprel>
```
