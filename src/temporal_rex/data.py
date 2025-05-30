"""
Data loading and preprocessing utilities for temporal relation extraction.

This module contains classes and functions for handling temporal relation datasets,
including XML parsing, data preprocessing, and dataset creation.
"""

import xml.etree.ElementTree as ET
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, TensorDataset
from enum import Enum
import re
import glob
import random
import os


class LabelType(Enum):
    """Enumeration for temporal relation labels."""
    BEFORE = 0
    AFTER = 1
    EQUAL = 2
    VAGUE = 3

    @staticmethod
    def to_class_index(label_type):
        """Convert label string to class index."""
        for label in LabelType:
            if label_type == label.name:
                return label.value


class temprel_ee:
    """Class for handling temporal relation event entities from XML."""
    
    def __init__(self, xml_element):
        self.xml_element = xml_element
        self.label = xml_element.attrib["LABEL"]
        self.sentdiff = int(xml_element.attrib["SENTDIFF"])
        self.docid = xml_element.attrib["DOCID"]
        self.source = xml_element.attrib["SOURCE"]
        self.target = xml_element.attrib["TARGET"]
        self.data = xml_element.text.strip().split()
        self.token = []
        self.lemma = []
        self.part_of_speech = []
        self.position = []
        self.length = len(self.data)
        self.event_ix = []
        self.text = ""
        self.event_offset = []

        is_start = True
        for i, d in enumerate(self.data):
            tmp = d.split("///")
            self.part_of_speech.append(tmp[-2])
            self.position.append(tmp[-1])
            self.token.append(tmp[0])
            self.lemma.append(tmp[1])

            if is_start:
                is_start = False
            else:
                self.text += " "

            if tmp[-1] == "E1":
                self.event_ix.append(i)
                self.event_offset.append(len(self.text))
            elif tmp[-1] == "E2":
                self.event_ix.append(i)
                self.event_offset.append(len(self.text))

            self.text += tmp[0]

        assert len(self.event_ix) == 2


class temprel_set:
    """Class for handling temporal relation datasets from XML files."""
    
    def __init__(self, xmlfname, datasetname="matres"):
        self.xmlfname = xmlfname
        self.datasetname = datasetname
        tree = ET.parse(xmlfname)
        root = tree.getroot()
        self.size = len(root)
        self.temprel_ee = []
        for e in root:
            self.temprel_ee.append(temprel_ee(e))

    def to_tensor(self, tokenizer):
        """Convert dataset to tensor format for training."""
        gathered_text = [ee.text for ee in self.temprel_ee]
        tokenized_output = tokenizer(
            gathered_text, padding=True, return_offsets_mapping=True
        )
        tokenized_event_ix = []
        for i in range(len(self.temprel_ee)):
            event_ix_pair = []
            for j, offset_pair in enumerate(tokenized_output["offset_mapping"][i]):
                if (
                    offset_pair[0] == self.temprel_ee[i].event_offset[0]
                    or offset_pair[0] == self.temprel_ee[i].event_offset[1]
                ) and offset_pair[0] != offset_pair[1]:
                    event_ix_pair.append(j)
            if len(event_ix_pair) != 2:
                raise ValueError(f"Instance {i} doesn't found 2 event idx.")
            tokenized_event_ix.append(event_ix_pair)
        input_ids = torch.LongTensor(tokenized_output["input_ids"])
        attention_mask = torch.LongTensor(tokenized_output["attention_mask"])
        tokenized_event_ix = torch.LongTensor(tokenized_event_ix)
        labels = torch.LongTensor(
            [LabelType.to_class_index(ee.label) for ee in self.temprel_ee]
        )
        return (
            TensorDataset(input_ids, attention_mask, tokenized_event_ix, labels),
            gathered_text,
        )


class TempDataset(Dataset):
    """PyTorch Dataset class for temporal relation data."""
    
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, item):
        return self.data[item], self.labels[item]


def clean_str(text):
    """Clean and preprocess text strings."""
    text = text.lower()
    return text.strip()


def load_data_and_labels(data):
    """Load data and labels from processed dataset."""
    data_all = []
    max_sentence_length = 0
    for idx, line in enumerate(data):
        id = idx
        relation = line[1]
        sentence = line[0]
        sentence = sentence.replace("#", "")
        sentence = sentence.replace("$", "")

        sentence = sentence.replace("<e1>", " $ ")
        sentence = sentence.replace("</e1>", " $ ")
        sentence = sentence.replace("<e2>", " # ")
        sentence = sentence.replace("</e2>", " # ")

        sentence = clean_str(sentence)
        sentence = "[CLS] " + sentence + " [SEP]"
        tokens = tokenizer.tokenize(sentence)
        if max_sentence_length < len(tokens):
            max_sentence_length = len(tokens)

        data_all.append([id, sentence, relation])

    df = pd.DataFrame(data=data_all, columns=["id", "sentence", "relation"])
    df["label"] = [class2label[r] for r in df["relation"]]
    x_text = df["sentence"].tolist()
    y = df["label"].tolist()

    x_text = np.array(x_text)
    y = np.array(y)

    return x_text, y, max_sentence_length


def tokenize_and_align_labels(examples):
    """Tokenize inputs and align labels for token classification."""
    tokenized_inputs = tokenizer(
        examples["tokens"], truncation=True, is_split_into_words=True
    )
    labels = []
    word_ids = tokenized_inputs.word_ids()
    previous_word_idx = None
    label_ids = []
    for i, label in enumerate(examples[f"tags"]):
        word_ids = tokenized_inputs.word_ids(batch_index=i)
        previous_word_idx = None
        label_ids = []
        for word_idx in word_ids:
            if word_idx is None:
                label_ids.append(-100)
            elif word_idx != previous_word_idx:
                label_ids.append(label[word_idx])
            else:
                label_ids.append(label[word_idx] if label_all_tokens else -100)
            previous_word_idx = word_idx

        labels.append(label_ids)

    tokenized_inputs["labels"] = labels
    return tokenized_inputs


def _get_tensorset(tokenizer):
    """Load and prepare tensor datasets for training."""
    traindevset = temprel_set("data/trainset-temprel.xml")
    traindev_tensorset, dataset = traindevset.to_tensor(tokenizer=tokenizer)
    train_idx = list(range(len(traindev_tensorset) - 1852))
    train_text = [dataset[i] for i in train_idx]
    dev_idx = list(range(len(traindev_tensorset) - 1852, len(traindev_tensorset)))
    dev_text = [dataset[i] for i in dev_idx]
    train_tensorset = Subset(traindev_tensorset, train_idx)
    dev_tensorset = Subset(traindev_tensorset, dev_idx)  # Last 21 docs
    print(
        f"All = {len(traindev_tensorset)}, Train={len(train_tensorset)}, Dev={len(dev_tensorset)}"
    )

    testset = temprel_set("data/testset-temprel.xml")
    test_tensorset, test_text = testset.to_tensor(tokenizer=tokenizer)
    print(f"Test = {len(test_tensorset)}")
    return (
        train_tensorset,
        dev_tensorset,
        test_tensorset,
        train_text,
        dev_text,
        test_text,
    )


def prepareData(val_text, val_label):
    """Prepare data for model input with special token handling."""
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
        if val_label != None:
            labels.append(val_label[cnt])
    return (x_token, x_mark_index_all, labels)


def inputCol(count):
    """Process temporal evaluation data from COL files."""
    input = ""
    allFiles = glob.glob("TE3-Silver-data-col/*.col")

    output = []
    text = []
    relations_all = []
    int_count = int(count)

    if count > 0:
        randomFiles = random.sample(allFiles, int_count)
    else:
        randomFiles = allFiles

    for num, colFile in enumerate(randomFiles):
        colFile_name = nameFile(colFile)
        with open(colFile) as f:
            try:
                next(f)
            except:
                print("Check")
                continue
            input = ""
            for ind, line in enumerate(f):
                columns = line.split("\t")
                if len(columns) >= 2:
                    relations = columns[17].split("||")
                    if relations[0] != "O":
                        for i in range(len(relations)):
                            relations[i] = relations[i].split(":")[:-1]
                            relations[i][0] += "-" + str(num) + "-"
                            relations[i][1] += "-" + str(num) + "-"
                        relations_all.extend(relations)
                        for relation in relations_all:
                            if relation[1].startswith("tmx0"):
                                relations_all.remove(relation)

                    input = (
                        input
                        + columns[0]
                        + "\t"
                        + parseEntity(columns[3], columns[11])
                        + "\n"
                    )
                    if parseEntity(columns[3], columns[11]) == "EVENT":
                        text.append(columns[3] + "-" + str(num) + "-" + columns[0])
                    elif parseEntity(columns[3], columns[11]) == "TIMEX3":
                        text.append(columns[11] + "-" + str(num) + "-" + columns[0])
                    else:
                        text.append(columns[0])
        filename = "/TE3-Silver-data/" + colFile_name
        output.append(filename)
        if not os.path.exists(os.path.dirname(filename)):
            try:
                os.makedirs(os.path.dirname(filename))
            except OSError as exc:
                if exc.errno != errno.EEXIST:
                    raise
        with open(filename, "w") as file:
            file.write(input)
            file.write(input)
            file.close()

    return (output, text, relations_all)


def nameFile(file):
    """Extract filename from file path."""
    file_name = file.split("/")
    return file_name[-1]


def parseEntity(event, time):
    """Parse entity type from event and time strings."""
    if event[0] == "e":
        c = "EVENT"
    elif time[0] == "t":
        c = "TIMEX3"
    else:
        c = "OTHERS"
    return c


# Define label mappings
class2label = {
    "BEFORE": 0,
    "IBEFORE": 7,
    "AFTER": 1,
    "IS_INCLUDED": 3,
    "SIMULTANEOUS": 2,
    "INCLUDES": 4,
    "IAFTER": 5,
    "BEGUN_BY": 6,
    "BEGINS": 8,
}

label2class = {
    0: "BEFORE",
    7: "IBEFORE",
    1: "AFTER",
    3: "IS_INCLUDED",
    2: "SIMULTANEOUS",
    4: "INCLUDES",
    5: "IAFTER",
    6: "BEGUN_BY",
    8: "BEGINS",
}
