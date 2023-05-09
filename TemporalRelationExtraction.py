# -*- coding: utf-8 -*-


from datasets import load_metric
from nltk.tokenize.treebank import TreebankWordDetokenizer
from nltk.tokenize import sent_tokenize
from nltk.tokenize import word_tokenize
from textattack.augmentation import EmbeddingAugmenter
from textattack.augmentation import WordNetAugmenter
from pysyntime import SynTime
from datasets import Dataset
from transformers import AutoTokenizer
from sklearn import preprocessing
from transformers import AutoModelForTokenClassification, TrainingArguments, Trainer
from torch import nn
from sklearn.utils import class_weight
from transformers import DataCollatorForTokenClassification
from enum import Enum
from torch.utils.data import TensorDataset
from math import ceil
from transformers import AdamW, get_linear_schedule_with_warmup
from tqdm import tqdm, trange
from torch.utils.data import DataLoader, Subset
from transformers import AutoTokenizer
from transformers import AutoConfig
from torch.utils.data import Dataset, DataLoader, TensorDataset, Subset
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from transformers import AdamW, get_linear_schedule_with_warmup
from tqdm import tqdm, trange
import torch.nn.functional as Fun
import torch
import xml.etree.ElementTree as ET
import torch
import torch.nn as nn
from transformers import AutoModel, BertPreTrainedModel
import spacy
import networkx as nx
import dgl
from dgl.nn import GATConv
import os
import torch
import torch.nn as nn
import time
import logging
import torch
import torch.nn as nn
import torch.optim as optim
import re
import pandas as pd
import numpy as np
import numpy as np
import torch
import nltk
import glob
import random
import os


nltk.download("punkt")


class CustomTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.get("labels")
        outputs = model(**inputs)
        logits = outputs.get("logits")
        loss_fct = nn.CrossEntropyLoss(weight=torch.tensor(class_weights).float())
        loss = loss_fct(logits.view(-1, self.model.config.num_labels), labels.view(-1))
        return (loss, outputs) if return_outputs else loss


class GAT(nn.Module):
    def __init__(
        self, in_feats, hidden_size, out_feats, num_heads, edge_types, no_nodes
    ):
        super(GAT, self).__init__()

        self.conv_layers = nn.ModuleList()
        self.edge_type2id = {}
        for et in edge_types:
            if et not in self.edge_type2id:
                self.edge_type2id[et] = len(self.edge_type2id)
            self.conv_layers.append(
                GATConv(
                    in_feats=in_feats,
                    out_feats=hidden_size,
                    num_heads=num_heads,
                    feat_drop=0.5,
                    attn_drop=0.5,
                )
            )

        self.out_layer = nn.Linear(
            hidden_size * num_heads * len(edge_types) * no_nodes, out_feats
        )

    def forward(self, g):
        out_list = []
        for et in self.edge_type2id:
            id = self.edge_type2id[et]
            et_edges = g.edata["rel"][:, id] == 1
            et_g = dgl.edge_subgraph(g, et_edges, relabel_nodes=False)
            et_g = dgl.add_self_loop(et_g)
            G = dgl.to_networkx(et_g)
            et_out = self.conv_layers[id](et_g, et_g.ndata["node_feat"])
            out_list.append(et_out)
        h = torch.flatten(torch.cat(out_list, dim=0))
        return self.out_layer(h)


# Used on MATRES Dataset
class TempRelModel(BertPreTrainedModel):
    def __init__(
        self,
        config,
        tokenizer,
        num_labels,
        edge_dict,
        max_nodes,
        pretrained_weights="distilbert-base-uncased",
    ):
        super(TempRelModel, self).__init__(config)
        self.num_labels = num_labels

        self.tokenizer = tokenizer
        self.bertModel = AutoModel.from_pretrained(pretrained_weights, config=config)

        d = config.hidden_size
        self.entity_dense = nn.Linear(d, d)
        self.CLS_dense = nn.Linear(d, d)
        self.ABC_dense = nn.Linear(d, d)
        self.all_dense = nn.Linear(d * 6 + 32, self.num_labels)

        self.entity_dense1 = nn.Linear(d, d)
        self.CLS_dense1 = nn.Linear(d, d)
        self.ABC_dense1 = nn.Linear(d, d)
        self.relative_time = nn.Linear(d * 4, 1)

        self.graph_layer = GAT(
            in_feats=96,
            hidden_size=64,
            out_feats=32,
            num_heads=16,
            edge_types=edge_dict,
            no_nodes=max_nodes,
        )

        nn.init.xavier_normal_(self.entity_dense.weight)
        nn.init.constant_(self.entity_dense.bias, 0.0)
        nn.init.xavier_normal_(self.CLS_dense.weight)
        nn.init.constant_(self.CLS_dense.bias, 0.0)
        nn.init.xavier_normal_(self.all_dense.weight)
        nn.init.constant_(self.all_dense.bias, 0)
        nn.init.xavier_normal_(self.ABC_dense.weight)
        nn.init.constant_(self.ABC_dense.bias, 0)

        self.dropout = nn.Dropout(0.1)

    def forward(
        self, input_ids, attention_mask, event_ix, graph, labels=None, output_time=False
    ):
        bertresult = self.bertModel(input_ids, attention_mask=attention_mask)
        sequence_output = bertresult[0]
        batch_size, sequence_length, hidden_size = sequence_output.size()

        event_1_ix, event_2_ix = event_ix.split(1, dim=-1)

        e1_result = []
        e2_result = []
        A_result = []
        B_result = []
        C_result = []
        graph_out = []
        e1_result = torch.gather(
            sequence_output,
            dim=1,
            index=event_1_ix.expand(batch_size, hidden_size).unsqueeze(dim=1),
        ).squeeze(dim=1)

        e2_result = torch.gather(
            sequence_output,
            dim=1,
            index=event_2_ix.expand(batch_size, hidden_size).unsqueeze(dim=1),
        ).squeeze(dim=1)
        for i in range(batch_size):
            a_vector = sequence_output[i, 0 : event_1_ix[i][0], :]
            if a_vector.size()[0] != 0:
                A_result.append(torch.mean(a_vector, dim=0, keepdim=True))
            else:
                A_result.append(
                    torch.mean(
                        sequence_output[i, 0 : event_1_ix[i][0], :], dim=0, keepdim=True
                    )
                )

            b_vector = sequence_output[i, event_1_ix[i][0] + 1 : event_2_ix[i][0], :]
            if b_vector.size()[0] != 0:
                B_result.append(
                    torch.squeeze(torch.mean(b_vector, dim=0, keepdim=True))
                )
            else:
                B = (e1_result[i] + e2_result[i]) / 2
                B_result.append(torch.squeeze(B))

            c_vector = sequence_output[i, event_2_ix[i][0] + 1 :, :]
            if c_vector.size()[0] != 0:
                C_result.append(torch.mean(c_vector, dim=0, keepdim=True))
            else:
                C_result.append(
                    torch.mean(sequence_output[i, -1, :], dim=0, keepdim=True)
                )

            graph_out.append(self.graph_layer(graph[i]))

        H_clr = sequence_output[:, 0]
        H_A = torch.stack(A_result, 0).squeeze(dim=1)
        H_B = torch.stack(B_result, 0).squeeze(dim=1)
        H_C = torch.stack(C_result, 0).squeeze(dim=1)
        H_Graph = torch.stack(graph_out)
        cls_dense1 = self.CLS_dense1(self.dropout(torch.tanh(H_clr)))
        e1_dense1 = self.entity_dense1(self.dropout(torch.tanh(e1_result)))
        e2_dense1 = self.entity_dense1(self.dropout(torch.tanh(e2_result)))
        A_dense1 = self.ABC_dense1(self.dropout(torch.tanh(H_A)))
        B_dense1 = self.ABC_dense1(self.dropout(torch.tanh(H_B)))
        C_dense1 = self.ABC_dense1(self.dropout(torch.tanh(H_C)))

        E1_Time = torch.tanh(
            self.relative_time(
                torch.cat((cls_dense1, A_dense1, e1_dense1, B_dense1), 1)
            )
        )
        E2_Time = torch.tanh(
            self.relative_time(
                torch.cat((cls_dense1, B_dense1, e2_dense1, C_dense1), 1)
            )
        )
        loss = 0
        if labels != None:
            relative = E1_Time - E2_Time
            mask_before = (labels == 0).float()
            relative_sum_before = ((1 + relative) > 0).float() * (1 + relative)
            loss += torch.sum(relative_sum_before * mask_before)
            mask_after = (labels == 1).float()
            relative_sum_after = ((1 - relative) > 0).float() * (1 - relative)
            loss += torch.sum(relative_sum_after * mask_after)
            mask_equal = (labels == 2).float()
            loss += torch.sum(torch.abs(relative * mask_equal))
            loss /= batch_size

        cls_dense = self.CLS_dense(self.dropout(torch.tanh(H_clr)))
        e1_dense = self.entity_dense(self.dropout(torch.tanh(e1_result)))
        e2_dense = self.entity_dense(self.dropout(torch.tanh(e2_result)))
        A_dense = self.ABC_dense(self.dropout(torch.tanh(H_A)))
        B_dense = self.ABC_dense(self.dropout(torch.tanh(H_B)))
        C_dense = self.ABC_dense(self.dropout(torch.tanh(H_C)))

        cat_result = torch.cat(
            (cls_dense, A_dense, e1_dense, B_dense, e2_dense, C_dense, H_Graph), 1
        )

        result = self.all_dense(cat_result)
        loss_fct = nn.CrossEntropyLoss()
        if labels != None:
            loss += loss_fct(result, labels)
        return [loss, result]


# Used on TempEval-3 Dataset
class TempRel2Model(BertPreTrainedModel):
    def __init__(
        self,
        config,
        tokenizer,
        num_labels,
        edge_dict,
        max_nodes,
        pretrained_weights="distilbert-base-uncased",
    ):
        super(TempRel2Model, self).__init__(config)
        self.num_labels = num_labels

        self.tokenizer = tokenizer
        self.bertModel = AutoModel.from_pretrained(pretrained_weights, config=config)

        d = config.hidden_size
        self.entity_dense = nn.Linear(d, d)
        self.CLS_dense = nn.Linear(d, d)
        self.ABC_dense = nn.Linear(d, d)
        self.all_dense = nn.Linear(d * 6, self.num_labels)

        self.entity_dense1 = nn.Linear(d, d)
        self.CLS_dense1 = nn.Linear(d, d)
        self.ABC_dense1 = nn.Linear(d, d)
        self.relative_time = nn.Linear(d * 4, 1)

        self.graph_layer = GAT(
            in_feats=96,
            hidden_size=32,
            out_feats=16,
            num_heads=8,
            edge_types=edge_dict,
            no_nodes=max_nodes,
        )

        nn.init.xavier_normal_(self.entity_dense.weight)
        nn.init.constant_(self.entity_dense.bias, 0.0)
        nn.init.xavier_normal_(self.CLS_dense.weight)
        nn.init.constant_(self.CLS_dense.bias, 0.0)
        nn.init.xavier_normal_(self.all_dense.weight)
        nn.init.constant_(self.all_dense.bias, 0)
        nn.init.xavier_normal_(self.ABC_dense.weight)
        nn.init.constant_(self.ABC_dense.bias, 0)

        self.dropout = nn.Dropout(0.1)

    def forward(self, x, x_mark_index_all, graph, labels=None):
        bertresult = self.bertModel(x)
        bertresult = bertresult[0]

        batch_size = x.size()[0]

        e1_result = []
        e2_result = []
        A_result = []
        B_result = []
        C_result = []
        graph_out = []
        for i in range(batch_size):
            cls = x_mark_index_all[i][0]
            e1 = x_mark_index_all[i][1]
            e2 = x_mark_index_all[i][2]
            sep = x_mark_index_all[i][3]

            entity1 = torch.mean(
                bertresult[i, e1[0] + 1 : e1[1], :], dim=0, keepdim=True
            )
            e1_result.append(entity1)

            entity2 = torch.mean(
                bertresult[i, e2[0] + 1 : e2[1], :], dim=0, keepdim=True
            )
            e2_result.append(entity2)

            a_vector = bertresult[i, cls[0] + 1 : e1[0], :]
            if a_vector.size()[0] != 0:
                A_result.append(torch.mean(a_vector, dim=0, keepdim=True))
            else:
                A_result.append(
                    torch.mean(bertresult[i, cls[0] : e1[0], :], dim=0, keepdim=True)
                )

            b_vector = bertresult[i, e1[1] + 1 : e2[0], :]
            if b_vector.size()[0] != 0:
                B_result.append(torch.mean(b_vector, dim=0, keepdim=True))
            else:
                B = (entity1 + entity2) / 2
                B_result.append(B)

            c_vector = bertresult[i, e2[1] + 1 : sep[0], :]
            if c_vector.size()[0] != 0:
                C_result.append(torch.mean(c_vector, dim=0, keepdim=True))
            else:
                C_result.append(
                    torch.mean(
                        bertresult[i, sep[0] : sep[0] + 1, :], dim=0, keepdim=True
                    )
                )

            graph_out.append(self.graph_layer(graph[i]))

        H_clr = bertresult[:, 0]
        H_e1 = torch.cat(e1_result, 0)
        H_e2 = torch.cat(e2_result, 0)
        H_A = torch.cat(A_result, 0)
        H_B = torch.cat(B_result, 0)
        H_C = torch.cat(C_result, 0)
        H_Graph = torch.stack(graph_out)

        cls_dense1 = self.CLS_dense1(self.dropout(torch.tanh(H_clr)))
        e1_dense1 = self.entity_dense1(self.dropout(torch.tanh(H_e1)))
        e2_dense1 = self.entity_dense1(self.dropout(torch.tanh(H_e2)))
        A_dense1 = self.ABC_dense1(self.dropout(torch.tanh(H_A)))
        B_dense1 = self.ABC_dense1(self.dropout(torch.tanh(H_B)))
        C_dense1 = self.ABC_dense1(self.dropout(torch.tanh(H_C)))

        E1_Time = torch.tanh(
            self.relative_time(
                torch.cat((cls_dense1, A_dense1, e1_dense1, B_dense1), 1)
            )
        )
        E2_Time = torch.tanh(
            self.relative_time(
                torch.cat((cls_dense1, B_dense1, e2_dense1, C_dense1), 1)
            )
        )
        loss = 0
        if labels != None:
            relative = E1_Time - E2_Time
            mask_before = (labels == 0).float()
            relative_sum_before = ((1 + relative) > 0).float() * (1 + relative)
            loss += torch.sum(relative_sum_before * mask_before)
            mask_after = (labels == 1).float()
            relative_sum_after = ((1 - relative) > 0).float() * (1 - relative)
            loss += torch.sum(relative_sum_after * mask_after)
            mask_equal = (labels == 2).float()
            loss += torch.sum(torch.abs(relative * mask_equal))
            loss /= batch_size

        cls_dense = self.CLS_dense(self.dropout(torch.tanh(H_clr)))
        e1_dense = self.entity_dense(self.dropout(torch.tanh(H_e1)))
        e2_dense = self.entity_dense(self.dropout(torch.tanh(H_e2)))
        A_dense = self.ABC_dense(self.dropout(torch.tanh(H_A)))
        B_dense = self.ABC_dense(self.dropout(torch.tanh(H_B)))
        C_dense = self.ABC_dense(self.dropout(torch.tanh(H_C)))

        cat_result = torch.cat(
            (cls_dense, A_dense, e1_dense, B_dense, e2_dense, C_dense), 1
        )

        result = self.all_dense(cat_result)
        loss_fct = nn.CrossEntropyLoss(weight=torch.tensor(class_weights).float())
        if labels != None:
            loss += loss_fct(result.to(torch.float64), labels.to(torch.float64))
        return [loss, result]


class TempDataset(Dataset):
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, item):
        return self.data[item], self.labels[item]


class LabelType(Enum):
    BEFORE = 0
    AFTER = 1
    EQUAL = 2
    VAGUE = 3

    @staticmethod
    def to_class_index(label_type):
        for label in LabelType:
            if label_type == label.name:
                return label.value


class temprel_ee:
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


def clean_str(text):
    text = text.lower()
    return text.strip()


def load_data_and_labels(data):
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


def calc_f1(predicted_labels, all_labels, label_type):
    confusion = np.zeros((len(label_type), len(label_type)))
    for i in range(len(predicted_labels)):
        confusion[all_labels[i]][predicted_labels[i]] += 1

    acc = 1.0 * np.sum([confusion[i][i] for i in range(4)]) / np.sum(confusion)
    true_positive = 0
    for i in range(len(label_type) - 1):
        true_positive += confusion[i][i]
    prec = true_positive / (np.sum(confusion) - np.sum(confusion, axis=0)[-1])
    rec = true_positive / (np.sum(confusion) - np.sum(confusion[-1][:]))
    f1 = 2 * prec * rec / (rec + prec)

    return (
        acc,
        prec,
        rec,
        f1,
    )


output_model = "/content/gdrive/MyDrive/model_train.pth"


def save(model, optimizer):
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
        },
        output_model,
    )


def train(
    model,
    train_dataloader,
    dev_dataloader,
    test_dataloader,
    total_train_graphs,
    total_test_graphs,
    device,
    n_gpu,
):
    num_training_steps_per_epoch = ceil(len(train_dataloader.dataset) / float(64))
    num_training_steps = 2 * num_training_steps_per_epoch
    parameters_to_optimize = list(model.named_parameters())
    no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [
                p
                for n, p in parameters_to_optimize
                if not any(nd in n for nd in no_decay)
            ],
            "weight_decay": 0.01,
        },
        {
            "params": [
                p for n, p in parameters_to_optimize if any(nd in n for nd in no_decay)
            ],
            "weight_decay": 0.0,
        },
    ]
    optimizer = AdamW(
        optimizer_grouped_parameters, lr=4e-5, eps=1e-8, betas=(0.9, 0.999)
    )

    num_warmup_steps = ceil(0.1 * num_training_steps)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps,
    )

    checkpoint = torch.load(output_model, map_location="cpu")
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    global_step = 0
    best_acc = 0.0
    update_per_batch = 2
    for epoch in range(1, 11, 1):
        model.train()
        global_loss = 0.0
        for i, batch in tqdm(
            enumerate(train_dataloader),
            desc=f"Running train for epoch {epoch}",
            total=len(train_dataloader),
        ):
            batch = [x.to(device) for x in batch]
            inputs = {
                "input_ids": batch[0],
                "attention_mask": batch[1],
                "event_ix": batch[2],
                "labels": batch[3],
                "graph": total_train_graphs[
                    i * len(batch[0]) : (i + 1) * len(batch[0])
                ],
            }
            outputs = model(**inputs)
            loss, logits = outputs[0], outputs[1]

            loss /= update_per_batch
            loss.backward()
            if (i + 1) % update_per_batch == 0 or (i + 1) == len(train_dataloader):
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1)

                optimizer.step()
                scheduler.step()
                model.zero_grad()
                global_step += 1
                global_loss = 0
        logging.info(f"Evaluation for epoch {epoch}")
        model.eval()
        label_type = LabelType
        all_logits, all_labels = [], []
        with torch.no_grad():
            for j, batch in tqdm(enumerate(dev_dataloader), total=len(dev_dataloader)):
                batch = [x.to(device) for x in batch]
                inputs = {
                    "input_ids": batch[0],
                    "attention_mask": batch[1],
                    "event_ix": batch[2],
                    "labels": batch[3],
                    "graph": total_test_graphs[
                        j * len(batch[0]) : (j + 1) * len(batch[0])
                    ],
                }
                outputs = model(**inputs)
                loss, logits = outputs[0], outputs[1]
                all_logits.append(logits)
                all_labels.append(inputs["labels"])

        all_logits = torch.cat(all_logits, dim=0)
        all_labels = torch.cat(all_labels, dim=0)

        predicted_logits, predicted_labels = torch.max(all_logits, dim=1)

        dev_acc, dev_prec, dev_rec, dev_f1 = calc_f1(
            predicted_labels, all_labels, label_type
        )
        print(f"Acc={dev_acc}, Precision={dev_prec}, Recall={dev_rec}, F1={dev_f1}")
        save(model, optimizer)
        if dev_f1 > best_acc:
            logging.info(f"New best, dev_f1={dev_f1} > best_f1={best_acc}")
            best_acc = dev_f1


def _get_tensorset(tokenizer):
    logging.info("***** Loading Dataset *****\n")
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
    logging.info(f"Test = {len(test_tensorset)}")
    return (
        train_tensorset,
        dev_tensorset,
        test_tensorset,
        train_text,
        dev_text,
        test_text,
    )


def generateDependencyGraph(edge_dict, train_text, count, total_train_graphs):
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


def prepareData(val_text, val_label):
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
    file_name = file.split("/")
    return file_name[-1]


def parseEntity(event, time):
    if event[0] == "e":
        c = "EVENT"
    elif time[0] == "t":
        c = "TIMEX3"
    else:
        c = "OTHERS"
    return c


def tokenize_and_align_labels(examples):
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


def compute_metrics(p):
    predictions, labels = p
    predictions = np.argmax(predictions, axis=2)
    true_predictions = [
        [label_list[p] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]
    true_labels = [
        [label_list[l] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]

    results = metric.compute(predictions=true_predictions, references=true_labels)
    return {
        "precision": results["overall_precision"],
        "recall": results["overall_recall"],
        "f1": results["overall_f1"],
        "accuracy": results["overall_accuracy"],
    }


files, text, relations_ = inputCol(300)


sentences = sent_tokenize(TreebankWordDetokenizer().detokenize(text))

for i in range(len(sentences)):
    sentences[i] = word_tokenize(sentences[i])

import re

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

data = []
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

EA = EmbeddingAugmenter()
WA = WordNetAugmenter()

tokens_list = [[]]
synonyms_list = []
tags = [[]]
synonyms_tags = []
for file in files:
    with open(file) as f:
        next(f)
        for line in f:
            columns = line.split("\t")
            tokens_list[-1].append(columns[0])
            tags[-1].append(columns[1].strip())
            if tags[-1][-1] == "EVENT":
                synonyms_list.append(EA.augment(tokens_list[-1][-1]))
                synonyms_tags.append("EVENT")
    tokens_list.append([])
    tags.append([])
tokens_list = tokens_list[:-1]


synTime = SynTime()
text = "The last 6 months surviving member of the team which first conquered Everest in 6 a.m. 17 Jan 1953 has died in a Derbyshire nursing home."
date = "2016-10-10"
timeMLText = synTime.extractTimexFromText(text, date)


le = preprocessing.LabelEncoder()
le.fit(tags[0])

for i in range(0, 150):
    tags[i] = le.transform(tags[i])

for i in range(150, 500):
    tags[i] = le.transform(tags[i])


train_dataset = Dataset.from_dict({"tokens": tokens_list[0:150], "tags": tags[0:150]})
test_dataset = Dataset.from_dict(
    {"tokens": tokens_list[150:500], "tags": tags[150:500]}
)

tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")

label_all_tokens = True


train_tokenize = train_dataset.map(tokenize_and_align_labels, batched=True)
test_tokenize = test_dataset.map(tokenize_and_align_labels, batched=True)


model = AutoModelForTokenClassification.from_pretrained(
    "distilbert-base-uncased", num_labels=3
)

class_weights = class_weight.compute_class_weight(
    class_weight="balanced", classes=[0, 1, 2], y=list(np.concatenate(tags).flat)
)


args = TrainingArguments(
    evaluation_strategy="epoch",
    output_dir="/output",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=10,
    weight_decay=0.01,
)


data_collator = DataCollatorForTokenClassification(tokenizer)

metric = load_metric("seqeval")

label_list = le.classes_

labels = [label_list[i] for i in train_dataset["tags"][0]]
metric.compute(predictions=[labels], references=[labels])


trainer = CustomTrainer(
    model,
    args,
    train_dataset=train_tokenize,
    eval_dataset=test_tokenize,
    data_collator=data_collator,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
)

trainer.train()

trainer.evaluate()

predictions, labels, _ = trainer.predict(train_tokenize)
predictions = np.argmax(predictions, axis=2)

true_predictions = [
    [label_list[p] for (p, l) in zip(prediction, label) if l != -100]
    for prediction, label in zip(predictions, labels)
]
true_labels = [
    [label_list[l] for (p, l) in zip(prediction, label) if l != -100]
    for prediction, label in zip(predictions, labels)
]

results = metric.compute(predictions=true_predictions, references=true_labels)
results

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
tags = [class2label[lab[1]] for lab in data]

class_weights = class_weight.compute_class_weight(
    class_weight="balanced", classes=np.unique(tags), y=tags
)


nlp = spacy.load("en_core_web_sm")


tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
(
    train_tensorset,
    dev_tensorset,
    test_tensorset,
    train_text,
    dev_text,
    test_text,
) = _get_tensorset(tokenizer)
bert_config = AutoConfig.from_pretrained("distilbert-base-uncased")
train_dataloader = DataLoader(train_tensorset, batch_size=32, shuffle=True)
dev_dataloader = DataLoader(dev_tensorset, batch_size=16, shuffle=False)
test_dataloader = DataLoader(test_tensorset, batch_size=16, shuffle=False)

total_train_graphs = []
total_dev_graphs = []
total_test_graphs = []
edge_dict = {
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
count = 0
total_train_graphs = generateDependencyGraph(
    edge_dict, train_text, count, total_train_graphs
)
total_dev_graphs = generateDependencyGraph(edge_dict, dev_text, count, total_dev_graphs)
total_test_graphs = generateDependencyGraph(
    edge_dict, test_text, count, total_test_graphs
)
max_nodes = max(
    [g.num_nodes() for g in total_train_graphs + total_test_graphs + total_dev_graphs]
)
for g in total_train_graphs:
    num_to_add = max_nodes - g.num_nodes()
    g.add_nodes(num_to_add)
    g.ndata["node_feat"][-num_to_add:] = 0
for g in total_dev_graphs:
    num_to_add = max_nodes - g.num_nodes()
    g.add_nodes(num_to_add)
    g.ndata["node_feat"][-num_to_add:] = 0
for g in total_test_graphs:
    num_to_add = max_nodes - g.num_nodes()
    g.add_nodes(num_to_add)
    g.ndata["node_feat"][-num_to_add:] = 0

model = TempRelModel(
    bert_config,
    tokenizer,
    4,
    edge_dict,
    max_nodes,
    pretrained_weights="distilbert-base-uncased",
)

train(
    model,
    train_dataloader,
    test_dataloader,
    dev_dataloader,
    total_train_graphs,
    total_test_graphs,
    torch.device("cpu"),
    0,
)

# Training on the TempEval-3 Dataset

special_tokens_dict = {"additional_special_tokens": ["$", "#"]}
tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
tokenizer.add_special_tokens(special_tokens_dict)

cls_id = tokenizer.cls_token_id
sep_id = tokenizer.sep_token_id
pad_id = tokenizer.pad_token_id
e1_id, e2_id = tokenizer.additional_special_tokens_ids

bert_config = AutoConfig.from_pretrained("distilbert-base-uncased")

x_val, y_val, _ = load_data_and_labels(data)
dataset = TempDataset(x_val, y_val)
train_indices, test_indices, _, _ = train_test_split(
    range(len(dataset)), dataset.labels, test_size=0.3
)
train_split = Subset(dataset, train_indices)
test_split = Subset(dataset, test_indices)
train_loader = DataLoader(train_split, batch_size=32, shuffle=True)
test_loader = DataLoader(test_split, batch_size=32)


global_step = 0
tr_loss = 0.0
pred_y = []
labels1 = []
pred_y1 = []
pred_y = []
labels = []
train_graphs = []
total_train_graphs = []
batch_train_graphs = []
test_graphs = []
total_test_graphs = []
batch_test_graphs = []
count = 0
for index, (val_text, val_label) in enumerate(train_loader):
    for text in val_text:
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
        train_graphs.append(g)
        total_train_graphs.append(g)
    batch_train_graphs.append(train_graphs)
    train_graphs = []
for index, (val_text, val_label) in enumerate(test_loader):
    for text in val_text:
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
        test_graphs.append(g)
        total_test_graphs.append(g)
    batch_test_graphs.append(test_graphs)
    test_graphs = []
max_nodes = max([g.num_nodes() for g in total_train_graphs + total_test_graphs])
for batch in batch_train_graphs:
    for g in batch:
        num_to_add = max_nodes - g.num_nodes()
        g.add_nodes(num_to_add)
        g.ndata["node_feat"][-num_to_add:] = 0
for batch in batch_test_graphs:
    for g in batch:
        num_to_add = max_nodes - g.num_nodes()
        g.add_nodes(num_to_add)
        g.ndata["node_feat"][-num_to_add:] = 0


model = TempRel2Model(
    bert_config,
    tokenizer,
    len(np.unique(tags)),
    edge_dict,
    max_nodes,
    pretrained_weights="distilbert-base-uncased",
)


"""output_model = '/content/gdrive/MyDrive/model_test.pth'
def save(model, optimizer):
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict()
    }, output_model)"""

t_total = len(train_loader) // 10
no_decay = ["bias", "LayerNorm.weight"]
optimizer_grouped_parameters = [
    {
        "params": [
            p
            for n, p in model.named_parameters()
            if not any(nd in n for nd in no_decay)
        ],
        "weight_decay": 0,
    },
    {
        "params": [
            p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)
        ],
        "weight_decay": 0.0,
    },
]
optimizer = AdamW(
    optimizer_grouped_parameters,
    lr=2e-5,
    eps=1e-8,
)
scheduler = get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps=0,
    num_training_steps=t_total,
)

"""checkpoint = torch.load(output_model, map_location='cpu')
model.load_state_dict(checkpoint['model_state_dict'])
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])"""

global_step = 0
tr_loss = 0.0
pred_y = []
labels1 = []
pred_y1 = []
model.zero_grad()
pred_y = []
labels = []
for count in range(1):
    for index, (val_text, val_label) in enumerate(train_loader):
        labels1 = []
        x_token = []
        x_mark_index_all = []
        x_token, x_mark_index_all, labels1 = prepareData(val_text, val_label)
        labels.extend(labels1)
        if x_token == []:
            continue
        x_token = np.vstack(x_token).astype(float)
        x_token = np.array(x_token)
        x_token = torch.from_numpy(x_token)
        model.train()
        out = model(
            x_token.clone().detach().requires_grad_(True).long(),
            x_mark_index_all,
            Fun.one_hot(
                torch.from_numpy(np.asarray(labels1)).to(torch.int64),
                batch_train_graphs,
                num_classes=len(np.unique(tags)),
            ),
        )
        loss = out[0]
        loss.backward()
        tr_loss += loss.item()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5)
        optimizer.step()
        scheduler.step()
        model.zero_grad()
        global_step += 1
        pred_y = []
        pred_y.extend(torch.max(out[1], 1)[1].tolist())
        pred_y1.extend(torch.max(out[1], 1)[1].tolist())
        val_acc = np.mean(np.equal(labels1, pred_y))
        print("Train: ACC: {}".format(val_acc))
    tlabels = []
    tpred_y = []
    with torch.no_grad():
        for index, (test_text, test_label) in enumerate(test_loader):
            tlabels1 = []
            tx_token = []
            tx_mark_index_all = []
            tx_token, tx_mark_index_all, tlabels1 = prepareData(test_text, test_label)
            tlabels.extend(tlabels1)
            if x_token == []:
                continue
            tx_token = np.vstack(tx_token).astype(float)
            tx_token = np.array(tx_token)
            tx_token = torch.from_numpy(tx_token)
            model.eval()
            out = model(
                tx_token.clone().detach().requires_grad_(True).long(),
                tx_mark_index_all,
                Fun.one_hot(
                    torch.from_numpy(np.asarray(tlabels1)).to(torch.int64),
                    batch_test_graphs,
                    num_classes=len(np.unique(tags)),
                ),
            )
            tpred_y.extend(torch.max(out[1], 1)[1].tolist())
    val_acc = np.mean(np.equal(tlabels, tpred_y))
    print("Testing")
    print("Test: ACC: {}".format(val_acc))

sentence = [" The patient went into a $ coma $ after he had a # heart attack # ."]
tx_token, tx_mark_index_all, tlabels1 = prepareData(sentence, None)
tx_token = np.vstack(tx_token).astype(float)
tx_token = np.array(tx_token)
tx_token = torch.from_numpy(tx_token)
out = model(tx_token.clone().detach().requires_grad_(True).long(), tx_mark_index_all)

print(label2class[torch.max(out[1], 1)[1].item()])
