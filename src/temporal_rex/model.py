"""
Model definitions for temporal relation extraction.

This module contains neural network architectures including BERT-based models
with Graph Attention Networks for temporal relation classification.
"""

import torch
import torch.nn as nn
import torch.nn.functional as Fun
from transformers import BertPreTrainedModel, AutoModel, Trainer
from transformers import AdamW, get_linear_schedule_with_warmup
from sklearn.utils import class_weight
import dgl
from dgl.nn import GATConv
import networkx as nx
import numpy as np
from math import ceil
import logging
from tqdm import tqdm


class CustomTrainer(Trainer):
    """Custom trainer with weighted loss function."""
    
    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.get("labels")
        outputs = model(**inputs)
        logits = outputs.get("logits")
        loss_fct = nn.CrossEntropyLoss(weight=torch.tensor(class_weights).float())
        loss = loss_fct(logits.view(-1, self.model.config.num_labels), labels.view(-1))
        return (loss, outputs) if return_outputs else loss


class GAT(nn.Module):
    """Graph Attention Network for processing dependency graphs."""
    
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


class TempRelModel(BertPreTrainedModel):
    """Temporal Relation Model for MATRES Dataset."""
    
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


class TempRel2Model(BertPreTrainedModel):
    """Temporal Relation Model for TempEval-3 Dataset."""
    
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


def train(
    model,
    train_dataloader,
    dev_dataloader,
    test_dataloader,
    total_train_graphs,
    total_test_graphs,
    device,
    n_gpu,
    output_model=None,
    LabelType=None,
    calc_f1=None,
):
    """Training function for temporal relation models."""
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

    # Load checkpoint if provided
    if output_model:
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

        if calc_f1:
            dev_acc, dev_prec, dev_rec, dev_f1 = calc_f1(
                predicted_labels, all_labels, label_type
            )
            print(f"Acc={dev_acc}, Precision={dev_prec}, Recall={dev_rec}, F1={dev_f1}")
        else:
            dev_f1 = 0.0  # Default if calc_f1 not provided
            
        # Save model using utils save function
        from .utils import save
        save(model, optimizer)
        
        if dev_f1 > best_acc:
            logging.info(f"New best, dev_f1={dev_f1} > best_f1={best_acc}")
            best_acc = dev_f1


# Import the enhanced predict_relations function from utils
from .utils import predict_relations

# Global variables that would be set during training
class_weights = None
