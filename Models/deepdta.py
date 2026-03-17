import os
import math
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import (
    roc_curve, auc,
    precision_recall_curve,
    confusion_matrix
)





class DeepDTA(nn.Module):
    def __init__(self):
        super().__init__()

        emb_dim = 128
        num_filters = 32

        self.smi_emb = nn.Embedding(len(smi_stoi)+1, emb_dim, padding_idx=0)
        self.seq_emb = nn.Embedding(len(seq_stoi)+1, emb_dim, padding_idx=0)

        self.smi_conv1 = nn.Conv1d(emb_dim, num_filters, 4)
        self.smi_conv2 = nn.Conv1d(num_filters, num_filters*2, 4)
        self.smi_conv3 = nn.Conv1d(num_filters*2, num_filters*3, 4)

        self.seq_conv1 = nn.Conv1d(emb_dim, num_filters, 8)
        self.seq_conv2 = nn.Conv1d(num_filters, num_filters*2, 8)
        self.seq_conv3 = nn.Conv1d(num_filters*2, num_filters*3, 8)

        self.fc1 = nn.Linear(num_filters*3*2, 1024)
        self.fc2 = nn.Linear(1024, 1024)
        self.fc3 = nn.Linear(1024, 512)
        self.out = nn.Linear(512, 1)

        self.dropout = nn.Dropout(0.1)

    def cnn_block(self, x, conv1, conv2, conv3):
        x = x.transpose(1,2)
        x = F.relu(conv1(x))
        x = F.relu(conv2(x))
        x = F.relu(conv3(x))
        x = torch.max(x, dim=2).values
        return x

    def forward(self, smi, seq):
        smi = self.smi_emb(smi)
        seq = self.seq_emb(seq)

        smi_feat = self.cnn_block(smi, self.smi_conv1, self.smi_conv2, self.smi_conv3)
        seq_feat = self.cnn_block(seq, self.seq_conv1, self.seq_conv2, self.seq_conv3)

        x = torch.cat([smi_feat, seq_feat], dim=1)

        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = F.relu(self.fc3(x))

        return self.out(x)


