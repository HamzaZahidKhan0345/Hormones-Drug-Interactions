import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.nn import TransformerConv, global_mean_pool


class DrugEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.gnn1 = TransformerConv(9, 128, heads=4, edge_dim=4)      # -> 512
        self.gnn2 = TransformerConv(512, 256, heads=4, edge_dim=4)    # -> 1024

        self.fp_fc = nn.Linear(2048, 256)
        self.final_fc = nn.Linear(1024 + 256, 1024)

    def forward(self, graph_batch, fp_batch):
        x = graph_batch.x
        edge_index = graph_batch.edge_index
        edge_attr = graph_batch.edge_attr
        batch = graph_batch.batch

        x = F.relu(self.gnn1(x, edge_index, edge_attr))
        x = self.gnn2(x, edge_index, edge_attr)

        g = global_mean_pool(x, batch)          # [B, 1024]
        fp = self.fp_fc(fp_batch)               # [B, 256]

        d = self.final_fc(torch.cat([g, fp], dim=-1))  # [B, 1024]
        return d


class FusionModule(nn.Module):
    def __init__(self, drug_dim=1024, esm_dim=480):
        super().__init__()
        self.proj_p = nn.Linear(esm_dim, drug_dim)

        self.fc = nn.Sequential(
            nn.Linear(drug_dim * 3, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 1)
        )

    def forward(self, drug, esm):
        p = self.proj_p(esm)
        prod = drug * p
        fused = torch.cat([drug, p, prod], dim=-1)
        return self.fc(fused)


class DTIModelCached(nn.Module):
    def __init__(self):
        super().__init__()
        self.drug = DrugEncoder()
        self.fusion = FusionModule()

    def forward(self, graph_batch, fp_batch, esm_batch):
        d = self.drug(graph_batch, fp_batch)
        out = self.fusion(d, esm_batch)
        return out