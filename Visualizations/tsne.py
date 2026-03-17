import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

from sklearn.manifold import TSNE
from torch_geometric.data import Batch
from torch_geometric.nn import TransformerConv, global_mean_pool


# PATHS


BASE_DIR = "/media/8TB_hardisk/hamza/hormones project/hormones"

FOLD_DIR    = os.path.join(BASE_DIR, "folds")
MODEL_DIR   = os.path.join(BASE_DIR, "models")
FEATURE_DIR = os.path.join(BASE_DIR, "features")
FIGURE_DIR  = os.path.join(BASE_DIR, "figures")

FP_PT_PATH  = os.path.join(FEATURE_DIR, "fingerprints/fingerprints.pt")
ESM_PT_PATH = os.path.join(FEATURE_DIR, "esm/esm_embeddings.pt")
BEST_MODEL_PATH = os.path.join(MODEL_DIR, "esm_fp_best_5fold.pt")

BEST_FOLD = 3

os.makedirs(FIGURE_DIR, exist_ok=True)

DEVICE = torch.device("cuda:7" if torch.cuda.device_count() > 7 else "cuda:0")


# SETTINGS


TSNE_SAMPLES = 2000
TSNE_PERPLEXITY = 30
TSNE_RANDOM_STATE = 42
BATCH_SIZE = 64


# DATASET


class CachedDTIDataset(torch.utils.data.Dataset):
    def __init__(self, df, fp_tensor, esm_tensor):
        self.df = df.reset_index(drop=True)
        self.fp = fp_tensor
        self.esm = esm_tensor

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        sid = int(row["sample_id"])

        graph = torch.load(row["graph_path"], weights_only=False)
        fp = self.fp[sid].float()
        esm = self.esm[sid].float()
        label = int(row["label"])

        return graph, fp, esm, label


def collate_cached(batch):
    graphs, fps, esms, labels = zip(*batch)
    return (
        Batch.from_data_list(list(graphs)),
        torch.stack(fps),
        torch.stack(esms),
        torch.tensor(labels)
    )


# MODEL (same as training)


class DrugEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.gnn1 = TransformerConv(9, 128, heads=4, edge_dim=4)
        self.gnn2 = TransformerConv(512, 256, heads=4, edge_dim=4)
        self.fp_fc = nn.Linear(2048, 256)
        self.final_fc = nn.Linear(1024 + 256, 1024)

    def forward(self, graph, fp):
        x = graph.x.to(DEVICE)
        edge_index = graph.edge_index.to(DEVICE)
        edge_attr = graph.edge_attr.to(DEVICE)
        batch = graph.batch.to(DEVICE)

        x = F.relu(self.gnn1(x, edge_index, edge_attr))
        x = self.gnn2(x, edge_index, edge_attr)

        g = global_mean_pool(x, batch)
        fp = self.fp_fc(fp.to(DEVICE))

        return self.final_fc(torch.cat([g, fp], dim=-1))


class FusionModule(nn.Module):
    def __init__(self):
        super().__init__()
        self.proj_p = nn.Linear(480, 1024)

    def forward(self, drug, esm):
        p = self.proj_p(esm.to(DEVICE))
        prod = drug * p
        return torch.cat([drug, p, prod], dim=-1)


class DTIModelCached(nn.Module):
    def __init__(self):
        super().__init__()
        self.drug = DrugEncoder()
        self.fusion = FusionModule()

    def forward(self, graph, fp, esm):
        d = self.drug(graph, fp)
        return self.fusion(d, esm)


# LOAD FEATURES


fp_tensor  = torch.load(FP_PT_PATH, map_location="cpu")
esm_tensor = torch.load(ESM_PT_PATH, map_location="cpu")

# ============================================================
# LOAD DATA
# ============================================================

val_df = pd.read_csv(os.path.join(FOLD_DIR, f"fold{BEST_FOLD}_val.csv"))
test_df = pd.read_csv(os.path.join(FOLD_DIR, "test.csv"))

for df in [val_df, test_df]:
    df["sample_id"] = df["sample_id"].astype(int)
    df["label"] = df["label"].astype(int)

val_df = val_df.sample(min(TSNE_SAMPLES, len(val_df)), random_state=TSNE_RANDOM_STATE)
test_df = test_df.sample(min(TSNE_SAMPLES, len(test_df)), random_state=TSNE_RANDOM_STATE)

val_loader = torch.utils.data.DataLoader(
    CachedDTIDataset(val_df, fp_tensor, esm_tensor),
    batch_size=BATCH_SIZE,
    shuffle=False,
    collate_fn=collate_cached
)

test_loader = torch.utils.data.DataLoader(
    CachedDTIDataset(test_df, fp_tensor, esm_tensor),
    batch_size=BATCH_SIZE,
    shuffle=False,
    collate_fn=collate_cached
)


# EMBEDDING EXTRACTION


def extract_embeddings(model, loader):
    model.eval()
    embs = []

    with torch.no_grad():
        for g, fp, esm, _ in loader:
            fused = model(g, fp, esm)
            embs.append(fused.cpu().numpy())

    return np.vstack(embs)


# t-SNE FUNCTION


def run_tsne(emb, labels, title, filename):
    tsne = TSNE(
        n_components=2,
        perplexity=TSNE_PERPLEXITY,
        random_state=TSNE_RANDOM_STATE,
        init="pca"
    )
    emb2d = tsne.fit_transform(emb)

    plt.figure(figsize=(6, 5))
    plt.scatter(
        emb2d[:, 0], emb2d[:, 1],
        c=labels, cmap="coolwarm",
        s=12, alpha=0.75
    )
    plt.xticks([])
    plt.yticks([])
    plt.title(title, fontweight="bold", fontsize=14)
    plt.tight_layout()

    path = os.path.join(FIGURE_DIR, filename)
    plt.savefig(path, dpi=1000)
    plt.close()
    print("Saved:", path)


# RUN


model = DTIModelCached().to(DEVICE)

ckpt = torch.load(BEST_MODEL_PATH, map_location=DEVICE, weights_only=False)
model.load_state_dict(ckpt["model_state"], strict=False)

emb_val = extract_embeddings(model, val_loader)
run_tsne(emb_val, val_df["label"].values, "t-SNE Validation", "tsne_val.png")

emb_test = extract_embeddings(model, test_loader)
run_tsne(emb_test, test_df["label"].values, "t-SNE Test", "tsne_test.png")

print("t-SNE completed")





