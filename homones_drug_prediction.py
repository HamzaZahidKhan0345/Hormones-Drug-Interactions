

import os
import time
import numpy as np
import pandas as pd

import torch
import torch.nn as nn

from rdkit import Chem
from rdkit.Chem import AllChem

from transformers import AutoTokenizer, AutoModel

from torch_geometric.data import Data




BASE_DIR = "/media/8TB_hardisk/hamza/hormones project/hormones"
DATA_DIR = os.path.join(BASE_DIR, "data")
FEATURE_DIR = os.path.join(BASE_DIR, "features")

GRAPH_DIR = os.path.join(FEATURE_DIR, "graphs")
FP_DIR = os.path.join(FEATURE_DIR, "fingerprints")
ESM_DIR = os.path.join(FEATURE_DIR, "esm")

CSV_PATH = os.path.join(DATA_DIR, "DTI_Hormone_2.csv")
MANIFEST_PATH = os.path.join(FEATURE_DIR, "manifest.csv")

FP_PT_PATH = os.path.join(FP_DIR, "fingerprints.pt")
FP_CSV_PATH = os.path.join(FP_DIR, "fingerprints.csv")

ESM_PT_PATH = os.path.join(ESM_DIR, "esm_embeddings.pt")
ESM_CSV_PATH = os.path.join(ESM_DIR, "esm_embeddings.csv")

# create directories
os.makedirs(GRAPH_DIR, exist_ok=True)
os.makedirs(FP_DIR, exist_ok=True)
os.makedirs(ESM_DIR, exist_ok=True)


DEVICE = torch.device("cuda:7" if torch.cuda.device_count() > 7 else "cuda:0")
print("Using device:", DEVICE)





def atom_features(atom):
    return [
        atom.GetAtomicNum(),
        atom.GetDegree(),
        atom.GetFormalCharge(),
        int(atom.GetHybridization()),
        int(atom.GetIsAromatic()),
        atom.GetTotalNumHs(),
        int(atom.IsInRing()),
        int(atom.GetChiralTag()),
        atom.GetImplicitValence()
    ]


def bond_features(bond):
    return [
        int(bond.GetBondType()),
        int(bond.GetIsConjugated()),
        int(bond.IsInRing()),
        int(bond.GetStereo())
    ]








def smiles_to_graph(smiles: str) -> Data:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        mol = Chem.MolFromSmiles("C")

    x = torch.tensor(
        [atom_features(a) for a in mol.GetAtoms()],
        dtype=torch.float
    )

    edge_index = []
    edge_attr = []

    for bond in mol.GetBonds():
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()
        bf = bond_features(bond)

        edge_index.append([i, j])
        edge_index.append([j, i])
        edge_attr.append(bf)
        edge_attr.append(bf)

    if len(edge_index) == 0:
        edge_index = torch.zeros((2, 0), dtype=torch.long)
        edge_attr = torch.zeros((0, 4), dtype=torch.float)
    else:
        edge_index = torch.tensor(edge_index, dtype=torch.long).t()
        edge_attr = torch.tensor(edge_attr, dtype=torch.float)

    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr)





def smiles_to_fingerprint(smiles, radius=2, nBits=2048):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        mol = Chem.MolFromSmiles("C")

    fp = AllChem.GetMorganFingerprintAsBitVect(
        mol, radius, nBits=nBits
    )
    return np.array(fp, dtype=np.float32)





df = pd.read_csv(CSV_PATH).reset_index(drop=True)

# assign stable index
df["sample_id"] = np.arange(len(df), dtype=int)

manifest = df[[
    "sample_id",
    "Drug_SMILES",
    "sequence_assigned",
    "Class"
]].copy()

manifest.rename(columns={"Class": "label"}, inplace=True)

manifest["graph_path"] = manifest["sample_id"].apply(
    lambda i: os.path.join(GRAPH_DIR, f"graph_{i}.pt")
)

manifest.to_csv(MANIFEST_PATH, index=False)

print("Total samples:", len(manifest))
print("Manifest saved to:", MANIFEST_PATH)
print(manifest["label"].value_counts(normalize=True))




N = len(manifest)
fps = np.zeros((N, 2048), dtype=np.float32)

t0 = time.time()
for i in range(N):
    sid = manifest.loc[i, "sample_id"]
    smiles = manifest.loc[i, "Drug_SMILES"]
    gpath = manifest.loc[i, "graph_path"]

    if not os.path.exists(gpath):
        g = smiles_to_graph(smiles)
        torch.save(g, gpath)

    fps[sid] = smiles_to_fingerprint(smiles)

    if (i + 1) % 200 == 0 or (i + 1) == N:
        print(f"[Graph+FP] {i+1}/{N}")

torch.save(torch.tensor(fps), FP_PT_PATH)
pd.DataFrame(fps, index=manifest["sample_id"]).to_csv(FP_CSV_PATH)

print("Fingerprints saved.")
print(f"Time: {time.time() - t0:.1f}s")







N = len(manifest)

tokenizer = AutoTokenizer.from_pretrained(
    "facebook/esm2_t12_35M_UR50D"
)
esm = AutoModel.from_pretrained(
    "facebook/esm2_t12_35M_UR50D"
).to(DEVICE)
esm.eval()

esm_embs = np.zeros((N, 480), dtype=np.float32)

with torch.no_grad():
    for i in range(N):
        seq = manifest.loc[i, "sequence_assigned"]

        tokens = tokenizer(
            seq,
            return_tensors="pt",
            truncation=True,
            max_length=1024
        )
        tokens = {k: v.to(DEVICE) for k, v in tokens.items()}

        out = esm(**tokens).last_hidden_state
        pooled = out.mean(dim=1)

        esm_embs[i] = pooled.squeeze(0).cpu().numpy()

        if (i + 1) % 100 == 0 or (i + 1) == N:
            print(f"[ESM] {i+1}/{N}")

# 🔴 THIS PART WAS MISSING
esm_tensor = torch.tensor(esm_embs, dtype=torch.float32)

torch.save(esm_tensor, ESM_PT_PATH)
pd.DataFrame(
    esm_embs,
    index=manifest["sample_id"]
).to_csv(ESM_CSV_PATH)

print("✅ ESM embeddings saved to:", ESM_PT_PATH)
print("✅ ESM CSV saved to:", ESM_CSV_PATH)





FOLD_DIR = "/media/8TB_hardisk/hamza/hormones project/hormones/folds"
FIGURE_DIR = "/media/8TB_hardisk/hamza/hormones project/hormones/figures"

os.makedirs(FOLD_DIR, exist_ok=True)
os.makedirs(FIGURE_DIR, exist_ok=True)

print("Fold directory:", FOLD_DIR)
print("Figure directory:", FIGURE_DIR)






MANIFEST_PATH = os.path.join(FEATURE_DIR, "manifest.csv")

if not os.path.exists(MANIFEST_PATH):
    raise FileNotFoundError("manifest.csv not found in features directory")

manifest = pd.read_csv(MANIFEST_PATH)

# Safety checks
assert "sample_id" in manifest.columns
assert "label" in manifest.columns
assert "graph_path" in manifest.columns

manifest["label"] = manifest["label"].astype(int)

print("Total samples:", len(manifest))
print("Label distribution:")
print(manifest["label"].value_counts(normalize=True))







from sklearn.model_selection import train_test_split

trainval_df, test_df = train_test_split(
    manifest,
    test_size=0.10,
    stratify=manifest["label"],
    random_state=42
)

print("Train+Val size:", len(trainval_df))
print("Test size:", len(test_df))

test_path = os.path.join(FOLD_DIR, "test.csv")
test_df.to_csv(test_path, index=False)

print("Saved independent test set to:", test_path)




from sklearn.model_selection import StratifiedKFold

skf = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

fold = 1
for train_idx, val_idx in skf.split(trainval_df, trainval_df["label"]):

    fold_train = trainval_df.iloc[train_idx].copy()
    fold_val   = trainval_df.iloc[val_idx].copy()

    train_path = os.path.join(FOLD_DIR, f"fold{fold}_train.csv")
    val_path   = os.path.join(FOLD_DIR, f"fold{fold}_val.csv")

    # IMPORTANT: do NOT reset index or sample_id
    fold_train.to_csv(train_path, index=False)
    fold_val.to_csv(val_path, index=False)

    print(f"Fold {fold} saved:")
    print("  Train:", fold_train.shape)
    print("  Val  :", fold_val.shape)

    fold += 1

print("\n✅ All 5 folds created and saved successfully.")





FP_PT_PATH = os.path.join(FP_DIR, "fingerprints.pt")
ESM_PT_PATH = os.path.join(ESM_DIR, "esm_embeddings.pt")

if not os.path.exists(FP_PT_PATH):
    raise FileNotFoundError("fingerprints.pt not found in features/fingerprints/")
if not os.path.exists(ESM_PT_PATH):
    raise FileNotFoundError("esm_embeddings.pt not found in features/esm/")

fp_tensor = torch.load(FP_PT_PATH, map_location="cpu")   # [N, 2048]
esm_tensor = torch.load(ESM_PT_PATH, map_location="cpu") # [N, 480]

print("Loaded fp_tensor:", tuple(fp_tensor.shape))
print("Loaded esm_tensor:", tuple(esm_tensor.shape))


class CachedDTIDataset(torch.utils.data.Dataset):
    """
    Loads graph from graph_path; loads fp/esm by sample_id index.
    Assumes sample_id matches row index used during feature creation.
    """
    def __init__(self, df_fold, fp_tensor, esm_tensor):
        self.df = df_fold.reset_index(drop=True)
        self.fp = fp_tensor
        self.esm = esm_tensor

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        sid = int(row["sample_id"])
        y = torch.tensor(float(row["label"]), dtype=torch.float32)

        graph_path = str(row["graph_path"])
        graph = torch.load(graph_path, weights_only=False)  # PyTorch 2.6 fix

        fp = self.fp[sid].float()     # [2048]
        esm = self.esm[sid].float()   # [480]

        return graph, fp, esm, y


from torch_geometric.data import Batch

def collate_cached(batch):
    graphs, fps, esms, ys = zip(*batch)
    graph_batch = Batch.from_data_list(list(graphs))
    fp_batch = torch.stack(fps, dim=0)
    esm_batch = torch.stack(esms, dim=0)
    y_batch = torch.stack(ys, dim=0).view(-1, 1)
    return graph_batch, fp_batch, esm_batch, y_batch





from torch_geometric.nn import TransformerConv, global_mean_pool

class DrugEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.gnn1 = TransformerConv(9, 128, heads=4, edge_dim=4)      # -> 512
        self.gnn2 = TransformerConv(512, 256, heads=4, edge_dim=4)    # -> 1024

        self.fp_fc = nn.Linear(2048, 256)
        self.final_fc = nn.Linear(1024 + 256, 1024)

    def forward(self, graph_batch: Batch, fp_batch: torch.Tensor):
        x = graph_batch.x.to(DEVICE)
        edge_index = graph_batch.edge_index.to(DEVICE)
        edge_attr = graph_batch.edge_attr.to(DEVICE)
        batch = graph_batch.batch.to(DEVICE)

        x = F.relu(self.gnn1(x, edge_index, edge_attr))
        x = self.gnn2(x, edge_index, edge_attr)

        g = global_mean_pool(x, batch)          # [B, 1024]
        fp = self.fp_fc(fp_batch.to(DEVICE))    # [B, 256]

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
        p = self.proj_p(esm.to(DEVICE))      # [B, 1024]
        prod = drug * p                      # [B, 1024]
        fused = torch.cat([drug, p, prod], dim=-1) #[]
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

    
    
    


from sklearn.metrics import confusion_matrix
import math

def safe_div(a, b):
    return float(a) / float(b) if b != 0 else 0.0


def compute_binary_metrics(y_true, y_logit, threshold=0.5):
    y_true = np.asarray(y_true).astype(int)
    y_logit = np.asarray(y_logit).astype(float)

    y_prob = 1.0 / (1.0 + np.exp(-y_logit))
    y_pred = (y_prob >= threshold).astype(int)

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

    acc = safe_div(tp + tn, tp + tn + fp + fn)
    sensitivity = safe_div(tp, tp + fn)
    specificity = safe_div(tn, tn + fp)
    precision = safe_div(tp, tp + fp)
    recall = sensitivity
    f1 = safe_div(2 * precision * recall, precision + recall)

    denom = math.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    mcc = safe_div((tp * tn - fp * fn), denom)

    return {
        "acc": acc,
        "sensitivity": sensitivity,
        "specificity": specificity,
        "precision": precision,
        "recall": recall,
        "mcc": mcc,
        "f1": f1
    }


criterion = nn.BCEWithLogitsLoss()


def run_one_epoch(model, loader, optimizer=None):
    train_mode = optimizer is not None
    model.train() if train_mode else model.eval()

    total_loss = 0.0
    y_true, y_logit = [], []

    for graph_batch, fp_batch, esm_batch, y_batch in loader:
        graph_batch = graph_batch.to(DEVICE)
        fp_batch = fp_batch.to(DEVICE)
        esm_batch = esm_batch.to(DEVICE)
        y_batch = y_batch.to(DEVICE)

        if train_mode:
            optimizer.zero_grad()

        out = model(graph_batch, fp_batch, esm_batch)
        loss = criterion(out, y_batch)

        if train_mode:
            loss.backward()
            optimizer.step()

        total_loss += loss.item() * y_batch.size(0)
        y_true.extend(y_batch.detach().cpu().numpy().reshape(-1))
        y_logit.extend(out.detach().cpu().numpy().reshape(-1))

    avg_loss = total_loss / len(y_true)
    metrics = compute_binary_metrics(y_true, y_logit)

    return avg_loss, metrics




from dataclasses import dataclass
@dataclass
class EarlyStopping:
    patience: int = 30
    min_delta: float = 0.0
    best_score: float = float("inf")
    num_bad_epochs: int = 0
    stop: bool = False

    def step(self, current_score):
        if current_score < self.best_score - self.min_delta:
            self.best_score = current_score
            self.num_bad_epochs = 0
            return True
        else:
            self.num_bad_epochs += 1
            if self.num_bad_epochs >= self.patience:
                self.stop = True
            return False

import torch.nn.functional as F  
MODEL_DIR = "/media/8TB_hardisk/hamza/hormones project/hormones/models"
FIGURE_DIR = "/media/8TB_hardisk/hamza/hormones project/hormones/figures"
FOLD_DIR   = "/media/8TB_hardisk/hamza/hormones project/hormones/folds"

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(FIGURE_DIR, exist_ok=True)
EPOCHS = 1000
PATIENCE = 30
BATCH_SIZE = 32
fold_results = []



print("\n==================== 5-FOLD TRAINING START ====================")


for fold in range(1, 6):
    print(f"\n==================== FOLD {fold} ====================")

    # ------------------------------------------------------------
    # Load fold CSVs
    # ------------------------------------------------------------
    train_df = pd.read_csv(os.path.join(FOLD_DIR, f"fold{fold}_train.csv"))
    val_df   = pd.read_csv(os.path.join(FOLD_DIR, f"fold{fold}_val.csv"))

    train_df["sample_id"] = train_df["sample_id"].astype(int)
    val_df["sample_id"]   = val_df["sample_id"].astype(int)
    train_df["label"]     = train_df["label"].astype(int)
    val_df["label"]       = val_df["label"].astype(int)

    # ------------------------------------------------------------
    # Dataset & DataLoader
    # ------------------------------------------------------------
    train_ds = CachedDTIDataset(train_df, fp_tensor, esm_tensor)
    val_ds   = CachedDTIDataset(val_df, fp_tensor, esm_tensor)

    train_loader = torch.utils.data.DataLoader(
        train_ds, batch_size=BATCH_SIZE, shuffle=True,
        collate_fn=collate_cached, num_workers=0
    )

    val_loader = torch.utils.data.DataLoader(
        val_ds, batch_size=BATCH_SIZE, shuffle=False,
        collate_fn=collate_cached, num_workers=0
    )

    # ------------------------------------------------------------
    # Model / Optimizer / Scheduler / EarlyStopping
    # ------------------------------------------------------------
    model = DTIModelCached().to(DEVICE)

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=3e-4, weight_decay=1e-5
    )

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5,
        patience=5, verbose=True, min_lr=1e-6
    )

    early = EarlyStopping(patience=PATIENCE)

    best_model_path = os.path.join(
        MODEL_DIR, f"esm_fp_fold{fold}_best.pt"
    )

    best_epoch = None

    # ------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------
    for epoch in range(1, EPOCHS + 1):
        t0 = time.time()

        train_loss, train_m = run_one_epoch(
            model, train_loader, optimizer=optimizer
        )
        val_loss, val_m = run_one_epoch(
            model, val_loader, optimizer=None
        )

        scheduler.step(val_loss)

        improved = early.step(val_loss)
        if improved:
            best_epoch = epoch
            torch.save({
                "fold": fold,
                "epoch": epoch,
                "val_loss": val_loss,
                "metrics": val_m,
                "model_state": model.state_dict()
            }, best_model_path)

        dt = time.time() - t0
        lr_now = optimizer.param_groups[0]["lr"]

        # --------------------------------------------------------
        # PRINT ALL METRICS
        # --------------------------------------------------------
        print(
            f"[Fold {fold} | Epoch {epoch:04d}] "
            f"TRAIN | loss={train_loss:.4f} "
            f"acc={train_m['acc']:.3f} "
            f"sen={train_m['sensitivity']:.3f} "
            f"spe={train_m['specificity']:.3f} "
            f"prec={train_m['precision']:.3f} "
            f"rec={train_m['recall']:.3f} "
            f"mcc={train_m['mcc']:.3f} "
            f"f1={train_m['f1']:.3f} || "
            f"VAL | loss={val_loss:.4f} "
            f"acc={val_m['acc']:.3f} "
            f"sen={val_m['sensitivity']:.3f} "
            f"spe={val_m['specificity']:.3f} "
            f"prec={val_m['precision']:.3f} "
            f"rec={val_m['recall']:.3f} "
            f"mcc={val_m['mcc']:.3f} "
            f"f1={val_m['f1']:.3f} || "
            f"best_val_loss={early.best_score:.4f} "
            f"bad={early.num_bad_epochs:02d}/{early.patience} "
            f"lr={lr_now:.2e} "
            f"time={dt:.1f}s "
            f"{'**SAVED**' if improved else ''}"
        )

        if early.stop:
            print(
                f"\n=== EARLY STOPPING | Fold {fold} | "
                f"Best epoch={best_epoch} | "
                f"Best val_loss={early.best_score:.6f} ==="
            )
            break

    fold_results.append({
        "fold": fold,
        "best_epoch": best_epoch,
        "best_val_loss": early.best_score,
        "model_path": best_model_path
    })

print("\n==================== 5-FOLD TRAINING COMPLETE ====================")
print(pd.DataFrame(fold_results))




fold_summary_df = pd.DataFrame(fold_results)

print("\nFold summary:")
print(fold_summary_df)

best_row = fold_summary_df.sort_values("best_val_loss").iloc[0]

BEST_FOLD = int(best_row["fold"])
BEST_MODEL_PATH = best_row["model_path"]

print("\nBest fold selected:", BEST_FOLD)
print("Best fold model:", BEST_MODEL_PATH)


GLOBAL_BEST_MODEL_PATH = os.path.join(
    MODEL_DIR, "esm_fp_best_5fold.pt"
)

best_ckpt = torch.load(BEST_MODEL_PATH, map_location=DEVICE)

torch.save({
    "selected_from_fold": BEST_FOLD,
    "epoch": best_row["best_epoch"],
    "val_loss": best_row["best_val_loss"],
    "model_state": best_ckpt["model_state"]
}, GLOBAL_BEST_MODEL_PATH)

print("\n✅ Global best model saved to:")
print(GLOBAL_BEST_MODEL_PATH)




from sklearn.metrics import roc_curve, auc, precision_recall_curve

# Load test CSV
test_csv = os.path.join(FOLD_DIR, "test.csv")
test_df = pd.read_csv(test_csv)

test_df["sample_id"] = test_df["sample_id"].astype(int)
test_df["label"] = test_df["label"].astype(int)


test_ds = CachedDTIDataset(test_df, fp_tensor, esm_tensor)
test_loader = torch.utils.data.DataLoader(
    test_ds, batch_size=64, shuffle=False,
    collate_fn=collate_cached, num_workers=0
)

model = DTIModelCached().to(DEVICE)
ckpt = torch.load(
    GLOBAL_BEST_MODEL_PATH,
    map_location=DEVICE,
    weights_only=False
)

model.load_state_dict(ckpt["model_state"])
model.eval()


y_true, y_prob = [], []

with torch.no_grad():
    for g, fp, esm, y in test_loader:
        g, fp, esm = g.to(DEVICE), fp.to(DEVICE), esm.to(DEVICE)
        out = model(g, fp, esm)
        prob = torch.sigmoid(out).cpu().numpy().ravel()

        y_prob.extend(prob)
        y_true.extend(y.numpy().ravel())

y_true = np.array(y_true)
y_prob = np.array(y_prob)


fpr, tpr, _ = roc_curve(y_true, y_prob)
roc_auc = auc(fpr, tpr)


precision, recall, _ = precision_recall_curve(y_true, y_prob)
pr_auc = auc(recall, precision)


import matplotlib.pyplot as plt

roc_path = os.path.join(FIGURE_DIR, "test_roc_curve.png")
plt.figure()
plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.3f}")
plt.plot([0, 1], [0, 1], "--")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve (Test Set)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(roc_path, dpi=200)
plt.close()

# Plot PR
pr_path = os.path.join(FIGURE_DIR, "test_pr_curve.png")
plt.figure()
plt.plot(recall, precision, label=f"AUC = {pr_auc:.3f}")
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Precision–Recall Curve (Test Set)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(pr_path, dpi=1000)
plt.close()

print("\n✅ Test evaluation completed")
print(f"ROC AUC: {roc_auc:.3f}")
print(f"PR AUC : {pr_auc:.3f}")
print("Saved:")
print(roc_path)
print(pr_path)




from sklearn.metrics import confusion_matrix


y_true, y_prob = [], []

model.eval()
with torch.no_grad():
    for g, fp, esm, y in test_loader:
        g, fp, esm = g.to(DEVICE), fp.to(DEVICE), esm.to(DEVICE)
        logits = model(g, fp, esm)
        probs = torch.sigmoid(logits).cpu().numpy().ravel()

        y_prob.extend(probs)
        y_true.extend(y.numpy().ravel())

y_true = np.array(y_true)
y_prob = np.array(y_prob)


THRESHOLD = 0.5
y_pred = (y_prob >= THRESHOLD).astype(int)


tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

# Metrics
accuracy    = (tp + tn) / (tp + tn + fp + fn)
sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
precision   = tp / (tp + fp) if (tp + fp) > 0 else 0.0
recall      = sensitivity
f1          = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

denom = np.sqrt((tp + fp)*(tp + fn)*(tn + fp)*(tn + fn))
mcc = (tp*tn - fp*fn) / denom if denom > 0 else 0.0


print("\n==================== TEST SET METRICS ====================")
print(f"Threshold          : {THRESHOLD}")
print(f"Accuracy           : {accuracy:.4f}")
print(f"Sensitivity (Recall): {sensitivity:.4f}")
print(f"Specificity        : {specificity:.4f}")
print(f"Precision          : {precision:.4f}")
print(f"F1-score           : {f1:.4f}")
print(f"MCC                : {mcc:.4f}")

print("\nConfusion Matrix [ [TN FP] [FN TP] ]")
print(np.array([[tn, fp],
                [fn, tp]]))








print("\n================ INDEPENDENT TEST RESULTS ================")
print(f"ROC AUC : {roc_auc:.4f}")
print(f"PR  AUC : {pr_auc:.4f}")
print(f"Accuracy: {accuracy:.4f}")
print(f"Sensitivity (Recall): {sensitivity:.4f}")
print(f"Specificity: {specificity:.4f}")
print(f"Precision : {precision_v:.4f}")
print(f"F1-score  : {f1:.4f}")
print(f"MCC       : {mcc:.4f}")

print("\nConfusion Matrix [[TN FP] [FN TP]]")
print(np.array([[tn, fp], [fn, tp]]))