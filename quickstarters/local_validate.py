# python
# filepath: /workspace/local_validate.py
# ...existing code...
# Quick holdout validation using ids from X_train (works locally)
from sklearn.metrics import roc_auc_score
import random

# get list of ids and split (stratify by y_train)
ids = y_train.index.tolist()
random.seed(42)
# simple stratified-ish split: pick 20% ids as holdout
unique_ids = list(y_train.index)
n_hold = max(1, int(0.2 * len(unique_ids)))
hold_ids = set(random.sample(unique_ids, n_hold))
train_ids = [i for i in unique_ids if i not in hold_ids]

# build X_train_small, y_train_small for training function (they expect the full X_train structure)
X_train_small = X_train.loc[train_ids]
y_train_small = y_train.loc[train_ids]

X_val = [g for gid, g in X_train.groupby(level="id") if gid in hold_ids]
val_ids = [gid for gid, g in X_train.groupby(level="id") if gid in hold_ids]
y_val = y_train.loc[val_ids].astype(int)

# train and infer
model_dir = "model_local"
train(X_train_small, y_train_small, model_dir)

gen = infer(X_val, model_dir)
next(gen)
preds = [p for p in gen]

# compute AUC
auc = roc_auc_score(y_val.values, preds)
print("Local holdout ROC AUC:", auc)
# ...existing code...