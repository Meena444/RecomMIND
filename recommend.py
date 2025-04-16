import torch as th
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import json
from model import MIND
from utils import padOrCut, load_metadata

# -------- Load Args and Data --------
class Args:
    seq_len = 5 #sequence length of history of user (last 5 items)
    D = 8 # Embedding dimensions for output
    K = 3 # No of interest capsules
    R = 3 # Routing interactions
    n_neg = 2
    lr = 0.001

args = Args()

print("Loading data and model...")

# Load ratings
ratings = pd.read_csv("data/Appliances.csv", header=None, names=['userId', 'itemId', 'rate', 'timestamp'])

# Load metadata for additional info about the dataset (cold-start recommendations)
with open("data/meta_Appliances.json") as f:
    metadata = [json.loads(line) for line in f]
meta_df = load_metadata(metadata)

# Encode user and item
unique_users = ratings['userId'].unique()
userEncId = {rawId: encId for encId, rawId in enumerate(unique_users)}
userRawId = {v: k for k, v in userEncId.items()}

unique_items = ratings['itemId'].unique()
itemEncId = {rawId: encId + 1 for encId, rawId in enumerate(unique_items)}
itemRawId = {v: k for k, v in itemEncId.items()}

ratings['userId'] = ratings['userId'].map(userEncId)
ratings['itemId'] = ratings['itemId'].map(itemEncId)

# Create embedding model
embedNum = len(itemEncId) + 1
model = MIND(args, embedNum)

# Load model weights
# Load checkpoint
checkpoint = th.load("model.pth")

# Only load layers that match
model_state_dict = model.state_dict()

# Filter the keys that exist in both the checkpoint and model state_dict
matched_state_dict = {k: v for k, v in checkpoint.items() if k in model_state_dict and v.shape == model_state_dict[k].shape}

# Update the model with the matched state_dict
model_state_dict.update(matched_state_dict)

# Load the updated state_dict
model.load_state_dict(model_state_dict)

# Now you can proceed with evaluation
model.eval()


# Get all item embeddings
itemEmbeds = model.itemEmbeds.weight.detach()  # shape (V, D)

# -------- Choose a User --------
user_id = int(input(f"Enter a user ID (encoded, 0 to {len(userEncId)-1}): "))
user_df = ratings[ratings['userId'] == user_id]
user_his = padOrCut(user_df['itemId'].values[:-1], args.seq_len)
true_next = user_df['itemId'].values[-1]

his_tensor = th.tensor(user_his).unsqueeze(0)  # shape: (1, L)
caps, coupling = model.B2IRouting_with_coupling(his_tensor, bs=1)

# -------- Compute Top-10 Scores --------
logits = th.matmul(caps, itemEmbeds.T)  # (1, K, V)
logits = logits.view(-1).detach().numpy()
top_indices = np.argpartition(logits, -10)[-10:]
top_scores = logits[top_indices]
top_sorted = top_indices[np.argsort(-top_scores)]

print("\nTop 10 recommended items:")
for idx, itemIdx in enumerate(top_sorted):
    rawId = itemRawId.get(itemIdx, "N/A")
    info = meta_df[meta_df['asin'] == rawId]
    title = info['brand'].values[0] if not info.empty else "Unknown"
    category = info['category'].values[0] if not info.empty else "Unknown"
    print(f"{idx + 1:2d}. Item ID: {rawId} | Brand: {title} | Category: {category}")

# -------- Visualization 1: Coupling Heatmap --------
plt.figure(figsize=(10, 6))
sns.heatmap(coupling.squeeze(0).detach().numpy(), cmap="YlGnBu", annot=True)
plt.title(f"Coupling Coefficients (User {user_id})")
plt.xlabel("Behavior History Index")
plt.ylabel("Interest Capsule")
plt.tight_layout()
plt.show(block=True)

# -------- Visualization 2: Interest to Item Distribution --------
top_k_per_cap = th.topk(th.matmul(caps.squeeze(0), itemEmbeds.T), 10, dim=1)
caps_items = top_k_per_cap.indices.detach().numpy()
caps_scores = top_k_per_cap.values.detach().numpy()

plt.figure(figsize=(12, 5))
for k in range(args.K):
    plt.subplot(1, args.K, k + 1)
    sim_scores = caps_scores[k]
    items = [itemRawId.get(i, "N/A") for i in caps_items[k]]
    sns.barplot(x=sim_scores, y=items)
    plt.title(f"Interest Capsule {k}")
    plt.xlabel("Similarity")
plt.tight_layout()
plt.show(block=True)
