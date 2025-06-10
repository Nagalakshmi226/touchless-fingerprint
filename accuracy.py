import numpy as math
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import normalize
import pickle

with open('/content/drive/MyDrive/fingerprint_embeds.pkl', 'rb') as f:
    data = pickle.load(f)

embeddings = normalize(data['vectors'])
labels = data['labels']

pairs = [(i, j) for i in range(len(labels)) for j in range(i+1, len(labels))]
similarity_scores = []
ground_truth = []

for idx1, idx2 in pairs:
    score = (embeddings[idx1] @ embeddings[idx2].T)
    similarity_scores.append(score)
    subject1 = labels[idx1].split('_')[0]
    subject2 = labels[idx2].split('_')[0]
    ground_truth.append(1 if subject1 == subject2 else 0)

roc_auc = roc_auc_score(ground_truth, similarity_scores)
print(f"ROC AUC Score: {roc_auc:.4f}")

THRESHOLD = 0.985
predictions = [1 if s >= THRESHOLD else 0 for s in similarity_scores]
accuracy = math.mean([p == t for p, t in zip(predictions, ground_truth)])
print(f"Accuracy at {THRESHOLD}: {accuracy * 100:.2f}%")
