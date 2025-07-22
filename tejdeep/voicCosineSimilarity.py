import soundfile as sf
from scipy.spatial.distance import cosine
from flask import Flask, request, jsonify
from flask_cors import CORS
from speechbrain.pretrained import SpeakerRecognition

import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import numpy as np

# Simulate actual voice embeddings (normally these are 192-512D, we'll simulate with 10D for simplicity)
np.random.seed(42)
query_embedding = np.random.rand(10)

# Simulate 5 enrolled speaker embeddings with varying closeness
enrolled_embeddings = {
    "Alice": query_embedding + np.random.normal(0, 0.01, 10),  # very close
    "Bob": query_embedding + np.random.normal(0, 0.1, 10),     # close-ish
    "Charlie": np.random.rand(10),                             # unrelated
    "Diana": np.random.rand(10),                               # unrelated
    "Eve": query_embedding + np.random.normal(0, 0.2, 10)      # further
}

# Compute cosine similarities
similarities = {name: 1 - cosine(query_embedding, emb) for name, emb in enrolled_embeddings.items()}

# Prepare embeddings for visualization using PCA (reduce to 2D)
all_embeddings = np.array([query_embedding] + list(enrolled_embeddings.values()))
pca = PCA(n_components=2)
reduced = pca.fit_transform(all_embeddings)

# Plot
plt.figure(figsize=(8, 6))
plt.scatter(reduced[0, 0], reduced[0, 1], c='blue', label='Query', s=100, marker='X')

for i, (name, score) in enumerate(similarities.items(), start=1):
    plt.scatter(reduced[i, 0], reduced[i, 1], label=f"{name} ({score:.2f})")
    plt.text(reduced[i, 0]+0.01, reduced[i, 1]+0.01, name, fontsize=9)

plt.title("Voice Match: Query vs Enrolled Embeddings (PCA View)")
plt.xlabel("PCA Component 1")
plt.ylabel("PCA Component 2")
plt.legend()
plt.grid(True)
plt.show()
