import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import cosine

# Simulated voice embeddings for demonstration (normally these would be 192-d or 512-d)
embedding_A = np.array([0.4, 0.9])
embedding_B = np.array([0.8, 1.8])  # Same speaker (scaled version of A)
embedding_C = np.array([-0.9, -0.4])  # Very different speaker
embedding_D = np.array([0.5, -1.0])   # Another different speaker

# Calculate cosine similarities with embedding_A as the query
similarities = {
    "B (same speaker)": 1 - cosine(embedding_A, embedding_B),
    "C (diff speaker)": 1 - cosine(embedding_A, embedding_C),
    "D (diff speaker)": 1 - cosine(embedding_A, embedding_D)
}

# Plot embeddings and their cosine similarity values
vectors = [embedding_A, embedding_B, embedding_C, embedding_D]
labels = ['A (query)', 'B', 'C', 'D']
colors = ['blue', 'green', 'red', 'orange']

plt.figure(figsize=(7, 7))
for vec, label, color in zip(vectors, labels, colors):
    plt.quiver(0, 0, vec[0], vec[1], angles='xy', scale_units='xy', scale=1, color=color)
    plt.text(vec[0]*1.1, vec[1]*1.1, label, fontsize=10, color=color)

# Annotate similarity values
y_offset = -0.3
for label, score in similarities.items():
    plt.text(-2, y_offset, f"Similarity with A & {label}: {score:.2f}", fontsize=10)
    y_offset -= 0.3

plt.xlim(-2, 2)
plt.ylim(-2, 2)
plt.axhline(0, color='gray', linewidth=0.5)
plt.axvline(0, color='gray', linewidth=0.5)
plt.grid(True)
plt.gca().set_aspect('equal')
plt.title("Voice Matching via Cosine Similarity (2D Demo)")
plt.xlabel("Embedding Dimension 1")
plt.ylabel("Embedding Dimension 2")
plt.show()