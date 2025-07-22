from flask import Flask, request, jsonify
import os
import uuid
import numpy as np
from speechbrain.pretrained import SpeakerRecognition
from scipy.spatial.distance import cosine
import soundfile as sf

app = Flask(__name__)
UPLOAD_FOLDER = 'voice_db'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Load pretrained speaker recognition model
spk_model = SpeakerRecognition.from_hparams(source="speechbrain/spkrec-ecapa-voxceleb")

def compute_embedding(filepath):
    signal, fs = sf.read(filepath)
    embedding = spk_model.encode_batch(signal=torch.tensor([signal]))
    return embedding.squeeze().numpy()

# Simple in-memory storage for demo
embeddings_db = {}

@app.route("/enroll", methods=["POST"])
def enroll_voice():
    """Add a new voice sample to the DB"""
    file = request.files["file"]
    name = request.form["name"]
    filename = os.path.join(UPLOAD_FOLDER, f"{name}_{uuid.uuid4().hex}.wav")
    file.save(filename)
    emb = compute_embedding(filename)
    embeddings_db[filename] = emb
    return jsonify({"message": "Voice enrolled", "file": filename})

@app.route("/match", methods=["POST"])
def match_voice():
    """Compare a new voice sample to all enrolled voices"""
    file = request.files["file"]
    query_path = os.path.join(UPLOAD_FOLDER, f"query_{uuid.uuid4().hex}.wav")
    file.save(query_path)
    query_emb = compute_embedding(query_path)

    best_score = -1
    best_match = None

    for ref_file, ref_emb in embeddings_db.items():
        similarity = 1 - cosine(query_emb, ref_emb)  # Cosine similarity
        if similarity > best_score:
            best_score = similarity
            best_match = ref_file

    return jsonify({
        "match_file": best_match,
        "match_percentage": round(best_score * 100, 2)
    })

if __name__ == "__main__":
    app.run(debug=True)
