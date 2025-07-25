from flask import send_file
import io
from flask_cors import CORS
from flask import Flask, request, jsonify ,render_template
import os
import uuid
import numpy as np
import torch
from speechbrain.inference import SpeakerRecognition
from scipy.spatial.distance import cosine
import soundfile as sf
from flask import send_from_directory
import matplotlib.pyplot as plt
from scipy.spatial.distance import cosine
from userdata import get_name_by_caller_id, get_status
import soundfile as sf

from speechbrain.pretrained import SpeakerRecognition

import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import numpy as np




import sys

app = Flask(__name__)
UPLOAD_FOLDER = 'tejdeep/voice_db'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Load pretrained speaker recognition model
spk_model = SpeakerRecognition.from_hparams(source="speechbrain/spkrec-ecapa-voxceleb")


def compute_embedding(filepath):
    signal, fs = sf.read(filepath)

    # Ensure it's a mono channel
    if len(signal.shape) > 1:
        signal = np.mean(signal, axis=1)

    # Convert to float32 to ensure model compatibility
    signal = torch.tensor(signal, dtype=torch.float32).unsqueeze(
        0)  # Shape: [1, time]

    with torch.no_grad():
        embedding = spk_model.encode_batch(signal)

    return embedding.squeeze().numpy()


# Simple in-memory storage for demo
embeddings_db = {}

@app.route('/')
def index():
    return render_template('uploadwav.html')

@app.route('/voice')
def voice_match_plot():
    # Simulate 10D embeddings
    np.random.seed(42)
    query_embedding = np.random.rand(10)

    enrolled_embeddings = {
        "Alice": query_embedding + np.random.normal(0, 0.01, 10),
        "Bob": query_embedding + np.random.normal(0, 0.1, 10),
        "Charlie": np.random.rand(10),
        "Diana": np.random.rand(10),
        "Eve": query_embedding + np.random.normal(0, 0.2, 10)
    }

    similarities = {name: 1 - cosine(query_embedding, emb) for name, emb in enrolled_embeddings.items()}

    all_embeddings = np.array([query_embedding] + list(enrolled_embeddings.values()))
    pca = PCA(n_components=2)
    reduced = pca.fit_transform(all_embeddings)

    # Create the plot
    plt.figure(figsize=(8, 6))
    plt.scatter(reduced[0, 0], reduced[0, 1], c='blue', label='Query', s=100, marker='X')

    for i, (name, score) in enumerate(similarities.items(), start=1):
        plt.scatter(reduced[i, 0], reduced[i, 1], label=f"{name} ({score:.2f})")
        plt.text(reduced[i, 0] + 0.01, reduced[i, 1] + 0.01, name, fontsize=9)

    plt.title("Voice Match: Query vs Enrolled Embeddings (PCA View)")
    plt.xlabel("PCA Component 1")
    plt.ylabel("PCA Component 2")
    plt.legend()
    plt.grid(True)

    # Save to BytesIO buffer
    img_bytes = io.BytesIO()
    plt.savefig(img_bytes, format='png')
    plt.close()
    img_bytes.seek(0)

    return send_file(img_bytes, mimetype='image/png')


@app.route('/cosine')
def show_plot():
    # Voice embeddings
    embedding_A = np.array([0.4, 0.9])
    embedding_B = np.array([0.8, 1.8])  # Same speaker (scaled A)
    embedding_C = np.array([-0.9, -0.4])  # Very different
    embedding_D = np.array([0.5, -1.0])   # Another different

    similarities = {
        "B (same speaker)": 1 - cosine(embedding_A, embedding_B),
        "C (diff speaker)": 1 - cosine(embedding_A, embedding_C),
        "D (diff speaker)": 1 - cosine(embedding_A, embedding_D)
    }

    vectors = [embedding_A, embedding_B, embedding_C, embedding_D]
    labels = ['A (query)', 'B', 'C', 'D']
    colors = ['blue', 'green', 'red', 'orange']

    plt.figure(figsize=(7, 7))
    for vec, label, color in zip(vectors, labels, colors):
        plt.quiver(0, 0, vec[0], vec[1], angles='xy', scale_units='xy', scale=1, color=color)
        plt.text(vec[0]*1.1, vec[1]*1.1, label, fontsize=10, color=color)

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

    # Save plot to a BytesIO object
    img_bytes = io.BytesIO()
    plt.savefig(img_bytes, format='png')
    plt.close()
    img_bytes.seek(0)

    return send_file(img_bytes, mimetype='image/png')

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
    try:
        file = request.files["file"]
        query_path = os.path.join(UPLOAD_FOLDER, f"query_{uuid.uuid4().hex}.wav")
        file.save(query_path)
        query_emb = compute_embedding(query_path)

        best_score = -1
        best_match = None

        for ref_file, ref_emb in embeddings_db.items():
            similarity = 1 - cosine(query_emb, ref_emb)
            if similarity > best_score:
                best_score = similarity
                best_match = ref_file

        if best_match is None:
            return jsonify({"error": "No enrolled voices to compare against."}), 400

        return jsonify({
            "match_file": best_match,
            "match_percentage": round(float(best_score) * 100, 2)

        })

    except Exception as e:
        print("🔥 Internal error in /match:", e)
        return jsonify({"error": str(e)}), 500

@app.route('/read-file', methods=['GET'])
def read_file():
    result = []
    try:
        with open("tejdeep\caller.txt", 'r') as file:
            lines = file.readlines()
            # Skip header row
            for line in lines[1:]:
                columns = line.strip().split()
                if len(columns) >= 9:
                     caller_id = columns[7]
                     timestamp = columns[8]
                     entry = {
                        "Name": get_name_by_caller_id(caller_id), 
                        "Phone no.": columns[2],
                        "Status": get_status(columns[5]),
                        "Location": columns[6],
                        "CallerID": columns[7],
                        "Duration": columns[8]
                    }
                     result.append(entry)
        return jsonify({"data": result})

    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500
        
@app.route('/get-audio-file', methods=['POST'])
def test_file_post():
    if not request.is_json:
        print("Invalid request: Expected JSON", file=sys.stderr)
        return {"error": "Expected JSON body"}, 400

    data = request.get_json()
    folder = data.get('folder')
    filename = data.get('filename')

    if not folder or not filename:
        print("Missing folder or filename", file=sys.stderr)
        return {"error": "Missing 'folder' or 'filename'"}, 400

    print(f"Serving file: {filename} from folder: {folder}", file=sys.stderr)

    try:
        return send_from_directory(folder, filename, as_attachment=True)
    except Exception as e:
        print(f"Error occurred: {e}", file=sys.stderr)
        return {"error": str(e)}, 500

if __name__ == "__main__":
#    http_thread = threading.Thread(target=run_http_server, daemon=True)
#    http_thread.start()
    CORS(app)
    app.run(debug=True, host="0.0.0.0")