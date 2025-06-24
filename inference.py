import joblib
import time
from model import extract_noise_features
import numpy as np
from glob import glob
import os


REAL_DIR = "/medias/db/VoxCeleb2/alaska2"

# ========== 1. Charger les modèles ==========
f_model = joblib.load("models/f_model.joblib")
h_model = joblib.load("models/h_model.joblib")
label_enc = joblib.load("models/label_encoder.joblib")

g_models = []
for i, label in enumerate(label_enc.classes_):
    g_path = f"models/g_model_{i}_{label}.joblib"
    g_models.append(joblib.load(g_path))

# ========== 2. Chemin d'image de test ==========
real_image_paths = sorted(
                glob(os.path.join(REAL_DIR, "**", "*.jpg"), recursive=True)
                + glob(os.path.join(REAL_DIR, "**", "*.png"), recursive=True)
            )
image_path = real_image_paths[0]  # <-- à adapter

# ========== 3. Prédiction avec mesure du temps ==========
start_time = time.time()

with open(image_path, "rb") as f:
    img_bytes = f.read()

# Feature extraction
feat = extract_noise_features(img_bytes, selected_channels=["Y", "Cb", "Cr"])
feat = feat.reshape(1, -1)  # (1, FEATURE_SIZE)

# f_model
f_probs = f_model.predict_proba(feat)

# g_models
g_preds = [g.predict_proba(feat)[:, 1] for g in g_models]
g_preds_stack = np.array(g_preds).reshape(1, -1)

# concaténer les features pour h_model
final_input = np.concatenate([f_probs, g_preds_stack], axis=1)

# prédiction finale
prob_fake = h_model.predict_proba(final_input)[0, 1]
label = h_model.predict(final_input)[0]
pred_label = "fake" if label == 1 else "real"

elapsed_time = time.time() - start_time

# ========== 4. Résultat ==========
print(f"🖼 Image testée : {os.path.basename(image_path)}")
print(f"🕒 Temps total de prédiction : {elapsed_time:.4f} s")
print(f"📊 Probabilité d'être fake : {prob_fake:.4f}")
print(f"🔍 Prédiction finale : {pred_label}")
