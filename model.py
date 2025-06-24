import os
import numpy as np
import cv2
from tqdm import tqdm
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from datasets.arrow_dataset import Dataset as ArrowDataset
from datasets import concatenate_datasets
from glob import glob
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
import random
from datasets import Dataset
import pandas as pd
import joblib



# =============== CONFIG ===============
FAKE_ARROW_DIR = "/medias/db/ImagingSecurity_misc/sitcharn/paper_reproduction/cache/datasets/nebula___df-arrow/default/0.0.0/93117d58649bcf660f80fecf2122fac1f59d0453"
# REAL_DIR = "/medias/db/ImagingSecurity_misc/Collaborations/Hermes deepfake challenge/data/defacto/COCO/train2017"
REAL_DIR = "/medias/db/VoxCeleb2/alaska2"
SYNTHBUSTER_DIR = "/medias/db/ImagingSecurity_misc/sitcharn/paper_reproduction/dataset/resized_data_Synthbuster"
BLOCK_SIZE = 8
NUM_BLOCKS = 100
CHANNELS = ["Y", "Cb", "Cr"]
MAX_REAL_IMAGES = 100
FEATURE_SIZE = int((NUM_BLOCKS * (NUM_BLOCKS - 1) / 2) * len(CHANNELS))  # Corrélation triangulaire

# =============== FEATURE EXTRACTION ===============
def extract_noise_features(image_bytes, selected_channels=CHANNELS):
    try:
        print("Selected channels: ", selected_channels)
        img_array = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError("Image non décodable")

        img_ycc = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)

        channel_map = {
            "Y": img_ycc[:, :, 0],
            "Cb": img_ycc[:, :, 2],
            "Cr": img_ycc[:, :, 1]
        }

        features = []
        for ch in selected_channels:
            Ic = channel_map[ch].astype(np.float32)
            L4 = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=np.float32)
            Fc = cv2.filter2D(Ic, -1, L4)

            h, w = Fc.shape
            blocks = [Fc[i:i+BLOCK_SIZE, j:j+BLOCK_SIZE].flatten()
                      for i in range(0, h - BLOCK_SIZE + 1, BLOCK_SIZE)
                      for j in range(0, w - BLOCK_SIZE + 1, BLOCK_SIZE)]

            if len(blocks) == 0:
                continue

            while len(blocks) < NUM_BLOCKS:
                blocks += blocks  # réplication
            blocks = blocks[:NUM_BLOCKS]

            selected_blocks = np.stack(blocks, axis=1)
            Rc = np.corrcoef(selected_blocks)
            if np.isnan(Rc).any():
                continue

            tril_indices = np.tril_indices_from(Rc, k=-1)
            SRc = Rc[tril_indices]
            features.append(SRc)

        if not features:
            raise ValueError("Aucune feature extraite")

        full_feat = np.concatenate(features)

        if full_feat.shape[0] != FEATURE_SIZE:
            pad_width = FEATURE_SIZE - full_feat.shape[0]
            full_feat = np.pad(full_feat, (0, pad_width), mode='constant')

        return full_feat

    except Exception as e:
        raise ValueError(f"Erreur dans l'image : {e}")

# =============== TRAINING PIPELINE ===============
from sklearn.metrics import accuracy_score, classification_report, log_loss, roc_auc_score

# =============== TRAINING PIPELINE ===============
# =============== TRAINING PIPELINE ===============
from sklearn.metrics import classification_report, accuracy_score, log_loss, roc_auc_score

def train_classifiers(X, y, gen_labels):
    print("🔧 Initialisation de l'entraînement...")

    # Affichage du ratio fake/real
    unique, counts = np.unique(y, return_counts=True)
    counts_dict = dict(zip(unique, counts))
    total = sum(counts)
    print("\n📈 Répartition des classes :")
    for cls in sorted(counts_dict):
        pct = 100 * counts_dict[cls] / total
        print(f"  - {cls}: {counts_dict[cls]} ({pct:.2f}%)")

    label_enc = LabelEncoder()
    gen_indices = label_enc.fit_transform(gen_labels)
    N = len(np.unique(gen_indices))

    print(f"\n📦 Nombre de générateurs différents : {N}")
    print("🔄 Split des données pour les modèles...")
    X_train, X_test, gen_train, gen_test, y_train, y_test, gen_labels_train, gen_labels_test = train_test_split(
        X, gen_indices, y, gen_labels, test_size=0.3, random_state=42
    )

    # ========== Modèle f ==========
    print("🏋️‍♂️ Entraînement du modèle f (générateur)...")
    f_model = LogisticRegression(max_iter=1000)
    f_model.fit(X_train, gen_train)
    f_probs_train = f_model.predict_proba(X_train)
    f_probs_test = f_model.predict_proba(X_test)
    f_preds_test = f_model.predict(X_test)

    print("\n📊 Métriques du modèle f (multi-class générateur):")
    print(classification_report(gen_test, f_preds_test, target_names=label_enc.classes_))
    print(f"🎯 Accuracy f_model: {accuracy_score(gen_test, f_preds_test):.4f}")
    print(f"🔢 Log loss f_model: {log_loss(gen_test, f_probs_test):.4f}")

    # ========== Modèles g ==========
    g_models = []
    g_preds_train_all = []
    g_preds_test_all = []

    print("\n🏗 Entraînement des modèles g (par générateur)...")
    for i in range(N):
        print(f"  🔹 Modèle g pour le générateur '{label_enc.classes_[i]}'")

        gi_labels = np.array([(g == i or y[idx] == "real") for idx, g in enumerate(gen_indices)], dtype=int)

        # Split g identique à celui utilisé pour f
        y_train_g = gi_labels[:len(X_train)]
        y_test_g = gi_labels[len(X_train):]

        g_model = LogisticRegression(max_iter=1000)
        g_model.fit(X_train, y_train_g)
        preds = g_model.predict(X_test)
        probs = g_model.predict_proba(X_test)[:, 1]

        try:
            auc = roc_auc_score(y_test_g, probs)
        except ValueError:
            auc = float("nan")

        if len(np.unique(y_test_g)) > 1:
            print(f"    🎯 Accuracy: {accuracy_score(y_test_g, preds):.4f}")
            print(f"    🔢 Log loss: {log_loss(y_test_g, probs):.4f}")
            print(f"    🧠 AUC: {auc:.4f}")
        else:
            print("    ⚠️ Skipping log loss and AUC: only one class in y_test_g")

        print(f"    🧾 Report:\n{classification_report(y_test_g, preds)}")

       
        # print(f"    🧾 Report:\n{classification_report(y_test_g, preds)}")

        g_models.append(g_model)
        g_preds_train_all.append(g_model.predict_proba(X_train)[:, 1])
        g_preds_test_all.append(g_model.predict_proba(X_test)[:, 1])

    print(f"len(g_preds_test_all_OOD): {len(g_preds_test_all)}")
    for i, arr in enumerate(g_preds_test_all):
        print(f"g_preds_test_all_OOD[{i}].shape = {arr.shape}")

    # ========== Modèle h (binaire FAKE vs REAL) ==========
    print("\n🎯 Entraînement du modèle h (final FAKE vs REAL)...")

    # Données d'entraînement h
    final_train_input = np.concatenate([f_probs_train, np.stack(g_preds_train_all, axis=1)], axis=1)
    h_train_labels = (np.array(y_train) == "fake").astype(int)

    h_model = LogisticRegression(max_iter=1000)
    h_model.fit(final_train_input, h_train_labels)

    # Évaluation sur test
    final_test_input = np.concatenate([f_probs_test, np.stack(g_preds_test_all, axis=1)], axis=1)
    h_test_labels = (np.array(y_test) == "fake").astype(int)

    h_preds = h_model.predict(final_test_input)
    h_probs = h_model.predict_proba(final_test_input)[:, 1]

    print("\n📊 Métriques du modèle h (binaire FAKE vs REAL):")
    print(classification_report(h_test_labels, h_preds))
    print(f"🎯 Accuracy h_model: {accuracy_score(h_test_labels, h_preds):.4f}")
    print(f"🔢 Log loss h_model: {log_loss(h_test_labels, h_probs):.4f}")
    print(f"📈 AUC h_model: {roc_auc_score(h_test_labels, h_probs):.4f}")

    print("✅ Tous les modèles ont été entraînés avec succès.")

        # Créer un répertoire pour sauvegarder les modèles
    os.makedirs("models", exist_ok=True)

    # Sauvegarde du modèle f
    joblib.dump(f_model, "models/f_model.joblib")

    # Sauvegarde des modèles g
    for i, g_model in enumerate(g_models):
        joblib.dump(g_model, f"models/g_model_{i}_{label_enc.classes_[i]}.joblib")

    # Sauvegarde du modèle h
    joblib.dump(h_model, "models/h_model.joblib")

    # Sauvegarde de l'encodeur de labels (nécessaire pour prédire avec f_model)
    joblib.dump(label_enc, "models/label_encoder.joblib")

    print("💾 Modèles sauvegardés dans le dossier `models/`")

    return f_model, g_models, h_model, label_enc, final_test_input

# =============== MAIN PIPELINE ===============
if __name__ == "__main__":

    CHANNELS = ["Y", "Cb", "Cr"]
    MAX_REAL_IMAGES = 100
    FEATURE_SIZE = int((NUM_BLOCKS * (NUM_BLOCKS - 1) / 2) * len(CHANNELS))  # Corrélation triangulaire
    auc_results = []
    MAX_IMAGES = 10000
    auc_results_OOD = []
    acc_OOD = []

    X_test_OOD, y_test_OOD, generator_labels_OOD = [], [], []
    success_count_ood, fail_count_ood = 0, 0
    fake_ood = sorted(
            glob(os.path.join(SYNTHBUSTER_DIR, "**", "*.jpg"), recursive=True)
            + glob(os.path.join(SYNTHBUSTER_DIR, "**", "*.png"), recursive=True)
        )

    success_count_fake, fail_count = 0, 0

    for path in tqdm(fake_ood, desc="Extracting fake image features"):
        try:
            with open(path, "rb") as f:
                img_bytes = f.read()
                gen_name = os.path.basename(os.path.dirname(path))
                # print("gen_name: ", gen_name)
                feat = extract_noise_features(img_bytes, selected_channels=["Y"])
                X_test_OOD.append(feat)
                y_test_OOD.append("fake")
                generator_labels_OOD.append(gen_name)
                success_count_fake += 1
        except Exception:
            fail_count += 1
    
    N_OOD = len(np.unique(generator_labels_OOD))

    X_test_OOD = np.array(X_test_OOD, dtype=np.float32)
    y_test_OOD = np.array(y_test_OOD)

    print("X_test_OOD shape: ", X_test_OOD.shape)
    print("y_test_OOD shape: ", y_test_OOD.shape)

    print("For OOD fake:")
    print(f"✅ Total features extraites: {success_count_fake}")
    print(f"❌ Images ignorées: {fail_count}")
    print(f"📊 Total analysé: {success_count_fake + fail_count}")
    print(f"📦 Générateurs détectés: {set(generator_labels_OOD)}")

    for MAX_IMAGES in [40000]:
        auc_results = []
        # for CHANNELS in [["Y"], ["Cb"], ["Cr"], ["Y", "Cb"], ["Y", "Cr"], ["Cb", "Cr"], ["Y", "Cb", "Cr"]]:
        for CHANNELS in [["Cb", "Cr"]]:

            print("-----"*50)
            print(f"For channels: {CHANNELS}: ")
            arrow_files = sorted([
                os.path.join(FAKE_ARROW_DIR, f) for f in os.listdir(FAKE_ARROW_DIR)
                if f.startswith("df-arrow-test") and f.endswith(".arrow")
            ])
            fake_dataset = concatenate_datasets([ArrowDataset.from_file(f) for f in arrow_files])
            fake_dataset = fake_dataset.shuffle(seed=42)

            X, y, generator_labels = [], [], []
            success_count_fake, fail_count = 0, 0

            for sample in tqdm(fake_dataset, desc="Extracting fake image features"):
                try:
                    img_bytes = sample["image"]
                    path = sample["image_path"]
                    gen_name = path.split("/")[0]
                    feat = extract_noise_features(img_bytes, selected_channels=CHANNELS)
                    X.append(feat)
                    y.append("fake")
                    generator_labels.append(gen_name)
                    success_count_fake += 1
                except Exception:
                    fail_count += 1
                if success_count_fake >= MAX_IMAGES:
                    break
            
            real_image_paths = sorted(
                glob(os.path.join(REAL_DIR, "**", "*.jpg"), recursive=True)
                + glob(os.path.join(REAL_DIR, "**", "*.png"), recursive=True)
            )

            random.seed(42)  # ou une autre valeur fixe
            random.shuffle(real_image_paths)

            real_count = 0
            for path in tqdm(real_image_paths, desc="Extracting real image features"):
                try:
                    with open(path, "rb") as f:
                        img_bytes = f.read()
                        feat = extract_noise_features(img_bytes, selected_channels=CHANNELS)

                        if feat is None or len(feat.shape) != 1 or (len(X) > 0 and feat.shape[0] != X[0].shape[0]):
                            raise ValueError("Vecteur de features invalide ou incohérent")

                        X.append(feat)
                        y.append("real")
                        generator_labels.append("real")
                        # success_count += 1
                        real_count += 1
                        # if real_count >= MAX_REAL_IMAGES:
                        #     break
                        if real_count >= success_count_fake:
                            break
                except Exception as e:
                    print(f"Erreur pour image réelle {os.path.basename(path)}: {e}")
                    fail_count += 1

            print("For fake:")
            print(f"\n✅ Total features extraites: {success_count_fake}")
            print(f"❌ Images ignorées: {fail_count}")
            print(f"📊 Total analysé: {success_count_fake + fail_count}")
            print(f"📦 Générateurs détectés: {set(generator_labels)}")

            X = np.array(X, dtype=np.float32)
            y = np.array(y)
            print(f"X shape: {X.shape}")
            print(f"y shape: {y.shape}")

            print(f"\n✅ Total features extraites: {success_count_fake}")
            print(f"❌ Images ignorées: {fail_count}")
            print(f"📊 Total analysé: {success_count_fake + fail_count}")
            print(f"📦 Générateurs détectés: {set(generator_labels)}")

            generator_labels = np.array(generator_labels)

            f_model, g_models, h_model, label_en, final_test_input = train_classifiers(X, y, generator_labels)

            # Affichage du ratio fake/real
            unique, counts = np.unique(y, return_counts=True)
            counts_dict = dict(zip(unique, counts))
            total = sum(counts)
            print("\n📈 Répartition des classes :")
            for cls in sorted(counts_dict):
                pct = 100 * counts_dict[cls] / total
                print(f"  - {cls}: {counts_dict[cls]} ({pct:.2f}%)")

            label_enc = LabelEncoder()
            gen_indices = label_enc.fit_transform(generator_labels)
            N = len(np.unique(gen_indices))

            print(f"\n📦 Nombre de générateurs différents : {N}")
            print("🔄 Split des données pour les modèles...")
            X_train, X_test, gen_train, gen_test, y_train, y_test, gen_labels_train, gen_labels_test = train_test_split(
                X, gen_indices, y, generator_labels, test_size=0.3, random_state=42
            )

            # Ajouter ces lignes juste après l'entraînement et évaluation du modèle h :
            fpr, tpr, _ = roc_curve((y_test == "fake").astype(int), h_model.predict_proba(final_test_input)[:, 1])
            roc_auc = auc(fpr, tpr)
            auc_results.append({
                "channels": CHANNELS.copy(),
                "fpr": fpr,
                "tpr": tpr,
                "auc": roc_auc
            })

            f_probs_test_OOD = f_model.predict_proba(X_test_OOD)
            f_preds_test_OOD = f_model.predict(X_test_OOD)

            g_preds_test_all_OOD = []

            for i in range(N):
                g_preds_test_all_OOD.append(g_models[i].predict_proba(X_test_OOD)[:, 1])

            print(f"len(g_preds_test_all_OOD): {len(g_preds_test_all_OOD)}")
            for i, arr in enumerate(g_preds_test_all_OOD):
                print(f"g_preds_test_all_OOD[{i}].shape = {arr.shape}")
            
            final_test_input_OOD = np.concatenate([f_probs_test_OOD, np.stack(g_preds_test_all_OOD, axis=1)], axis=1)


            # Ajouter ces lignes juste après l'entraînement et évaluation du modèle h :

            fpr, tpr, _ = roc_curve((y_test_OOD == "fake").astype(int), h_model.predict_proba(final_test_input_OOD)[:, 1])
            roc_auc = auc(fpr, tpr)
            auc_results_OOD.append({
                "channels": CHANNELS.copy(),
                "fpr": fpr,
                "tpr": tpr,
                "auc": roc_auc
            })
            
            h_test_labels_OOD = (np.array(y_test_OOD) == "fake").astype(int)
            h_probs_OOD = h_model.predict_proba(final_test_input_OOD)[:, 1]
            h_preds_OOD = h_model.predict(final_test_input_OOD)

            acc_OOD_tmp = accuracy_score(h_test_labels_OOD, h_preds_OOD)
            print(f"📈 AUC h_model for OOD: {roc_auc_score(h_test_labels_OOD, h_probs_OOD):.4f}")
            print(f"🎯 Accuracy h_model: {acc_OOD_tmp}")
            acc_OOD.append({
                "MAX_IMAGES": MAX_IMAGES,
                "CHANNELS": "-".join(CHANNELS),  # pour une seule string type "Y-Cb-Cr"
                "ACC": acc_OOD_tmp
            })

            df = pd.DataFrame(acc_OOD)
            df.to_csv("acc_OOD_results.csv", index=False)
            print("✅ Résultats sauvegardés dans acc_OOD_results.csv")

            print("\n✅ Entraînement terminé")
            print("-----"*50)

        plt.figure(figsize=(10, 8))
        for result in auc_results:
            plt.plot(result["fpr"], result["tpr"], label=f"{'+'.join(result['channels'])} (AUC={result['auc']:.2f})")

        plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("ROC Curves par combinaison de canaux")
        plt.legend(loc="lower right")
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(f"ID-TEST_alaska_roc_curves_{MAX_IMAGES}_images.png")  # len(y) = nb total d'images
        plt.show()

        # roc curve on OOD 
        plt.figure(figsize=(10, 8))
        for result in auc_results_OOD:
            plt.plot(result["fpr"], result["tpr"], label=f"{'+'.join(result['channels'])} (AUC={result['auc']:.2f})")

        plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("ROC Curves par combinaison de canaux")
        plt.legend(loc="lower right")
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(f"OOD_alaska_roc_curves_{MAX_IMAGES}_images.png")  # len(y) = nb total d'images
        plt.show()

