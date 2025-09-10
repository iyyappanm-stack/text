import os
import numpy as np
import pandas as pd
import librosa
import tensorflow as tf
from glob import glob
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from scipy.stats import pearsonr

# =====================================
# 1. GPU Check
# =====================================
print("🔍 Checking GPU availability...")
gpus = tf.config.list_physical_devices("GPU")
if gpus:
    print(f"✅ GPU available: {gpus}")
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)  # prevent OOM
else:
    print("⚠️ No GPU detected, training will run on CPU!")

# =====================================
# 2. Preprocess single audio
# =====================================
def process_audio(file_path, score, chunk_duration=4, overlap_duration=2, n_mels=128):
    """Returns list of (mel_spec, score) pairs for one audio file"""
    try:
        audio_data, sample_rate = librosa.load(file_path, sr=None)
    except Exception:
        return []

    chunk_samples = chunk_duration * sample_rate
    overlap_samples = overlap_duration * sample_rate
    num_chunks = int(np.ceil((len(audio_data) - chunk_samples) /
                             (chunk_samples - overlap_samples))) + 1

    samples = []
    for i in range(num_chunks):
        start = i * (chunk_samples - overlap_samples)
        end = start + chunk_samples
        chunk = audio_data[start:end]

        if len(chunk) < chunk_samples:
            chunk = np.pad(chunk, (0, chunk_samples - len(chunk)))

        mel_spec = librosa.feature.melspectrogram(y=chunk, sr=sample_rate, n_mels=n_mels)
        mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
        mel_spec_db = mel_spec_db.astype(np.float32)

        samples.append((mel_spec_db, score))
    return samples

# =====================================
# 3. Build dataset
# =====================================
def build_dataset(data_dir, df):
    X, Y = [], []
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="📂 Preparing dataset"):
        base_id = row["call_id"].replace(".wav", "")
        score = float(row["Pronunciation"])
        matching_files = glob(os.path.join(data_dir, f"{base_id}_*.wav"))
        for file_path in matching_files:
            samples = process_audio(file_path, score)
            for mel, label in samples:
                X.append(mel)
                Y.append(label)
    return np.array(X), np.array(Y, dtype=np.float32)

# =====================================
# 4. Convert to tf.data
# =====================================
def make_tf_dataset(X, Y, target_shape=(64, 64), batch_size=16, shuffle=True):
    ds = tf.data.Dataset.from_tensor_slices((X, Y))

    def _resize(x, y):
        x = tf.expand_dims(x, -1)  # (n_mels, time, 1)
        x = tf.image.resize(x, target_shape)  # GPU-based resize
        return x, y

    if shuffle:
        ds = ds.shuffle(buffer_size=len(X))
    ds = ds.map(_resize, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return ds

# =====================================
# 5. Load Data
# =====================================
data_dir = "/home/intel/projects/data/audio"
csv_file = "/home/intel/projects/data/new_label.csv"
df = pd.read_csv(csv_file)

X, Y = build_dataset(data_dir, df)


print(f"✅ Dataset ready: {X.shape}, {Y.shape}")

X_train, X_val, Y_train, Y_val = train_test_split(X, Y, test_size=0.2, random_state=42)
train_ds = make_tf_dataset(X_train, Y_train, batch_size=16)
val_ds = make_tf_dataset(X_val, Y_val, batch_size=16, shuffle=False)

print(f"✅ Train set: {len(X_train)}, Val set: {len(X_val)}")

# =====================================
# 6. Model Definition
# =====================================
def build_model(input_shape=(64, 64, 1)):
    model = tf.keras.Sequential([
        tf.keras.layers.Conv2D(32, (3, 3), activation="relu", input_shape=input_shape),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Conv2D(64, (3, 3), activation="relu"),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation="relu"),
        tf.keras.layers.Dense(1, activation="linear")
    ])
    return model

model_reg = build_model()
model_reg.compile(optimizer="adam", loss="mse", metrics=["mae"])
model_reg.summary()

# =====================================
# 7. Training
# =====================================
callbacks = [
    tf.keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True),
    tf.keras.callbacks.ReduceLROnPlateau(patience=5),
    tf.keras.callbacks.ModelCheckpoint("best_model.keras", save_best_only=True)
]

print("🚀 Starting training...")
history = model_reg.fit(
    train_ds,
    validation_data=val_ds,
    epochs=50,
    callbacks=callbacks,
    verbose=1
)

print("🎉 Training finished!")

# Save final model
model_reg.save("final_model_1.h5")
print("💾 Model saved as final_model.h5")

# =====================================
# 8. Plot training history
# =====================================
plt.figure(figsize=(12,5))
plt.subplot(1,2,1)
plt.plot(history.history["loss"], label="Train Loss")
plt.plot(history.history["val_loss"], label="Val Loss")
plt.xlabel("Epochs")
plt.ylabel("MSE Loss")
plt.title("Loss Curve")
plt.legend()

plt.subplot(1,2,2)
plt.plot(history.history["mae"], label="Train MAE")
plt.plot(history.history["val_mae"], label="Val MAE")
plt.xlabel("Epochs")
plt.ylabel("MAE")
plt.title("MAE Curve")
plt.legend()
plt.show()

# =====================================
# 9. Evaluation
# =====================================
def evaluate_regression_model(model, X_test, Y_test, model_name):
    print(f"\n===== {model_name} RESULTS =====\n")

    test_ds = make_tf_dataset(X_test, Y_test, batch_size=16, shuffle=False)

    test_loss, test_mae = model.evaluate(test_ds, verbose=0)
    print(f"📉 Test - Loss (MSE): {test_loss:.4f} | MAE: {test_mae:.4f}")

    y_pred = model.predict(test_ds).flatten()
    y_true = Y_test.flatten()

    corr, p_value = pearsonr(y_true, y_pred)
    print(f"📊 Pearson correlation: {corr:.4f} (p={p_value:.4e})")

    mse = mean_squared_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    print(f"📊 MSE: {mse:.4f} | R²: {r2:.4f}")

    plt.figure(figsize=(6,6))
    plt.scatter(y_true, y_pred, alpha=0.6)
    plt.xlabel("True Pronunciation Scores")
    plt.ylabel("Predicted Scores")
    plt.title(f"{model_name} Predictions vs True")
    plt.plot([y_true.min(), y_true.max()],
             [y_true.min(), y_true.max()], 'r--')
    plt.show()

# Run evaluation
evaluate_regression_model(model_reg, X_val, Y_val, "Pronunciation CNN Regression")
