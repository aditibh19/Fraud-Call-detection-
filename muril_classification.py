"""
Full training: MuRIL Hindi Classification (TensorFlow)
"""

import os
import joblib
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report, accuracy_score
from transformers import AutoTokenizer, TFAutoModelForSequenceClassification

# ----------------- CONFIG -----------------
DATA_PATH = "data/synthetic_calls_hindi_10000_like_seed.xlsx"
MODEL_DIR = "model/muril_classifier"
MODEL_NAME = "google/muril-base-cased"
MAX_LEN = 256
BATCH_SIZE = 16
EPOCHS = 5
RANDOM_STATE = 42
# ------------------------------------------

os.makedirs(MODEL_DIR, exist_ok=True)
tf.get_logger().setLevel("ERROR")

# ----------------- DATA CLEANING -----------------
def clean_text(text: str) -> str:
    text = str(text).replace("\n", " ").replace("\r", " ").strip()
    hindi_punct = "।,!?;:-()[]{}\"'“”"
    for p in hindi_punct:
        text = text.replace(p, " ")
    return " ".join(text.split())

def load_dataset(path=DATA_PATH, text_col="transcript_hin", label_col="label"):
    df = pd.read_excel(path)
    df = df[[text_col, label_col]].dropna()
    df.rename(columns={text_col: "text", label_col: "label"}, inplace=True)
    df["text"] = df["text"].apply(clean_text)
    return df

# ----------------- TOKENIZATION -----------------
def encode(tokenizer, texts):
    return tokenizer(
        texts,
        truncation=True,
        padding=True,
        max_length=MAX_LEN,
        return_tensors="tf"
    )

# ----------------- TRAIN & SAVE MODEL -----------------
def train_and_save(df):
    # Label encoding
    le = LabelEncoder()
    df["label_enc"] = le.fit_transform(df["label"])
    joblib.dump(le, f"{MODEL_DIR}/label_encoder.pkl")
    num_labels = len(le.classes_)

    # Stratified split
    X, y = df["text"].tolist(), df["label_enc"].tolist()
    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=RANDOM_STATE)
    for tr, ts in sss.split(X, y):
        X_train = [X[i] for i in tr]
        X_test = [X[i] for i in ts]
        y_train = [y[i] for i in tr]
        y_test = [y[i] for i in ts]

    # Tokenizer & Model
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = TFAutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=num_labels)

    # Encode
    train_enc = encode(tokenizer, X_train)
    test_enc = encode(tokenizer, X_test)

    y_train = tf.convert_to_tensor(y_train)
    y_test = tf.convert_to_tensor(y_test)

    # Class weights
    cw = compute_class_weight(class_weight="balanced", classes=np.unique(y_train.numpy()), y=y_train.numpy())
    class_weights = dict(enumerate(cw.astype("float64")))
    print("Class Weights:", class_weights)

    # Compile
    loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    optimizer = tf.keras.optimizers.Adam(learning_rate=3e-5)
    model.compile(optimizer=optimizer, loss=loss, metrics=["accuracy"])

    # Early stopping
    early_stop = tf.keras.callbacks.EarlyStopping(
        monitor='val_accuracy',
        patience=2,
        restore_best_weights=True
    )

    # Train
    model.fit(
        dict(train_enc),
        y_train,
        validation_data=(dict(test_enc), y_test),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        class_weight=class_weights,
        callbacks=[early_stop]
    )

    # Evaluation
    preds = np.argmax(model.predict(dict(test_enc)).logits, axis=1)
    print("\nClassification Report:\n", classification_report(y_test, preds, target_names=le.classes_))
    print("Accuracy:", accuracy_score(y_test, preds))

    # Save model & tokenizer
    model.save_pretrained(MODEL_DIR)
    tokenizer.save_pretrained(MODEL_DIR)
    print("✅ Model saved at:", MODEL_DIR)

# ----------------- LOAD MODEL -----------------
def load_model():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
    model = TFAutoModelForSequenceClassification.from_pretrained(MODEL_DIR)
    le = joblib.load(f"{MODEL_DIR}/label_encoder.pkl")
    return model, tokenizer, le

# ----------------- PREDICTION -----------------
def predict(model, tokenizer, le, text):
    enc = encode(tokenizer, [clean_text(text)])
    logits = model(enc).logits
    pred = int(tf.argmax(logits, axis=1).numpy()[0])
    return le.inverse_transform([pred])[0]

def test_samples(model, tokenizer, le):
    samples = [
        "sir recharge kab hoga",
        "account ka statement bhejo",
        "internet bohot slow hai",
        "OTP bataye warna account block ho jayega",
        "mera loan process batao"
    ]
    for s in samples:
        print(s, "→", predict(model, tokenizer, le, s))

# ----------------- MAIN -----------------
if __name__ == "__main__":
    model_files_exist = (
        os.path.exists(f"{MODEL_DIR}/config.json")
        and os.path.exists(f"{MODEL_DIR}/label_encoder.pkl")
    )

    if model_files_exist:
        print("✅ Loading saved model...")
        model, tokenizer, le = load_model()
    else:
        print("🚀 Training full model...")
        df = load_dataset()
        train_and_save(df)
        model, tokenizer, le = load_model()

    print("\n🔎 Testing classifier...\n")
    test_samples(model, tokenizer, le)
