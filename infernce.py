import os
import joblib
import warnings
warnings.filterwarnings("ignore")

import tensorflow as tf
from transformers import AutoTokenizer, TFAutoModelForSequenceClassification

# ----------------- CONFIG -----------------
MODEL_DIR = "model/muril_classifier"
MAX_LEN = 128  # for inference
# ------------------------------------------

# ----------------- LOAD MODEL & TOKENIZER -----------------
if not os.path.exists(MODEL_DIR):
    raise ValueError(f"Model directory '{MODEL_DIR}' does not exist. Please check the path.")

tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
model = TFAutoModelForSequenceClassification.from_pretrained(MODEL_DIR)
le = joblib.load(os.path.join(MODEL_DIR, "label_encoder.pkl"))

# ----------------- INFERENCE FUNCTION -----------------
def classify(text):
    """
    Predicts label for a single text input.
    Returns the string label and class probabilities.
    """
    # Split into sentences if it's a paragraph
    sentences = text.split(".")  # simple split, can be improved with nltk/sentencepiece
    final_probs = None
    count = 0  # number of valid sentences

    for sent in sentences:
        sent = sent.strip()
        if not sent:
            continue
        inputs = tokenizer(
            [sent],
            return_tensors="tf",
            truncation=True,
            padding="max_length",
            max_length=MAX_LEN
        )
        logits = model(inputs).logits
        probs = tf.nn.softmax(logits, axis=-1).numpy()[0]

        if final_probs is None:
            final_probs = probs
        else:
            final_probs += probs

        count += 1

    # Average if multiple sentences
    if final_probs is not None and count > 0:
        final_probs = final_probs / count
    else:
        final_probs = [0] * len(le.classes_)

    pred_idx = int(tf.argmax(final_probs))
    pred_label = le.inverse_transform([pred_idx])[0]
    return pred_label, final_probs

# ----------------- TEST ON CONVERSATION -----------------
conversation_samples = [
    "sir recharge kab hoga",
    "account ka statement bhejo",
    "internet bohot slow hai",
    "OTP bataye warna account block ho jayega",
    "mera loan process batao",
    "appointment reminder kal afternoon",
    "aapka package delivery ho gaya hai",
    """Customer Survey 2 Minutes
We’d love your feedback! Participation is optional, and no fee or OTP is required.""",
    ""
]

print("===== Conversation Test =====\n")
for text in conversation_samples:
    label, prob = classify(text)
    print(f"TEXT: {text}")
    print(f"PREDICTED: {label}, PROBABILITIES: {prob}")
    print("---")
