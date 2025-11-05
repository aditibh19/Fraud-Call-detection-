import warnings
warnings.filterwarnings("ignore")

from transformers import AutoTokenizer, TFAutoModelForSequenceClassification
import tensorflow as tf
import joblib

# ===== LOAD MODEL AND TOKENIZER =====
MODEL_DIR = "model/muril_call_classifier"
tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
model = TFAutoModelForSequenceClassification.from_pretrained(MODEL_DIR)

# ===== LOAD LABEL ENCODER =====
label_encoder = joblib.load(f"{MODEL_DIR}/label_encoder.pkl")

# ===== CLASSIFY FUNCTION =====
def classify(text):
    """Return predicted label for a given text"""
    inputs = tokenizer(
        [text],
        return_tensors="tf",
        truncation=True,
        padding="max_length",
        max_length=128
    )
    outputs = model(**inputs)
    probs = tf.nn.softmax(outputs.logits, axis=-1).numpy()[0]
    pred_label = label_encoder.inverse_transform([tf.argmax(probs).numpy()])[0]
    return pred_label

# ===== ENGLISH TEST CASES =====
english_test = [
    ("We detected unusual login, please verify OTP", "fraud"),
    ("Your package will be delivered today", "benign"),
    ("Security reminder: we are updating information about your plan. Use only the official channel.", "benign"),
    ("Urgent: your account has been compromised, provide OTP immediately", "fraud"),
    ("Your subscription has been renewed successfully", "benign")
]

# ===== EVALUATE =====
total = len(english_test)
correct = 0

print("\n==== ENGLISH TEXT EVALUATION ====\n")
for text, expected in english_test:
    pred = classify(text)
    if pred == expected:
        correct += 1
        status = "✅ CORRECT"
    else:
        status = "❌ WRONG"
    print(f"{status}\nText: {text}\nExpected: {expected}, Predicted: {pred}\n")

accuracy = correct / total * 100
print(f"✅ English Accuracy: {correct}/{total} = {accuracy:.2f}%")
