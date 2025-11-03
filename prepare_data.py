import warnings
warnings.filterwarnings("ignore")

from transformers import AutoTokenizer, TFAutoModelForSequenceClassification
import tensorflow as tf


# ========== LOAD MODEL ==========
MODEL_DIR = "model/muril_call_classifier"

tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
model = TFAutoModelForSequenceClassification.from_pretrained(MODEL_DIR)


# ========== CLASSIFICATION FUNCTION ==========
def classify(text):
    if not text or text.strip() == "":
        return "Invalid Input: Text is empty!"
    
    inputs = tokenizer(
        [text],
        return_tensors="tf",
        truncation=True,
        padding="max_length",
        max_length=128
    )

    outputs = model(**inputs)
    probs = tf.nn.softmax(outputs.logits, axis=-1).numpy()[0]

    fraud_prob = float(probs[0])   # label 0 = fraud
    benign_prob = float(probs[1])

    print(f" → Probabilities => Fraud: {fraud_prob:.4f}, Benign: {benign_prob:.4f}")

    if fraud_prob >= 0.80:
        return "fraud"
    return "benign"


# ========== TEST CASES ==========
test_cases = {
    "Hinglish": [
        "Bhai OTP bhej, account verify karna hai",
        "Kal delivery boy ayega, call pick kar lena"
    ],

    "Roman Hindi": [
        "Aapka bank account band ho jayega OTP do",
        "Aapka parcel aaj pohonch jayega"
    ],

    "Devanagari (Hindi Script)": [
        "हम आपके बैंक से बात कर रहे हैं, तुरंत OTP बताइए",
        "आपका पार्सल आज शाम तक पहुँच जाएगा"
    ],

    "English": [
        "We detected unusual login, please verify OTP",
        "Your package will be delivered today"
    ]
}


# ========== RUN EVALUATION ==========
print("\n========== MODEL EVALUATION ==========\n")

for category, sentences in test_cases.items():
    print(f"\n---- {category.upper()} ----\n")
    for text in sentences:
        print(f"TEXT: {text}")
        result = classify(text)
        print(f"Prediction: {result}\n")
