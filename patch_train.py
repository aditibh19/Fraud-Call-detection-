import pandas as pd
from transformers import AutoTokenizer, TFAutoModelForSequenceClassification
import tensorflow as tf

MODEL_PATH = "model/muril_call_classifier"   # your trained model folder

# ✅ Load benign-only CSV
df = pd.read_csv("benign_extra.csv")
df = df.rename(columns={df.columns[0]: "text"})
df["label"] = 0   # ✅ benign = 0

print("✅ Benign patch dataset loaded:", df.shape)
print(df.head())

# ✅ Load tokenizer & model
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = TFAutoModelForSequenceClassification.from_pretrained(MODEL_PATH)

# ✅ Tokenize
train_encodings = tokenizer(
    df["text"].tolist(),
    truncation=True,
    padding=True,
    max_length=64
)

labels = tf.convert_to_tensor(df["label"].values, dtype=tf.int32)

train_dataset = tf.data.Dataset.from_tensor_slices((
    dict(train_encodings),
    labels
)).batch(8)

optimizer = tf.keras.optimizers.Adam(learning_rate=2e-5)
loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

model.compile(optimizer=optimizer, loss=loss, metrics=["accuracy"])

# ✅ Train only a little — patch training
model.fit(train_dataset, epochs=2)

# ✅ Save updated model
model.save_pretrained(MODEL_PATH)
tokenizer.save_pretrained(MODEL_PATH)

print("✅ Patch training finished!")
print("⚡ Updated model saved at:", MODEL_PATH)
