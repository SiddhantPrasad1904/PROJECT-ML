import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertForSequenceClassification , Trainer, TrainingArguments
from datasets import load_dataset
import os
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report, confusion_matrix
import numpy as np
import xgboost as xgb
import matplotlib.pyplot as plt
import seaborn as sns

# Download if not already installed
"""nltk.download("stopwords")
nltk.download("punkt")
nltk.download("wordnet")"""

# Load the dataset
df = pd.read_csv("D:\Siddhant\c++\c++ course\ML lab\PROJECT-ML\labeled_data.csv")

# Drop unnecessary columns
df = df.drop(columns=["Unnamed: 0", "count"])

# Check unique class labels
print("Unique class labels:", df["class"].unique())

# Adjust num_labels based on unique labels
num_labels = len(df["class"].unique())

# Initialize NLP tools
stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()

# Function for text cleaning
def clean_text(text):
    text = text.lower()
    text = re.sub(r"http\S+|www\S+", "", text)
    text = re.sub(r"[^a-zA-Z*#@\s]", "", text)
    words = word_tokenize(text)
    words = [lemmatizer.lemmatize(word) for word in words if word not in stop_words]
    return " ".join(words)

# Apply text cleaning
df["cleaned_tweet"] = df["tweet"].apply(clean_text)

# Load BERT tokenizer
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

# Check if a saved model exists
model_path = "./saved_model_v6"
if os.path.exists(model_path):
    print("Loading saved model...")
    model = BertForSequenceClassification.from_pretrained(model_path)
    tokenizer = BertTokenizer.from_pretrained(model_path)
    train_model = False
else:
    print("Training new model...")
    model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=num_labels)
    train_model = True

# Move model to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
model.to(device)

# Training arguments
training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    save_strategy="epoch",
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=3,
    weight_decay=0.01,
)

def tokenize_data(texts, labels, tokenizer, max_length=128):
    encodings = tokenizer(list(texts), truncation=True, padding=True, max_length=max_length, return_tensors="pt")
    return encodings["input_ids"].cpu(), encodings["attention_mask"].cpu(), torch.tensor(labels.values).cpu()

# Tokenize dataset
X_input_ids, X_attention_mask, y_labels = tokenize_data(df["cleaned_tweet"], df["class"], tokenizer)

class HateSpeechDataset(Dataset):
    def __init__(self, input_ids, attention_mask, labels):
        self.input_ids = input_ids
        self.attention_mask = attention_mask
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return {
            "input_ids": self.input_ids[idx],
            "attention_mask": self.attention_mask[idx],
            "labels": self.labels[idx]
        }

# Create dataset
dataset = HateSpeechDataset(X_input_ids, X_attention_mask, y_labels)

# Split into training and test sets
train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size
train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])

# Define Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset
)

# Train the model only if necessary
if train_model:
    trainer.train()
    model.save_pretrained(model_path)
    tokenizer.save_pretrained(model_path)
    print("Model saved successfully!")
else:
    print("Skipping training as a saved model is loaded.")

# Evaluate model
def evaluate_model(trainer, test_dataset):
    predictions = trainer.predict(test_dataset)
    preds = torch.argmax(torch.tensor(predictions.predictions), dim=1).numpy()
    labels = predictions.label_ids
    accuracy = accuracy_score(labels, preds)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average="weighted")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-score: {f1:.4f}")
    print("Classification Report:\n", classification_report(labels, preds))

    # Confusion matrix
    cm = confusion_matrix(labels, preds)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")
    plt.show()

evaluate_model(trainer, test_dataset)

# Extract BERT embeddings for XGBoost
model.eval()
def extract_embeddings(texts, tokenizer, model, device, batch_size=16):
    embeddings = []
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i+batch_size].tolist()
        inputs = tokenizer(batch_texts, padding=True, truncation=True, max_length=128, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = model.bert(**inputs)
        embeddings.append(outputs.last_hidden_state[:, 0, :].cpu().numpy())
    return np.vstack(embeddings)

X_train_texts = df["cleaned_tweet"].iloc[:train_size]
X_test_texts = df["cleaned_tweet"].iloc[train_size:]

original_train_tweets = df["tweet"].iloc[:train_size].reset_index(drop=True)
original_test_tweets = df["tweet"].iloc[train_size:].reset_index(drop=True)

y_train = y_labels[:train_size].numpy()
y_test = y_labels[train_size:].numpy()

X_train_embeddings = extract_embeddings(X_train_texts, tokenizer, model, device)
X_test_embeddings = extract_embeddings(X_test_texts, tokenizer, model, device)

xgb_clf = xgb.XGBClassifier(n_estimators=500, max_depth=10, learning_rate=0.03, subsample=0.9)
xgb_clf.fit(X_train_embeddings, y_train)

xgb_preds = xgb_clf.predict(X_test_embeddings)

print("XGBoost Accuracy:", accuracy_score(y_test, xgb_preds))
print("Classification Report:\n", classification_report(y_test, xgb_preds))

'''
# Show predictions
print("\n--- Sample Predictions ---")

print("Correctly Classified:")
correct_indices = np.where(y_test == xgb_preds)[0]
for idx in correct_indices[:5]:
    print(f"[Class {y_test[idx]}] {original_test_tweets.iloc[idx]}")
    print(f"Predicted: {xgb_preds[idx]}\n")

print("Incorrectly Classified:")
wrong_indices = np.where(y_test != xgb_preds)[0]
for idx in wrong_indices[:5]:
    print(f"[Actual: {y_test[idx]} | Predicted: {xgb_preds[idx]}]")
    print(original_test_tweets.iloc[idx])
    print()'''
'''
# --- Plot Training Loss and Accuracy (if training was done) ---
if train_model:
    training_history = trainer.state.log_history

    train_loss = [log["loss"] for log in training_history if "loss" in log]
    eval_acc = [log["eval_accuracy"] for log in training_history if "eval_accuracy" in log]

    epochs = list(range(1, len(train_loss)+1))

    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_loss, marker='o', label="Train Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training Loss per Epoch")
    plt.grid(True)

    plt.subplot(1, 2, 2)
    plt.plot(range(1, len(eval_acc)+1), eval_acc, marker='o', color='green', label="Eval Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Evaluation Accuracy per Epoch")
    plt.grid(True)

    plt.tight_layout()
    plt.show()
# --- SHAP Analysis for XGBoost ---
import shap

explainer = shap.Explainer(xgb_clf)
shap_values = explainer(X_test_embeddings[:100])  # Limit for speed

# Visualize SHAP values
shap.summary_plot(shap_values, X_test_embeddings[:100])'''
