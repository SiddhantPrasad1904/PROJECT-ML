import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertForSequenceClassification , Trainer, TrainingArguments , AutoTokenizer, AdamW
from datasets import load_dataset
import os
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report
from sklearn.utils.class_weight import compute_class_weight
import xgboost as xgb
import numpy as np

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
model_path = "./saved_model_v4"
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

# Compute class weights for imbalance
class_weights = compute_class_weight("balanced", classes=np.unique(df["class"]), y=df["class"])
class_weights = torch.tensor(class_weights, dtype=torch.float).to(device)

training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    save_strategy="epoch",
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=3,  # Increased epochs
    weight_decay=0.1,  # Higher weight decay
)

def tokenize_data(texts, labels, tokenizer, max_length=128):
    encodings = tokenizer(list(texts), truncation=True, padding=True, max_length=max_length, return_tensors="pt")
    return encodings["input_ids"].cpu(), encodings["attention_mask"].cpu(), torch.tensor(labels.values).cpu()

# Tokenize dataset
X_input_ids, X_attention_mask, y_labels = tokenize_data(df["cleaned_tweet"], df["class"], tokenizer)

class HateSpeechDataset(Dataset):
    def __init__(self, input_ids, attention_mask, labels):
        self.input_ids = input_ids.cpu()
        self.attention_mask = attention_mask.cpu()
        self.labels = labels.cpu()

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

# Create data loaders
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, pin_memory=(device == "cpu"))
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False, pin_memory=(device == "cpu"))

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

def evaluate_model(trainer, test_dataset):
    predictions = trainer.predict(test_dataset)
    preds = torch.argmax(torch.tensor(predictions.predictions), dim=1).numpy()
    labels = predictions.label_ids
    accuracy = accuracy_score(labels, preds)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average="weighted")
    report = classification_report(labels, preds)
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-score: {f1:.4f}")
    print("Classification Report:\n", report)

# Run evaluation
evaluate_model(trainer, test_dataset)

# Extract BERT embeddings for XGBoost
model.eval()
def extract_embeddings(texts, tokenizer, model, device):
    inputs = tokenizer(texts, padding=True, truncation=True, max_length=128, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model.bert(**inputs)
    return outputs.last_hidden_state[:, 0, :].cpu().numpy()

X_train_embeddings = extract_embeddings(df["cleaned_tweet"].iloc[:train_size], tokenizer, model, device)
X_test_embeddings = extract_embeddings(df["cleaned_tweet"].iloc[train_size:], tokenizer, model, device)

y_train, y_test = y_labels[:train_size].numpy(), y_labels[train_size:].numpy()

# Train XGBoost classifier
xgb_clf = xgb.XGBClassifier(n_estimators=100, max_depth=5, learning_rate=0.1)
xgb_clf.fit(X_train_embeddings, y_train)

# Evaluate XGBoost
xgb_preds = xgb_clf.predict(X_test_embeddings)
print("XGBoost Accuracy:", accuracy_score(y_test, xgb_preds))
print("Classification Report:\n", classification_report(y_test, xgb_preds))
