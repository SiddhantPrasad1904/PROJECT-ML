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
from sklearn.ensemble import RandomForestClassifier
from collections import Counter

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
    text = text.lower()  # Convert to lowercase
    text = re.sub(r"http\S+|www\S+", "", text)  # Remove URLs
    text = re.sub(r"[^a-zA-Z\s]", "", text)  # Remove punctuation and numbers
    words = word_tokenize(text)  # Tokenization
    words = [lemmatizer.lemmatize(word) for word in words if word not in stop_words]  # Lemmatization & Stopword removal
    return " ".join(words)

# Apply text cleaning
df["cleaned_tweet"] = df["tweet"].apply(clean_text)

# Load BERT tokenizer
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

# Check if a saved model exists
model_path = "./saved_model"
if os.path.exists(model_path):
    print("Loading saved model...")
    model = BertForSequenceClassification.from_pretrained(model_path)
    tokenizer = BertTokenizer.from_pretrained(model_path)
    train_model = False  # Flag to skip training
else:
    print("Training new model...")
    model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=num_labels)
    train_model = True  # Train only if no saved model exists

# Move model to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
model.to(device)

# Update deprecated argument
eval_strategy = "epoch"

training_args = TrainingArguments(
    output_dir="./results",
    eval_strategy=eval_strategy,
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

# Implement Random Forest Classifier for comparison
rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
rf_classifier.fit(X_input_ids.numpy(), y_labels.numpy())
rf_preds = rf_classifier.predict(X_input_ids.numpy())

# Evaluate Random Forest Model
rf_accuracy = accuracy_score(y_labels.numpy(), rf_preds)
rf_precision, rf_recall, rf_f1, _ = precision_recall_fscore_support(y_labels.numpy(), rf_preds, average="weighted")
rf_report = classification_report(y_labels.numpy(), rf_preds)
print(f"Random Forest Accuracy: {rf_accuracy:.4f}")
print(f"Random Forest Precision: {rf_precision:.4f}")
print(f"Random Forest Recall: {rf_recall:.4f}")
print(f"Random Forest F1-score: {rf_f1:.4f}")
print("Random Forest Classification Report:\n", rf_report)

# Define evaluate_model function
def evaluate_model(trainer, test_dataset):
    print("Evaluating model...")
    metrics = trainer.evaluate(eval_dataset=test_dataset)
    print("Evaluation Metrics:", metrics)

# Run evaluation
evaluate_model(trainer, test_dataset)

# Function for real-time predictions
def predict_text(text, model, tokenizer, device):
    model.eval()
    text_cleaned = clean_text(text)
    encoding = tokenizer(text_cleaned, truncation=True, padding=True, max_length=128, return_tensors="pt")
    input_ids = encoding["input_ids"].to(device)
    attention_mask = encoding["attention_mask"].to(device)
    with torch.no_grad():
        outputs = model(input_ids, attention_mask=attention_mask)
        logits = outputs.logits
        prediction = torch.argmax(logits, dim=1).cpu().item()
    return prediction

# Real-time prediction loop
while True:
    user_input = input("Enter a tweet (or type 'exit' to quit): ")
    if user_input.lower() == "exit":
        break
    prediction = predict_text(user_input, model, tokenizer, device)
    print(f"Predicted class: {prediction}")
