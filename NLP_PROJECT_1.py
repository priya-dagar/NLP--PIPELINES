# Fine-Tune LLM Script
# (No shell commands included — install dependencies manually)

import os
import numpy as np
from datasets import load_dataset, DatasetDict, ClassLabel
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
import evaluate
import torch
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

DATASET_NAME = "ag_news"
MODEL_NAME = "distilbert-base-uncased"
MAX_LENGTH = 128
RANDOM_SEED = 42
BATCH_SIZE = 16
LR = 2e-5
EPOCHS = 3
OUTPUT_DIR = "./model-output"

os.makedirs(OUTPUT_DIR, exist_ok=True)
device = "cuda" if torch.cuda.is_available() else "cpu"
print("Device:", device)

raw_datasets = load_dataset(DATASET_NAME)

def prepare_ag_news_split(ds):
    def concat(example):
        example["text"] = (example.get("title","") + " - " + example.get("description","")).strip()
        return example
    return ds.map(concat)

raw_datasets = prepare_ag_news_split(raw_datasets)
text_column = "text"
label_column = "label"

train_testvalid = raw_datasets["train"].train_test_split(test_size=0.2, seed=RANDOM_SEED)
test_valid = train_testvalid["test"].train_test_split(test_size=0.5, seed=RANDOM_SEED)

datasets = DatasetDict({
    "train": train_testvalid["train"],
    "validation": test_valid["train"],
    "test": test_valid["test"]
})

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

def preprocess_function(examples):
    return tokenizer(examples[text_column], truncation=True, padding="max_length", max_length=MAX_LENGTH)

encoded = datasets.map(preprocess_function, batched=True)
encoded = encoded.remove_columns(
    [c for c in encoded["train"].column_names if c not in (label_column, "input_ids","attention_mask","token_type_ids")]
)
encoded.set_format(type="torch")

if isinstance(datasets["train"].features[label_column], ClassLabel):
    label_list = datasets["train"].features[label_column].names
    num_labels = len(label_list)
else:
    label_list = None
    num_labels = len(set(datasets["train"][label_column]))

model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=num_labels).to(device)

accuracy = evaluate.load("accuracy")
precision = evaluate.load("precision")
recall = evaluate.load("recall")
f1 = evaluate.load("f1")

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=1)
    return {
        "accuracy": accuracy.compute(predictions=preds, references=labels)["accuracy"],
        "precision": precision.compute(predictions=preds, references=labels, average="weighted")["precision"],
        "recall": recall.compute(predictions=preds, references=labels, average="weighted")["recall"],
        "f1": f1.compute(predictions=preds, references=labels, average="weighted")["f1"],
    }

training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    num_train_epochs=EPOCHS,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,
    learning_rate=LR,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="f1",
    logging_dir=os.path.join(OUTPUT_DIR, "logs"),
    logging_steps=50,
    seed=RANDOM_SEED
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=encoded["train"],
    eval_dataset=encoded["validation"],
    tokenizer=tokenizer,
    compute_metrics=compute_metrics
)

trainer.train()

metrics = trainer.evaluate(encoded["test"])
print("Test metrics:", metrics)

pred_output = trainer.predict(encoded["test"])
preds = np.argmax(pred_output.predictions, axis=1)
labels = pred_output.label_ids

print(classification_report(labels, preds, target_names=label_list if label_list else None))

cm = confusion_matrix(labels, preds)
plt.figure(figsize=(8,6))
sns.heatmap(cm, annot=True, fmt="d", xticklabels=label_list, yticklabels=label_list)
plt.savefig("confusion_matrix.png")

model.save_pretrained(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)

print("All done. Saved to", OUTPUT_DIR)
