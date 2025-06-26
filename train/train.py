from transformers import BertTokenizerFast, BertForTokenClassification, TrainingArguments, Trainer
from datasets import Dataset
import json
import numpy as np
from seqeval.metrics import f1_score

# =============================================================================
# 1. Dataset Preparation & Tokenization (as before)
# =============================================================================
with open('anime_filenames_v2.json', 'r', encoding='utf-8') as file:
    data = json.load(file)
dataset = Dataset.from_list(data)

model_name = "prajjwal1/bert-small" 
tokenizer = BertTokenizerFast.from_pretrained(model_name)

def tokenize_and_align_labels(examples):
  tokenized_inputs = tokenizer(
    examples["filename"],
    max_length=128,              # Maximum sequence length
    stride=32,                   # Overlap tokens to maintain context
    padding="max_length",
    return_offsets_mapping=True
  )
  
  labels = []
  for i, (offsets, entities) in enumerate(zip(tokenized_inputs["offset_mapping"], examples["entities"])):
      label_ids = []
      for offset in offsets:
          # Skip special tokens ([CLS], [SEP], etc.)
          if offset[0] == 0 and offset[1] == 0:
              label_ids.append(-100)
              continue
              
          # Determine token label
          found = False
          for ent in entities:
              if offset[0] >= ent["start"] and offset[1] <= ent["end"]:
                  label = f"B-{ent['label']}" if offset[0] == ent["start"] else f"I-{ent['label']}"
                  label_ids.append(label)
                  found = True
                  break
          if not found:
              label_ids.append("O")
      labels.append(label_ids)
  
  # Convert labels to label IDs
  label_map = {
    "O": 0,
    "B-SUBMITTER": 1, "I-SUBMITTER": 2,
    "B-SERIES": 3, "I-SERIES": 4,
    "B-SEASON": 5, "I-SEASON": 6,
    "B-EPISODE": 7, "I-EPISODE": 8,
    "B-SUBTITLE": 9, "I-SUBTITLE": 10,
    "B-SOURCE": 11, "I-SOURCE": 12,
    "B-QUALITY": 13, "I-QUALITY": 14,
    "B-ENCODE": 15, "I-ENCODE": 16,
    "B-AUDIO": 17, "I-AUDIO": 18,
    "B-HASH": 19, "I-HASH": 20,
    "B-FILE_TYPE": 21, "I-FILE_TYPE": 22
  }
  tokenized_inputs["labels"] = [
    [label_map.get(l, -100) for l in label_seq] for label_seq in labels
  ]
  return tokenized_inputs

tokenized_dataset = dataset.map(tokenize_and_align_labels, batched=True)
split_dataset = tokenized_dataset.train_test_split(test_size=0.1)
train_dataset = split_dataset["train"]
eval_dataset = split_dataset["test"]

# =============================================================================
# 2. Model Initialization Function
# =============================================================================
def model_init():
  return BertForTokenClassification.from_pretrained(
    model_name,
    num_labels=23,
    id2label={
      0: "O",
      1: "B-SUBMITTER", 2: "I-SUBMITTER",
      3: "B-SERIES", 4: "I-SERIES",
      5: "B-SEASON", 6: "I-SEASON",
      7: "B-EPISODE", 8: "I-EPISODE",
      9: "B-SUBTITLE", 10: "I-SUBTITLE",
      11: "B-SOURCE", 12: "I-SOURCE",
      13: "B-QUALITY", 14: "I-QUALITY",
      15: "B-ENCODE", 16: "I-ENCODE",
      17: "B-AUDIO", 18: "I-AUDIO",
      19: "B-HASH", 20: "I-HASH",
      21: "B-FILE_TYPE", 22: "I-FILE_TYPE"
    },
    label2id={
      "O": 0,
      "B-SUBMITTER": 1, "I-SUBMITTER": 2,
      "B-SERIES": 3, "I-SERIES": 4,
      "B-SEASON": 5, "I-SEASON": 6,
      "B-EPISODE": 7, "I-EPISODE": 8,
      "B-SUBTITLE": 9, "I-SUBTITLE": 10,
      "B-SOURCE": 11, "I-SOURCE": 12,
      "B-QUALITY": 13, "I-QUALITY": 14,
      "B-ENCODE": 15, "I-ENCODE": 16,
      "B-AUDIO": 17, "I-AUDIO": 18,
      "B-HASH": 19, "I-HASH": 20,
      "B-FILE_TYPE": 21, "I-FILE_TYPE": 22
    }
  )

# =============================================================================
# 3. Define Compute Metrics Function
# =============================================================================

dummy_model = model_init()
label_list = list(dummy_model.config.id2label.values())
del dummy_model

def compute_metrics(p):
    predictions, labels = p
    predictions = np.argmax(predictions, axis=2)
    
    # Remove ignored index (-100) and align labels
    true_labels = [[label_list[l] for l in label if l != -100] for label in labels]
    true_predictions = [
        [label_list[p] for (p, l) in zip(pred, label) if l != -100]
        for pred, label in zip(predictions, labels)
    ]
    f1 = f1_score(true_labels, true_predictions)

    return {"f1": f1}

# =============================================================================
# 4. Setup Default Training Arguments
# =============================================================================
training_args = TrainingArguments(
    output_dir="output",
    num_train_epochs=25,
    per_device_train_batch_size=250,
    # learning_rate=3e-5,
    # warmup_ratio=0.1,
    # weight_decay=0.01,
    logging_steps=100,
    eval_strategy="steps",
    eval_steps=500,
    save_steps=500,
    fp16=True,
    load_best_model_at_end=True,
    metric_for_best_model="f1",
)

# =============================================================================
# 5. Initialize Trainer
# =============================================================================
trainer = Trainer(
    model_init=model_init,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    compute_metrics=compute_metrics
)

# =============================================================================
# 6. Define the Hyperparameter Search Space
# # =============================================================================
# def hp_space(trial):
#     return {
#         "learning_rate": trial.suggest_float("learning_rate", 1e-6, 1e-4, log=True),
#         "per_device_train_batch_size": trial.suggest_categorical("per_device_train_batch_size", [16, 32, 64]),
#         "num_train_epochs": trial.suggest_int("num_train_epochs", 3, 5),
#         "warmup_ratio": trial.suggest_float("warmup_ratio", 0.0, 0.2),
#     }

# =============================================================================
# 7. Run Hyperparameter Search
# =============================================================================
# best_run = trainer.hyperparameter_search(
#     direction="maximize",  # maximizing the F1 score
#     hp_space=hp_space,
#     n_trials=10  # Number of hyperparameter sets to try
# )

# print("Best run:", best_run)

# After the search, you can update the training arguments and re-train on the full training data if needed:
# for param, value in best_run.hyperparameters.items():
#     setattr(training_args, param, value)

# trainer.args = training_args

# =============================================================================
# 8. Final Training with Best Hyperparameters
# =============================================================================
trainer.train()
model = trainer.model
model.save_pretrained("filename_ner_model")
tokenizer.save_pretrained("filename_ner_model")
