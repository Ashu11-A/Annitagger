from transformers import BertTokenizerFast, BertForTokenClassification, BertConfig
import torch

# Set your model directory
model_dir = "./filename_ner_model"

# Load the configuration and tokenizer
config = BertConfig.from_pretrained(model_dir)
tokenizer = BertTokenizerFast.from_pretrained(model_dir)

# Load the model
model = BertForTokenClassification.from_pretrained(model_dir, config=config, use_safetensors=True)

# Test prediction
filename = "[Z-A] Re-Zero kara Hajimeru Isekai Seikatsu - s01e14 [WEB 1080p].mkv"
inputs = tokenizer(filename, return_tensors="pt", truncation=True, is_split_into_words=False)

with torch.no_grad():
    outputs = model(**inputs)

logits = outputs.logits
predictions = torch.argmax(logits, dim=2)

tokens = tokenizer.convert_ids_to_tokens(inputs.input_ids[0])
predicted_labels = [model.config.id2label[pred.item()] for pred in predictions[0]]

entities = []
current_entity = None
current_value = []

for token, label in zip(tokens, predicted_labels):
    if label.startswith('B-'):
        if current_entity:
            entities.append({'entity': current_entity, 'value': tokenizer.convert_tokens_to_string(current_value)})
        current_entity = label[2:]
        current_value = [token]
    elif label.startswith('I-') and current_entity == label[2:]:
        current_value.append(token)
    else:
        if current_entity:
            entities.append({'entity': current_entity, 'value': tokenizer.convert_tokens_to_string(current_value)})
        current_entity = None
        current_value = []

if current_entity:
    entities.append({'entity': current_entity, 'value': tokenizer.convert_tokens_to_string(current_value)})

print(entities)
