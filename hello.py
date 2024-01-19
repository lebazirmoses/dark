import pandas as pd
from sklearn.model_selection import train_test_split
from transformers import BertTokenizerFast, BertForSequenceClassification

# Load the pre-trained BERT model and tokenizer
model = BertForSequenceClassification.from_pretrained("bert-base-uncased")
tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")

# Replace 'train_dataset_file' and 'eval_dataset_file' with your actual file names
train_dataset_file = "path/to/train_dataset.csv"
eval_dataset_file = "path/to/eval_dataset.csv"

# Load your dataset function (replace this with your actual loading logic)
def load_your_dataset_function(file_path):
    df = pd.read_csv(file_path)
    return df['text'].tolist(), df['label'].tolist()

# Load training and evaluation datasets
train_texts, train_labels = load_your_dataset_function(train_dataset_file)
eval_texts, eval_labels = load_your_dataset_function(eval_dataset_file)

# Tokenize the datasets
train_encodings = tokenizer(train_texts, truncation=True, padding=True)
eval_encodings = tokenizer(eval_texts, truncation=True, padding=True)

# Create torch datasets
import torch
train_dataset = torch.utils.data.TensorDataset(
    torch.tensor(train_encodings['input_ids']),
    torch.tensor(train_encodings['attention_mask']),
    torch.tensor(train_labels)
)

eval_dataset = torch.utils.data.TensorDataset(
    torch.tensor(eval_encodings['input_ids']),
    torch.tensor(eval_encodings['attention_mask']),
    torch.tensor(eval_labels)
)

# Define the training arguments
training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=3,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    warmup_steps=500,
    weight_decay=0.01,
    logging_dir='./logs',
)

# Create a Trainer object
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
)

# Fine-tune the BERT model
trainer.train()


# Now that the model is fine-tuned, you can use it for text classification

from transformers import pipeline

# Create a text classification pipeline using the fine-tuned BERT model
classifier = pipeline("text-classification", model=model, tokenizer=tokenizer)

# Example usage: Predict the presence of dark patterns in a given text
result = classifier("In this basket sneaking, we force you to pay before getting your products.")
print(result)
