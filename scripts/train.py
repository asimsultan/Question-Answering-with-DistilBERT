import os
import torch
from transformers import DistilBertForQuestionAnswering, DistilBertTokenizer, AdamW, get_scheduler
from datasets import load_dataset
from utils import get_device, prepare_features, create_data_loader

# Parameters
model_name = 'distilbert-base-uncased'
max_length = 384
doc_stride = 128
batch_size = 8
epochs = 3
learning_rate = 3e-5

# Load Dataset
dataset = load_dataset('squad')
train_dataset = dataset['train']
validation_dataset = dataset['validation']

# Tokenizer
tokenizer = DistilBertTokenizer.from_pretrained(model_name)

# Tokenize Data
train_dataset = train_dataset.map(lambda examples: prepare_features(examples, tokenizer, max_length, doc_stride), batched=True, remove_columns=train_dataset.column_names)
validation_dataset = validation_dataset.map(lambda examples: prepare_features(examples, tokenizer, max_length, doc_stride), batched=True, remove_columns=validation_dataset.column_names)

train_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'start_positions', 'end_positions'])
validation_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'start_positions', 'end_positions'])

# DataLoader
train_loader = create_data_loader(train_dataset, batch_size, RandomSampler)
validation_loader = create_data_loader(validation_dataset, batch_size, SequentialSampler)

# Model
device = get_device()
model = DistilBertForQuestionAnswering.from_pretrained(model_name)
model.to(device)

# Optimizer and Scheduler
optimizer = AdamW(model.parameters(), lr=learning_rate)
num_training_steps = len(train_loader) * epochs
lr_scheduler = get_scheduler(
    name="linear", optimizer=optimizer, num_warmup_steps=0, num_training_steps=num_training_steps
)

# Training Function
def train_epoch(model, data_loader, optimizer, device, scheduler):
    model.train()
    total_loss = 0

    for batch in data_loader:
        b_input_ids = batch['input_ids'].to(device)
        b_attention_mask = batch['attention_mask'].to(device)
        b_start_positions = batch['start_positions'].to(device)
        b_end_positions = batch['end_positions'].to(device)

        outputs = model(
            b_input_ids,
            attention_mask=b_attention_mask,
            start_positions=b_start_positions,
            end_positions=b_end_positions
        )
        loss = outputs.loss
        total_loss += loss.item()

        loss.backward()
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()

    avg_loss = total_loss / len(data_loader)
    return avg_loss

# Training Loop
for epoch in range(epochs):
    train_loss = train_epoch(model, train_loader, optimizer, device, lr_scheduler)
    print(f'Epoch {epoch+1}/{epochs}')
    print(f'Train Loss: {train_loss}')

# Save Model
model_dir = './models'
if not os.path.exists(model_dir):
    os.makedirs(model_dir)

model.save_pretrained(model_dir)
tokenizer.save_pretrained(model_dir)