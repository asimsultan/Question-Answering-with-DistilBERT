import torch
from transformers import DistilBertForQuestionAnswering, DistilBertTokenizer
from datasets import load_dataset
from utils import get_device, prepare_features, create_data_loader

# Parameters
model_dir = './models'
max_length = 384
doc_stride = 128
batch_size = 8

# Load Model and Tokenizer
model = DistilBertForQuestionAnswering.from_pretrained(model_dir)
tokenizer = DistilBertTokenizer.from_pretrained(model_dir)

# Device
device = get_device()
model.to(device)

# Load Dataset
dataset = load_dataset('squad')
validation_dataset = dataset['validation']

# Tokenize Data
validation_dataset = validation_dataset.map(lambda examples: prepare_features(examples, tokenizer, max_length, doc_stride), batched=True, remove_columns=validation_dataset.column_names)

validation_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'start_positions', 'end_positions'])

# DataLoader
validation_loader = create_data_loader(validation_dataset, batch_size, SequentialSampler)

# Evaluation Function
def evaluate(model, data_loader, device):
    model.eval()
    total_loss = 0

    with torch.no_grad():
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

    avg_loss = total_loss / len(data_loader)
    return avg_loss

# Evaluate
validation_loss = evaluate(model, validation_loader, device)
print(f'Validation Loss: {validation_loss}')