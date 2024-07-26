import torch
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from transformers import DistilBertTokenizer

def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

def prepare_features(examples, tokenizer, max_length, doc_stride):
    features = tokenizer(
        examples['question'],
        examples['context'],
        truncation="only_second",
        max_length=max_length,
        stride=doc_stride,
        return_overflowing_tokens=True,
        return_offsets_mapping=True,
        padding="max_length"
    )

    sample_mapping = features.pop("overflow_to_sample_mapping")
    offset_mapping = features.pop("offset_mapping")

    features["start_positions"] = []
    features["end_positions"] = []

    for i, offsets in enumerate(offset_mapping):
        input_ids = features["input_ids"][i]
        cls_index = input_ids.index(tokenizer.cls_token_id)

        sequence_ids = features.sequence_ids(i)
        sample_index = sample_mapping[i]
        answers = examples["answers"][sample_index]
        if len(answers["answer_start"]) == 0:
            features["start_positions"].append(cls_index)
            features["end_positions"].append(cls_index)
        else:
            start_char = answers["answer_start"][0]
            end_char = start_char + len(answers["text"][0])

            token_start_index = 0
            while sequence_ids[token_start_index] != 1:
                token_start_index += 1

            token_end_index = len(input_ids) - 1
            while sequence_ids[token_end_index] != 1:
                token_end_index -= 1

            if not (offsets[token_start_index][0] <= start_char and offsets[token_end_index][1] >= end_char):
                features["start_positions"].append(cls_index)
                features["end_positions"].append(cls_index)
            else:
                while token_start_index < len(offsets) and offsets[token_start_index][0] <= start_char:
                    token_start_index += 1
                features["start_positions"].append(token_start_index - 1)

                while offsets[token_end_index][1] >= end_char:
                    token_end_index -= 1
                features["end_positions"].append(token_end_index + 1)

    return features

def create_data_loader(dataset, batch_size, sampler):
    data_sampler = sampler(dataset)
    data_loader = DataLoader(dataset, sampler=data_sampler, batch_size=batch_size)
    return data_loader
