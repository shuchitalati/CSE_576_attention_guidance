import torch
import json
import ast

TRAIN_BLOCK_SIZE = 512


def load_dataset(dataset_path: str):
    with open(dataset_path) as f:
        dataset = f.readlines()
        dataset = [json.loads(d) for d in dataset]

    return dataset


def encode_inverse_scaling_dataset(dataset, tokenizer):
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = 'left'

    input_ids, attention_mask, position_ids, labels = [], [], [], []
    sentences, sentences_ids, answer_ids = [], [], []

    s_ids_idx = 0

    for i, data in enumerate(dataset):
        for j, cls in enumerate(ast.literal_eval(data["classes"])):
            sentences.append(data["prompt"] + cls)
            sentences_ids.append(
                (s_ids_idx, i, j))  # Sentence at s_ids_idx of the input is from i'th row and j'th class of the dataset
            s_ids_idx += 1
            labels.append(tokenizer.encode(cls, return_tensors='pt'))
        answer_ids.append(data["answer_index"])

    max_len = max([len(tokenizer.encode(s)) for s in sentences]) + 5

    for i, sentence in enumerate(sentences):
        encoded_dict = tokenizer.encode_plus(
            text=sentence,  # Sentence 1
            max_length=max_len,  # Pad & truncate all sentences.
            padding='max_length',
            return_attention_mask=True,  # Construct attn. masks.
            return_tensors='pt',  # Return pytorch tensors.
        )
        input_ids.append(encoded_dict["input_ids"])
        attention_mask.append(encoded_dict["attention_mask"])
        encoded_labels_len = len(labels[i][-1])
        labels[i] = torch.tensor([[-100] * (max_len - encoded_labels_len) + list(labels[i][-1])])

    input_ids = torch.cat(input_ids, dim=0)
    attention_mask = torch.cat(attention_mask, dim=0)
    position_ids = attention_mask.long().cumsum(-1) - 1
    position_ids.masked_fill_(attention_mask == 0, 1)
    labels = torch.cat(labels, dim=0)

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "position_ids": position_ids,
        "labels": labels,
        "sentence_ids": sentences_ids,
        "answer_ids": answer_ids
    }


def group_texts(examples):
    # Concatenate all texts.
    concatenated_examples = {k: sum(examples[k], []) for k in examples.keys()}
    total_length = len(concatenated_examples[list(examples.keys())[0]])
    # We drop the small remainder, we could add padding if the model supported it instead of this drop, you can
    # customize this part to your needs.
    if total_length >= TRAIN_BLOCK_SIZE:
        total_length = (total_length // TRAIN_BLOCK_SIZE) * TRAIN_BLOCK_SIZE
    # Split by chunks of TRAIN_BLOCK_SIZE.
    result = {
        k: [t[i: i + TRAIN_BLOCK_SIZE] for i in range(0, total_length, TRAIN_BLOCK_SIZE)]
        for k, t in concatenated_examples.items()
    }
    result["labels"] = result["input_ids"].copy()
    return result
