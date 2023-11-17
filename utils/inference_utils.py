import torch
from torch.utils.data import DataLoader, SequentialSampler
from torch.utils.data import TensorDataset
from itertools import groupby
import numpy as np
from torch.nn import CrossEntropyLoss


def calculate_sequence_loss(model, device, encoded_dataset, batch_size = 32) -> list[float]:
    dataset = TensorDataset(
        encoded_dataset["input_ids"], 
        encoded_dataset["attention_mask"], 
        encoded_dataset["position_ids"], 
        encoded_dataset["labels"]
        )

    dataloader = DataLoader(
                dataset,  # The training samples.
                sampler = SequentialSampler(dataset), # Select batches sequentially
                batch_size = batch_size # Trains with this batch size.
            )
    prediction_losses = []

    for _, batch in enumerate(dataloader):
        b_input_ids = batch[0].to(device)
        b_input_mask = batch[1].to(device)
        b_position_ids = batch[2].to(device)
        b_labels = batch[3].to(device)

        with torch.no_grad():
            outputs = model(b_input_ids, attention_mask=b_input_mask, position_ids=b_position_ids, labels=b_labels)
            logits = outputs.logits

            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = b_labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = CrossEntropyLoss(reduction='none')
            loss = loss_fct(shift_logits.permute(0, 2, 1), shift_labels)
            mean_loss = torch.sum(loss, dim=1)/torch.count_nonzero(loss, dim=1)
            prediction_losses.append(mean_loss.detach().cpu())

        torch.cuda.empty_cache()
    
    return torch.cat(prediction_losses)


def calculate_classification_accuracy(encoded_dataset, prediction_losses):
    predicted_indices = [0 for _ in range(len(encoded_dataset['answer_ids']))]

    for key, group in groupby(encoded_dataset['sentence_ids'], lambda s: s[1]):
        losses = []
        class_ids = []

        for (sentence_id, dataset_id, class_id) in group:
            losses.append(prediction_losses[sentence_id])
            class_ids.append(class_id)

        predicted_indices[key] = class_ids[np.argmin(losses)]

    num_correct_predictions = 0

    for i, answer_id in enumerate(encoded_dataset["answer_ids"]):
        if predicted_indices[i] == answer_id:
            num_correct_predictions += 1

    return num_correct_predictions/len(predicted_indices)