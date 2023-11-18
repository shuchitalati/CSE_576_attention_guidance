from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import torch
from torch import nn
from torch.utils.data import Dataset
from transformers import Trainer
from transformers.data.data_collator import DataCollator
from transformers.modeling_utils import PreTrainedModel, unwrap_model
from transformers.tokenization_utils_base import PreTrainedTokenizerBase
from transformers.trainer_callback import TrainerCallback, TrainerState, TrainerControl
from transformers.trainer_utils import EvalPrediction
from transformers.training_args import TrainingArguments
from transformers.models.auto.modeling_auto import MODEL_FOR_CAUSAL_LM_MAPPING_NAMES
from transformers.utils import is_peft_available

if is_peft_available():
    from peft import PeftModel


class AttentionGuidanceTrainer(Trainer):
    def __init__(
            self,
            model: Union[PreTrainedModel, nn.Module],
            tokenizer: Optional[PreTrainedTokenizerBase],
            attention_guidance_pattern: dict[int, list[str]],
            args: TrainingArguments = None,
            data_collator: Optional[DataCollator] = None,
            train_dataset: Optional[Dataset] = None,
            eval_dataset: Optional[Union[Dataset, Dict[str, Dataset]]] = None,
            model_init: Optional[Callable[[], PreTrainedModel]] = None,
            compute_metrics: Optional[Callable[[EvalPrediction], Dict]] = None,
            callbacks: Optional[List[TrainerCallback]] = None,
            optimizers: Tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LambdaLR] = (None, None),
            preprocess_logits_for_metrics: Optional[Callable[[torch.Tensor, torch.Tensor], torch.Tensor]] = None,
    ):
        super().__init__(
            model,
            args,
            data_collator,
            train_dataset,
            eval_dataset,
            tokenizer,
            model_init,
            compute_metrics,
            callbacks,
            optimizers,
            preprocess_logits_for_metrics
        )

        self.t_total = None
        self.global_step = 0
        self.attention_guidance_pattern: dict[int, list[int]] = {}

        for head_num in attention_guidance_pattern:
            self.attention_guidance_pattern[head_num] = []

            for token in attention_guidance_pattern[head_num]:
                input_ids = tokenizer.encode(token)

                if len(input_ids) > 1:
                    raise ValueError("Provided multi-token string. Only strings that tokenize into a single token can" +
                                     " be used for attention guidance")

                self.attention_guidance_pattern[head_num].extend(input_ids)

            if len(self.attention_guidance_pattern[head_num]) == 0:
                raise ValueError("Token list provided for each attention head must not be empty")

    def compute_loss(self, model, inputs, return_outputs=False):
        if self.label_smoother is not None and "labels" in inputs:
            labels = inputs.pop("labels")
        else:
            labels = None

        try:
            outputs = model(**inputs, output_attentions=True)
            attentions = outputs.attentions
        except TypeError as e:
            raise TypeError("Attention Guidance is only meant for Attention based models. " +
                            + "Please check that you have provided such a model to the trainer") from e

        if self.args.past_index >= 0:
            self._past = outputs[self.args.past_index]

        if labels is not None:
            if is_peft_available() and isinstance(model, PeftModel):
                model_name = unwrap_model(model.base_model)._get_name()
            else:
                model_name = unwrap_model(model)._get_name()
            if model_name in MODEL_FOR_CAUSAL_LM_MAPPING_NAMES.values():
                loss = self.label_smoother(outputs, labels, shift_labels=True)
            else:
                loss = self.label_smoother(outputs, labels)
        else:
            if isinstance(outputs, dict) and "loss" not in outputs:
                raise ValueError(
                    "The model did not return a loss from the inputs, only the following keys: "
                    f"{','.join(outputs.keys())}. For reference, the inputs it received are {','.join(inputs.keys())}."
                )
            loss = outputs["loss"] if isinstance(outputs, dict) else outputs[0]

        loss += self.compute_attention_guidance_loss(inputs["input_ids"], attentions) * self.linear_schedule_for_scale()

        return (loss, outputs) if return_outputs else loss

    def train(
            self,
            resume_from_checkpoint: str | bool | None = None, trial: Any | Dict[str, Any] = None,
            ignore_keys_for_eval: List[str] | None = None,
            **kwargs
    ):
        train_dataloader = self.get_train_dataloader()
        if self.args.max_steps > 0:
            self.t_total = self.args.max_steps
        else:
            self.t_total = int(len(train_dataloader) //
                               self.args.gradient_accumulation_steps * self.args.num_train_epochs)

        return super().train(resume_from_checkpoint, trial, ignore_keys_for_eval, **kwargs)

    def training_step(self, model: nn.Module, inputs: Dict[str, torch.Tensor | Any]) -> torch.Tensor:
        self.global_step += 1
        return super().training_step(model, inputs)

    def create_attention_guidance_targets(self, input_ids):
        batch_size = input_ids.size()[0]
        sequence_len = input_ids.size()[-1]
        ag_num = len(self.attention_guidance_pattern.keys())  # Number of attention heads being guided
        targets = torch.zeros(batch_size, ag_num, sequence_len, sequence_len)

        for head_num in self.attention_guidance_pattern.keys():
            for token_input_id in self.attention_guidance_pattern[head_num]:
                seq_ids = (input_ids == token_input_id).nonzero(as_tuple=True)
                targets[seq_ids[0], head_num, :, seq_ids[1]] = 1

        return targets.to(self.args.device)

    def compute_attention_guidance_loss(self, input_ids, attentions):
        targets = self.create_attention_guidance_targets(input_ids)
        loss_fn = torch.nn.MSELoss()
        loss = 0.

        for layer_num in range(len(attentions)):
            for head_num in self.attention_guidance_pattern.keys():
                loss += loss_fn(targets[:, head_num], attentions[layer_num][:, head_num])

        return loss

    def linear_schedule_for_scale(self, num_stagnant_steps=20000):
        """
        Linear schedule for scale, the relative weight assigned
        to ag_loss
        """
        tot = self.t_total
        cur = self.global_step

        if cur < num_stagnant_steps:
            return 1.0
        else:
            return max(0.0, float(tot - cur) / float(max(1, tot - num_stagnant_steps)))


# Print progress to stdout
class StdoutCallback(TrainerCallback):
    def __init__(self):
        super().__init__()
        self.cur_steps = 0

    def on_step_end(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        self.cur_steps += 1
        print(f"Completed {self.cur_steps}/{state.max_steps} Training Steps")
