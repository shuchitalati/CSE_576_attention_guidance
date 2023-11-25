import os

from datasets import load_dataset, DownloadConfig
from transformers import TrainingArguments, DataCollatorForLanguageModeling, GPT2LMHeadModel, GPT2Tokenizer
from transformers.trainer_utils import get_last_checkpoint

from attention_guidance_trainer import AttentionGuidanceTrainer, StdoutCallback
from utils.dataset_utils import group_texts

PRETRAINED_MODEL_NAME = "gpt2"
CHECKPOINT_DIR = "./checkpoints"
FINETUNED_SAVE_DIR = "./models"
FINETUNED_MODEL_NAME = "gpt2-ag"
DATASET_NAME = 'Skylion007/openwebtext'

CONTRAST_TOKENS = [' not', ' Not', ' but', ' But', ' though', ' Though', 'though', 'not', ' Unlike', ' Nevertheless', ' Nonetheless', ' Despite', ' Contrast', ' Contrary', ' Whereas', ' Alternatively', ' unlike', ' nevertheless', ' nonetheless', ' despite', ' contrast', ' contrary', ' whereas', ' alternatively']

# Keys are the attention head number, values are the tokens that are guided
ATTENTION_GUIDANCE_PATTERN = {
    0: CONTRAST_TOKENS,
}

TRAINING_ARGS = TrainingArguments(
    per_device_train_batch_size=32,
    output_dir=CHECKPOINT_DIR,
    num_train_epochs=2,
    save_strategy='epoch'
)

tokenizer = GPT2Tokenizer.from_pretrained(PRETRAINED_MODEL_NAME)
tokenizer.pad_token = tokenizer.eos_token
model = GPT2LMHeadModel.from_pretrained(PRETRAINED_MODEL_NAME)

raw_dataset = load_dataset(
    path=DATASET_NAME,
    split='train[:1000000]',  # use train[:int_value] to load a subset of the dataset, mainly for testing purposes
    download_config=DownloadConfig(cache_dir="./dataset/gpt2")
)

tokenized_dataset = raw_dataset.map(
    lambda x: tokenizer(x["text"]),
    batched=True,
    remove_columns=["text"]
)

lm_dataset = tokenized_dataset.map(
    group_texts,
    batched=True
)

data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

trainer = AttentionGuidanceTrainer(
    model=model,
    tokenizer=tokenizer,
    attention_guidance_pattern=ATTENTION_GUIDANCE_PATTERN,
    args=TRAINING_ARGS,
    train_dataset=lm_dataset,
    data_collator=data_collator
)

resume_from_checkpoint = False
if get_last_checkpoint(CHECKPOINT_DIR):
    resume_from_checkpoint = True

trainer.train(resume_from_checkpoint=resume_from_checkpoint)

trainer.save_model(os.path.join(FINETUNED_SAVE_DIR, FINETUNED_MODEL_NAME))
