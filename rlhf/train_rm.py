# Adapted from https://github.com/CarperAI/trlx/tree/main

import os

from sklearn.model_selection import train_test_split
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, Trainer, TrainingArguments
from torch import nn
import torch
from torch.utils.data import Dataset

from tqdm.auto import tqdm


class PairwiseDataset(Dataset):
    def __init__(self, data, tokenizer, max_length):
        super().__init__()
        self.input_ids = []
        self.decoder_input_ids_chosen = []
        self.decoder_input_ids_rejected = []
        self.attn_masks = []
        
        for item in tqdm(data):
            prompt, chosen, rejected = item["prompt"], item["chosen"], item["rejected"]
            
            # Tokenize the summaries first to determine their length
            chosen_summary_ids = tokenizer.encode(chosen, add_special_tokens=False)
            rejected_summary_ids = tokenizer.encode(rejected, add_special_tokens=False)

            # Allocate tokens to the summaries, ensuring neither is truncated
            max_summary_length = max(len(chosen_summary_ids), len(rejected_summary_ids))
            available_length = max_length - max_summary_length - 1  # -1 for EOS token

            # Tokenize the prompt with truncation
            prompt_ids = tokenizer.encode(prompt + '\n', add_special_tokens=False, max_length=available_length, truncation=True)

            # Combine prompt and summary ids, adding EOS token at the end
            chosen_input_ids = prompt_ids + chosen_summary_ids + [tokenizer.eos_token_id]
            rejected_input_ids = prompt_ids + rejected_summary_ids + [tokenizer.eos_token_id]

            # Create attention masks
            chosen_attention_mask = [1] * len(chosen_input_ids)
            rejected_attention_mask = [1] * len(rejected_input_ids)

            # Padding
            chosen_padding_length = max_length - len(chosen_input_ids)
            chosen_input_ids.extend([tokenizer.pad_token_id] * chosen_padding_length)
            chosen_attention_mask.extend([0] * chosen_padding_length)

            rejected_padding_length = max_length - len(rejected_input_ids)
            rejected_input_ids.extend([tokenizer.pad_token_id] * rejected_padding_length)
            rejected_attention_mask.extend([0] * rejected_padding_length)

            # Add to lists
            self.input_ids.append(torch.tensor(chosen_input_ids))  # The same input_ids are used for both chosen and rejected, as they share the prompt
            self.decoder_input_ids_chosen.append(torch.tensor(chosen_input_ids))
            self.decoder_input_ids_rejected.append(torch.tensor(rejected_input_ids))
            self.attn_masks.append(torch.tensor(chosen_attention_mask))  # Same attention mask can be used for both chosen and rejected


    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return (
            self.input_ids[idx],
            self.decoder_input_ids_chosen[idx],
            self.decoder_input_ids_rejected[idx],
            self.attn_masks[idx],
        )


class DataCollatorReward:
    def __call__(self, data):
        batch = {}
        batch["input_ids"] = torch.cat([f[0].unsqueeze(0) for f in data], dim=0)
        batch["decoder_input_ids"] = torch.cat([f[1].unsqueeze(0) for f in data] + [f[2].unsqueeze(0) for f in data], dim=0)
        batch["attention_mask"] = torch.cat([f[3].unsqueeze(0) for f in data] + [f[3].unsqueeze(0) for f in data], dim=0)
        batch["labels"] = torch.tensor([0] * len(data) + [1] * len(data))
        return batch


# The model is loaded using `AutoModelForSeq2SeqLM` which is appropriate for BART.
# The linear layer (`self.v_head`) is adjusted to match the dimensions of BART's hidden states (`self.config.d_model`).
# The forward pass now accounts for both `input_ids` (for the encoder) and `decoder_input_ids` (for the decoder).

class BARTRewardModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.model = AutoModelForSeq2SeqLM.from_pretrained(config)
        self.config = self.model.config
        self.v_head = nn.Linear(self.config.d_model, 1, bias=False) # d_model is used for BART
        self.tokenizer = AutoTokenizer.from_pretrained(config)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.PAD_ID = self.tokenizer(self.tokenizer.pad_token)["input_ids"][0]


    def forward(self,
                input_ids=None,
                attention_mask=None,
                decoder_input_ids=None,
                labels=None,
                return_dict=False,
                output_attentions=False,
                output_hidden_states=False):
        # Add decoder-specific inputs, processing, and outputs
        model_outputs = self.model(input_ids=input_ids,
                                   attention_mask=attention_mask,
                                   decoder_input_ids=decoder_input_ids,
                                   return_dict=True)

        # Use the last hidden state from decoder
        hidden_states = model_outputs.decoder_last_hidden_state
        
        rewards = self.v_head(hidden_states).squeeze(-1)
        reward_scores = []
        bs = input_ids.shape[0] // 2
	# Note half is chosen and another half is rejected.
        chosen = input_ids[:bs]
        rejected = input_ids[bs:]
        chosen_rewards = rewards[:bs]
        rejected_rewards = rewards[bs:]
        # compute pairwise loss. Only backprop on last value before padding
        loss = 0
        for i in range(bs):
            # Find the index of the first occurrence where chosen summary input_ids
	        # and rejected summary input_ids are different.
            divergence_ind = (chosen[i] != rejected[i]).nonzero()[0]


	    # Find the index of the first occurrence of the padding token the chosen summary.
            c_inds = (chosen[i] == self.PAD_ID).nonzero()
            c_ind = c_inds[0].item() if len(c_inds) > 0 else chosen.shape[1]


	    # Find the index of the first occurrence of the padding token the rejected summary.
            r_inds = (rejected[i] == self.PAD_ID).nonzero()
            r_ind = r_inds[0].item() if len(r_inds) > 0 else rejected.shape[1]
            end_ind = max(c_ind, r_ind)
        
	    # Find the slice of reward which belongs to diverging input_ids
            c_truncated_reward = chosen_rewards[i][divergence_ind:end_ind]
            r_truncated_reward = rejected_rewards[i][divergence_ind:end_ind]
            reward_scores.append(c_truncated_reward[-1])  # reward at last token
            
            # Compute loss
            loss += -torch.log(
                torch.sigmoid(c_truncated_reward - r_truncated_reward)
            ).mean()
            loss = loss / bs
        return {"loss": loss, "reward_scores": torch.stack(reward_scores)}


def compute_metrics(eval_preds):
    chosen_end_scores = eval_preds.predictions[0]  # chosen scores
    rejected_end_scores = eval_preds.predictions[1]  # rejected scores

    result = {}
    acc = sum(chosen_end_scores > rejected_end_scores) / len(rejected_end_scores)
    result["accuracy"] = acc

    return result

if __name__ == "__main__":

    sft_model_checkpoint = "../models/BART-SFT-r1/checkpoint-4200/"
    tokenizer = AutoTokenizer.from_pretrained(sft_model_checkpoint)


    tokenizer.pad_token = tokenizer.eos_token


    if not os.path.exists("rm_checkpoint"):
        os.mkdir("rm_checkpoint")

    batch_size = 2
    training_args = TrainingArguments(
        output_dir="rm_checkpoint/",
        num_train_epochs=5,
        logging_steps=5,
        gradient_accumulation_steps=4,
        save_strategy="steps",
        evaluation_strategy="steps",
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        eval_accumulation_steps=1,
        eval_steps=500,
        save_steps=500,
        warmup_steps=100,
        logging_dir="./logs",
        fp16=True,
        bf16=False,
        learning_rate=1e-5,
        save_total_limit=1,
        # deepspeed = "ds_config.json"
    )


    rm = BARTRewardModel(sft_model_checkpoint)

    # Freeze the first 70% of the hidden layers for both encoder and decoder
    encoder_layers = rm.model.model.encoder.layers
    decoder_layers = rm.model.model.decoder.layers
    num_layers = len(encoder_layers) # same num of enc/dec layers
    num_unfrozen = int(0.3 * num_layers)
    for layer in encoder_layers[:-num_unfrozen] + decoder_layers[:-num_unfrozen]:
        for param in layer.parameters():
            param.requires_grad = False


    data_path = "comparisons_dataset.csv"
    dataset = load_dataset("csv", data_files=data_path)
    dataset = dataset['train'].train_test_split(test_size=0.1, seed=42) # test is actually validation

    max_length = 1024
    train_dataset = PairwiseDataset(dataset['train'], tokenizer, max_length=max_length)
    val_dataset = PairwiseDataset(dataset['test'], tokenizer, max_length=max_length)

    data_collator = DataCollatorReward()

    trainer = Trainer(
        model=rm,
        args=training_args,
        train_dataset=train_dataset,
        compute_metrics=compute_metrics,
        eval_dataset=val_dataset,
        data_collator=data_collator,
    ).train()
