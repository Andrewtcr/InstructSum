# %%
import transformers
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
)
from datasets import load_dataset, load_from_disk, DatasetDict
import numpy as np
from evaluate import load
import nltk

nltk.download("punkt")

max_input = 1024
max_target = 128

model_checkpoint = "facebook/bart-large-cnn"
model = AutoModelForSeq2SeqLM.from_pretrained(model_checkpoint)
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)

metric = load("rouge")


def preprocess_data(examples):
    # get all the articles, prepend each with "bias;"
    inputs = [
        f"{bias}; {article}"
        for bias, article in zip(examples["summary_bias"], examples["article"])
    ]
    # tokenize the inputs
    model_inputs = tokenizer(
        inputs, max_length=max_input, padding="max_length", truncation=True
    )

    # tokenize the summaries
    targets = tokenizer(
        examples["summary"],
        max_length=max_target,
        padding="max_length",
        truncation=True,
    )

    # set labels
    model_inputs["labels"] = targets["input_ids"]
    # return the tokenized data
    # input_ids, attention_mask and labels
    return model_inputs


def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
    # Replace -100 in the labels as we can't decode them.
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    # Rouge expects a newline after each sentence
    decoded_preds = [
        "\n".join(nltk.sent_tokenize(pred.strip())) for pred in decoded_preds
    ]
    decoded_labels = [
        "\n".join(nltk.sent_tokenize(label.strip())) for label in decoded_labels
    ]

    # Note that other metrics may not have a `use_aggregator` parameter
    # and thus will return a list, computing a metric for each sentence.
    result = metric.compute(
        predictions=decoded_preds,
        references=decoded_labels,
        use_stemmer=True,
        use_aggregator=True,
    )
    # Extract a few results
    result = {key: value * 100 for key, value in result.items()}

    # Add mean generated length
    prediction_lens = [
        np.count_nonzero(pred != tokenizer.pad_token_id) for pred in predictions
    ]
    result["gen_len"] = np.mean(prediction_lens)

    return {k: round(v, 4) for k, v in result.items()}


def main():
    raw_datasets = load_from_disk("../data/hf_dataset")

    # Define the sample sizes
    train_sample_size = 50  # Adjust as needed
    val_sample_size = 5  # Adjust as needed
    test_sample_size = 5  # Adjust as needed

    # Sample the datasets
    sampled_datasets = DatasetDict(
        {
            "train": raw_datasets["train"]
            .shuffle(seed=42)
            .select(range(train_sample_size)),
            "validation": raw_datasets["validation"]
            .shuffle(seed=42)
            .select(range(val_sample_size)),
            "test": raw_datasets["test"]
            .shuffle(seed=42)
            .select(range(test_sample_size)),
        }
    )

    tokenized_data = sampled_datasets.map(preprocess_data, batched=True)

    batch_size = 4
    model_name = model_checkpoint.split("/")[-1]
    args = Seq2SeqTrainingArguments(
        f"{model_name}-finetuned",
        evaluation_strategy="steps",
        eval_steps=0.25,
        learning_rate=2e-5,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        weight_decay=0.01,
        log_level="info",
        logging_dir="./log",
        logging_first_step=True,
        logging_steps=5,
        save_total_limit=5,
        save_strategy="steps",
        save_steps=10,
        load_best_model_at_end=True,
        num_train_epochs=1,
        predict_with_generate=True,
        fp16=True,
    )

    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

    trainer = Seq2SeqTrainer(
        model,
        args,
        train_dataset=tokenized_data["train"],
        eval_dataset=tokenized_data["validation"],
        data_collator=data_collator,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
    )

    trainer.train()
    trainer.save_model("output")
    trainer.save_state()


if __name__ == "__main__":
    main()
