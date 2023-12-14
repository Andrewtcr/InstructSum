import transformers
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
    BartConfig,
    TrainerCallback
)
from datasets import load_dataset, load_from_disk, DatasetDict
import numpy as np
from evaluate import load
import nltk

nltk.download("punkt")

max_input = 1024
max_target = 128

model_checkpoint = "facebook/bart-large-cnn"
config = BartConfig.from_pretrained(model_checkpoint)

# Add dropout
# config.dropout= 0.5
# config.attention_dropout = 0.5

model = AutoModelForSeq2SeqLM.from_pretrained(model_checkpoint, config=config)
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)

metric = load("rouge")


class EarlyStoppingCallback(TrainerCallback):
    def __init__(self, early_stopping_patience: int, early_stopping_threshold: float, metric: str):
        self.early_stopping_patience = early_stopping_patience
        self.early_stopping_threshold = early_stopping_threshold
        self.metric = metric
        self.best_score = None
        self.patience_counter = 0

    def on_evaluate(self, args, state, control, **kwargs):
        # Check if the current evaluation loss is better (less) than the previous best
        current_score = kwargs["metrics"][self.metric]
        if self.best_score is None or current_score < self.best_score - self.early_stopping_threshold:
            self.best_score = current_score
            self.patience_counter = 0
        else:
            self.patience_counter += 1

        # Check if patience has run out
        if self.patience_counter >= self.early_stopping_patience:
            print("Early stopping triggered")
            control.should_training_stop = True



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
    raw_datasets = load_from_disk("/home/andrew/InstructSum/data/hf_dataset")
    raw_datasets.shuffle(seed=42)

    tokenized_data = raw_datasets.map(preprocess_data, batched=True)

    batch_size = 4
    model_name = "BART-SFT-r1"
    eval_metric = "rouge1"
    args = Seq2SeqTrainingArguments(
        model_name,
        evaluation_strategy="steps",
        eval_steps=200,
        warmup_steps=500,
        learning_rate=5e-6,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        weight_decay=0.01,
        log_level="info",
        logging_dir=f"./logs/{model_name}",
        logging_first_step=True,
        logging_steps=5,
        save_total_limit=3,
        save_strategy="steps",
        save_steps=200,
        load_best_model_at_end=True,
        metric_for_best_model=eval_metric,
        num_train_epochs=10,
        predict_with_generate=True,
        fp16=True,
    )
    
    early_stopping_callback = EarlyStoppingCallback(
        early_stopping_patience=5,  # Number of evaluations to wait without improvement
        early_stopping_threshold=0.0,  # Minimum change to qualify as an improvement
        metric=eval_metric
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
        # callbacks=[early_stopping_callback]
    )

    trainer.train(resume_from_checkpoint=True)
    # trainer.save_model("output")
    # trainer.save_state()


if __name__ == "__main__":
    main()
