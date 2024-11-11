import torch
import transformers
model_checkpoint = "t5-small"

from datasets import load_dataset
from evaluate import load
import nltk
nltk.download('punkt')

raw_datasets = load_dataset("xsum")
metric = load("rouge")
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
if model_checkpoint in ["t5-small","t5-base","t5-larg","t5-3b","t5-lib"]:
    prefix = "summarize: "
else:
    prefix = ""

def preprocess_function(batch):
    inputs = [prefix + doc for doc in batch["document"]]
    model_inputs = tokenizer(inputs,padding=True,truncation=True)

    with tokenizer.as_target_tokenizer():
        labels = tokenizer(batch["summary"],padding=True,truncation=True)
    
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

tokenized_datasets = raw_datasets.map(preprocess_function,batched=True)

from transformers import AutoModelForSeq2SeqLM,DataCollatorForSeq2Seq, Seq2SeqTrainingArguments,Seq2SeqTrainer
model = AutoModelForSeq2SeqLM.from_pretrained(model_checkpoint)

BATCH_SIZE = 16

model_name = model_checkpoint.split("/")[-1]
args = Seq2SeqTrainingArguments(
    "test-summarization",
    evaluation_strategy = "epoch",
    learning_rate = 2e-5,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size = BATCH_SIZE,
    weight_decay=0.01,
    save_total_limit = 3,
    num_train_epochs = 1,
    predict_with_generate = True,
    fp16=True
)

data_collator = DataCollatorForSeq2Seq(tokenizer,model=model)

import numpy as numpy
nltk.download('punkt')

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    decoded_preds = tokenizer.batch_decode(predictions,skip_special_tokens=True)

    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels,skip_special_tokens=True)

    decoded_preds = ['\n'.join(nltk.sent_tokenize(pred.strip())) for pred in decoded_preds]
    decoded_labels = ['\n'.join(nltk.sent_tokenize(label.strip())) for label in decoded_labels]

    result = metric.compute(predictions = decoded_preds, references = decoded_labels, user_stemmer=True)
    result = {key: value.mid.fmeasure *100 for key, value in result.items()}
    
    #에측 길이 평균
    predictions_lens = [np.count_nonzero(pred != tokenizer.pad_token_id) for pred in predictions]
    result['gen_len'] = np.mean(predictions_lens)

    return {k: rount(v,4) for k,v in result.items()}

trainer = Seq2SeqTrainer(
    model,
    args,
    train_dataset = tokenized_datasets["train"],
    eval_dataset = tokenized_datasets["validation"],
    data_collator = data_collator,
    tokenizer = tokenizer,
    compute_metrics =  compute_metrics
)

trainer.train()

pred = trainer.predict(tokenized_datasets['test'])
print(compute_metrics(pred.predictions, tokenized_datasets['labels']))
