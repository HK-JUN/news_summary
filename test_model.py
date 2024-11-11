from transformers import AutoModelForSeq2SeqLM,DataCollatorForSeq2Seq,AutoTokenizer

checkpoint_path = "/home/user3/workplace/summary/test-summarization/checkpoint-12000"
model_name = "t5-small"
model = AutoModelForSeq2SeqLM.from_pretrained(checkpoint_path)
tokenizer = AutoTokenizer.from_pretrained(model_name)