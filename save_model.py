from transformers import BertTokenizer, TFBertForSequenceClassification, pipeline
import tensorflow as tf
import datasets

train_data, test_data  = datasets.load_dataset("imdb", split =['train', 'test'])
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
model = TFBertForSequenceClassification.from_pretrained("bert-base-uncased")

# inputs = tokenizer("Hello, my dog is cute", return_tensors="tf")

pipe = pipeline(task="text-classification", model=model, framework='tf', tokenizer=tokenizer)

print(test_data['text'])