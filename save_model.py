from transformers import BertTokenizer, TFBertForSequenceClassification, pipeline
import tensorflow as tf
import datasets

# from tensorflow.keras.datasets import imdb
#
# (X_train, y_train), (X_test, y_test) = imdb.load_data()

train_data, test_data  = datasets.load_dataset("imdb", split =['train', 'test'])
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
model = TFBertForSequenceClassification.from_pretrained("bert-base-uncased")

model.save_pretrained('./bert-base')

# inputs = tokenizer("Hello, my dog is cute", return_tensors="tf")

pipe = pipeline(task="text-classification", model=model, framework='tf', tokenizer=tokenizer)

print(test_data['label'][:512])
print(pipe(test_data['text'][:512]))