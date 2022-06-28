from transformers import BertTokenizer, TFBertForSequenceClassification, pipeline

model = TFBertForSequenceClassification.from_pretrained("bert-base-uncased")
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

model.save_pretrained('./bert-base-uncased')
tokenizer.save_pretrained('./bert-base-uncased')