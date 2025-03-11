import gradio as gr
import pandas as pd
from transformers import pipeline
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification

df = pd.read_csv("WhatsApp.csv")

tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased-finetuned-sst-2-english')
model = DistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased-finetuned-sst-2-english')

nlp = pipeline("sentiment-analysis", model=model, tokenizer=tokenizer)

def sentiment_analysis_model(review):  
  res = nlp(review)
  label = res[0]['label']
  score = res[0]['score']
  output = f"{label}, Score: {score}"
  return output

demo = gr.Interface(fn=sentiment_analysis_model, inputs="text", outputs="text", title="SENTIMENT ANALYSIS using DistilBert model", 
                    description="Sentiment Analysis using Pretrained DistilBert model. The DistilBert model is trained on a sample data of google playstore reviews for WhatsApp application. Provide a review and the model will classify it to be POSITIVE or NEGATIVE")
demo.launch()
