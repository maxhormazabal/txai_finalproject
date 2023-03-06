#Import own functions
from txai_utils import *

# Libraries
import numpy as np
from transformers import AutoModelForSequenceClassification, AutoTokenizer, DataCollatorWithPadding, TrainingArguments, Trainer
from datasets import load_dataset, Value, Features, ClassLabel, load_metric
import torch
import torch.nn as nn
import transformers
import shap

def countClasses(df):
  sum = 0
  for element in df.veracity.value_counts(): sum = sum + element
  for i in range(len(df.veracity.value_counts().index)): print(str(df.veracity.value_counts().index[i]) + " -> " + str(round(100*df.veracity.value_counts()[i]/sum, 1)))

def tokenize_function(batch,tokenizer,colname):
    return tokenizer(batch[colname], truncation=True)

def tokenize_function_trip(batch,tokenizer):
    return tokenizer(batch['review_coment'], truncation=True)

# Function that evaluates the performance of our classification model

def compute_metrics(eval_preds):
    metric1 = load_metric("accuracy")
    metric2 = load_metric("f1")
    logits, labels = eval_preds
    predictions = np.argmax(logits, axis=-1)
    accuracy = metric1.compute(predictions=predictions, references=labels)["accuracy"]
    f1c = metric2.compute(predictions=predictions, references=labels, average=None)["f1"]
    return {"accuracy": accuracy, "f1c": f1c}

def generateExplainer(model_file,pretrained_type,process_type = 'sentiment-analysis',runner = -1):
    # Load the PyTorch model and tokenizer
    model = torch.load(model_file)
    tokenizer = AutoTokenizer.from_pretrained(pretrained_type)

    # Transform the model into a pipeline
    model = transformers.pipeline(
        process_type,
        model=model,
        tokenizer=tokenizer,
        device=runner # use 0 for the first GPU, -1 for CPU
    )

    # explain the model on two sample inputs
    return(shap.Explainer(model))

def visualizeExplanation(explainer,text,target = 1):
    shap_values = explainer([text])
    shap.plots.text(shap_values[0,:,target])
