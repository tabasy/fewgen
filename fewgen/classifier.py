import re, copy
import os, re, json
from random import choice
import logging

import torch
import numpy as np
from datasets import Dataset, DatasetDict
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, ConfusionMatrixDisplay, plot_confusion_matrix
from sklearn.pipeline import Pipeline
from sklearn.base import clone
from matplotlib import pyplot as plt

from fewgen.dataset import prepare_dataset, get_dataset_label_names
from fewgen.vectorizer import vectorize_by_descriptions, vectorize_by_examples, vectorize_by_nli
from fewgen.description import Description, save_descriptions, load_descriptions
from fewgen.finetune import finetune_lm, generate_finetuning_data


class FewgenClassifier:
  
  def __init__(self, descriptions, base_model_name=None, language_model=None, tokenizer=None):
    
    self.tokenizer = tokenizer or AutoTokenizer.from_pretrained(base_model_name)
    self.lm = language_model or AutoModel.from_pretrained(base_model_name)
    
    self.labels = list(descriptions.keys())
    self.label2i = {lb: i for i, lb in enumerate(self.labels)}
    class_descriptions = {}
    
    for c, man_descs in descriptions.items():
      class_descriptions[c] = []
      for man_desc in man_descs:
        desc = Description.from_text(tokenizer=self.tokenizer, full=man_desc)
        class_descriptions[c].append(desc)
        
    self.descriptions = sum(class_descriptions.values(), [])
    self.class_descriptions = class_descriptions
        
    self.classifier = Pipeline([
      ('scaler', StandardScaler()), 
      ('logreg', LogisticRegression(solver='saga', penalty='l2', C=0.1))
    ])
                         
  def finetune_lm(self, dataset, finetune_args=None):
      
    finetune_dataset = generate_finetuning_data({
      'trainset': dataset,
      'testset': dataset,
      'class_descriptions': self.class_descriptions})
                  
    ft_kwargs = finetune_args if isinstance(finetune_args, dict) else {}          
    finetune_lm(self.lm, self.tokenizer, finetune_dataset, **ft_kwargs)
                         
  def train(self, dataset, finetune_lm=False, finetune_args=None):

    if finetune_lm:
      self.finetune_lm(dataset, finetune_args)

    train_x = vectorize_by_descriptions(dataset['text'], self.lm, self.descriptions)
    train_y = np.array([self.label2i[lb] for lb in dataset['label']])
    
    self.classifier.fit(train_x, train_y)
    train_preds = self.classifier.predict(train_x)
    
    results =  {
      'train_acc': accuracy_score(train_y, train_preds),
      'train_f1': f1_score(train_y, train_preds, average='macro'),
    }
    
    logging.info(f'train accuracy: {results["train_acc"]:.1%}')
    logging.info(f'train f-score: {results["train_f1"]:.1%}')
  
    return results  

  def test(self, dataset):
    
    test_x = vectorize_by_descriptions(dataset['text'], self.lm, self.descriptions)
    test_y = np.array([self.label2i[lb] for lb in dataset['label']])
    
    test_preds = self.classifier.predict(test_x)
    
    results =  {
      'test_acc': accuracy_score(test_y, test_preds),
      'test_f1': f1_score(test_y, test_preds, average='macro'),
    }
    
    logging.info(f'test accuracy: {results["test_acc"]:.1%}')
    logging.info(f'test f-score: {results["test_f1"]:.1%}')
  
    return results
                         
  def predict_proba(self, texts):
    
    texts = [texts] if isinstance(texts, str) else texts
    
    features = vectorize_by_descriptions(texts, self.lm, self.descriptions)
    probs = self.classifier.predict_proba(features)
    return probs
    
  def classify(self, texts):
    
    probs = self.predict_proba(texts)
    return [self.labels[pred] for pred in probs.argmax(axis=-1)]

  evaluate = test
  fit = train
  predict = classify
  