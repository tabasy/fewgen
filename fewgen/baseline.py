import re
import logging

import numpy as np
from datasets import Dataset, DatasetDict
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.manifold import TSNE
from sklearn.metrics import accuracy_score, f1_score, ConfusionMatrixDisplay, plot_confusion_matrix
from matplotlib import pyplot as plt

from fewgen.dataset import prepare_dataset, get_dataset_label_names
from fewgen.vectorizer import vectorize_by_embedding
from fewgen.description import Description
from fewgen.generator import DiverseDescriptionGenerator
from fewgen.finetune import finetune_clf_lm
from fewgen.util import load_model, load_clf_model
from fewgen.experiment import run_classifier, plot_features, sweep_classifiers

default_config = {
  'vectorizer_model': {'model_name': 'gpt2', 'device': 'cuda', 
                       'finetune': {'early_stopping_threshold': 0.01, 'epochs': 30}},
  
  'device': 'cuda',
  'dataset': {
    'dataset_name': 'glue/sst2',
    'shuffle': False, 
    'shuffle_seed': 0,
    'train_ex_per_class': 16,
    'test_ex_per_class': 64,
    'test_split_name': 'validation',
  },
  'prompt': '',
  'feature_mode': 'avg',  # avg_emb, last_emb, next_probs, finetuned_classifier
  'feature_processor': StandardScaler(),
  'classifier': LogisticRegression(solver='newton-cg'),
  'plot': True,
}

  
def experiment_baseline(cfg):
  prompt = cfg.get('prompt')
  vec_model_name = cfg['vectorizer_model']['model_name']
  
  dataset_name = cfg['dataset']['dataset_name']
  
  logging.info(f'loading {dataset_name} dataset')
  trainset, testset = prepare_dataset(**cfg['dataset'], prompt=prompt)
  label_names = get_dataset_label_names(trainset)
  
  train_y, test_y = np.array(trainset['label']), np.array(testset['label'])
  train_texts, test_texts = trainset['text'], testset['text']

  feature_mode = cfg['feature_mode']
  
  if cfg['vectorizer_model'].get('finetune', False):
    logging.info(f'load {vec_model_name} model')
    finetuning_dataset = DatasetDict({'train': trainset, 'validation': trainset})
    vec_tk, vec_lm = load_clf_model(vec_model_name, num_labels=len(np.unique(train_y)),
                                    device=cfg['vectorizer_model']['device'])
    logging.info(f'finetune {vec_model_name} as a sequence classifier')

    if 'batch_size' not in cfg['vectorizer_model']['finetune']:
        cfg['vectorizer_model']['finetune']['batch_size'] = 2 if 'medium' in vec_model_name else 4
    finetune_clf_lm(vec_lm, vec_tk, finetuning_dataset, **cfg['vectorizer_model']['finetune'])
  else:
    logging.info(f'load {vec_model_name} language model')
    vec_tk, vec_lm = load_model(vec_model_name, device=cfg['vectorizer_model']['device'])

  logging.info(f'get features')
  train_x = vectorize_by_embedding(train_texts, vec_lm, vec_tk, mode=feature_mode)
  test_x = vectorize_by_embedding(test_texts, vec_lm, vec_tk, mode=feature_mode)
  
  logging.info(f'train features shape: {train_x.shape}')
  logging.info(f'test features shape: {test_x.shape}')
  
  preprocessor, classifier = cfg['feature_processor'], cfg['classifier']

  if preprocessor is not None:
    if hasattr(preprocessor, 'fit'):
      preprocessor.fit(train_x)
    train_x = preprocessor.transform(train_x)
    test_x = preprocessor.transform(test_x)
    logging.info(f'train features reduced to: {train_x.shape}')
    logging.info(f'test features reduced to: {test_x.shape}')   
    
  clf_result = run_classifier(cfg['classifier'], train_x, train_y, test_x, test_y)
  train_acc, test_acc = clf_result['train_acc'], clf_result['test_acc']
  train_f1, test_f1 = clf_result['train_f1'], clf_result['test_f1']
  
  if cfg['plot']:
    plot_features(train_x, train_y, names=label_names, tsne=False,
                  title=f'{dataset_name}/{vec_model_name}: train acc: {train_acc}\ttest acc: {test_acc}')
    plot_features(test_x, test_y, names=label_names, tsne=False, 
                  title=f'{dataset_name}/{vec_model_name}: train f1: {train_f1}\ttest f1: {test_f1}')
    
  logging.info(f'train acc: {train_acc}\t\ttest acc: {test_acc}')
  logging.info(f'train f1: {train_f1}\t\ttest f1: {test_f1}')
  
  return {
    'trainset': trainset,
    'testset': testset,
    'label_names': label_names,
    'train_x': train_x,
    'test_x': test_x,
    'train_y': train_y,
    'test_y': test_y,
    'feature_processor': preprocessor,
    'classifier': classifier,
    'train_acc': train_acc, 
    'test_acc': test_acc, 
    'train_f1': train_f1, 
    'test_f1': test_f1, 
  }