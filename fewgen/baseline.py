import os, copy
import re, json
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
from sklearn.pipeline import Pipeline
from matplotlib import pyplot as plt

from fewgen.dataset import prepare_dataset, get_dataset_label_names
from fewgen.vectorizer import vectorize_by_embedding
from fewgen.description import Description
from fewgen.generator import DiverseDescriptionGenerator
from fewgen.finetune import finetune_clf_lm
from fewgen.util import load_model, load_nli, load_clf_model, run_classifier, plot_features
from fewgen.experiment import sweep_classifiers


EXP_DIR = 'experiments'
BASELINE_LOG_PATH = os.path.join(EXP_DIR, 'baseline.jsonl')


default_config = {
  'model': {
    'generator_model_name': 'gpt2',
    'vectorizer_device': 'cuda',
    'finetune': False,
    'feature_mode': 'avg'
  },
  
  'vectorizer_model': {'model_name': 'gpt2', 'device': 'cuda', 
                       'finetune': False},
  'dataset': {
    'dataset_name': 'glue/sst2',
    'shuffle': True, 
    'shuffle_seed': 110,
    'train_ex_per_class': 16,
    'test_ex_per_class': 64,
    'test_split_name': 'test',
  },
  'prompt': '',
  'feature_mode': 'avg',  # avg_emb, last_emb, next_probs, finetuned_classifier

  'classifier': {'no_l2': Pipeline([('scaler', StandardScaler()), ('logreg', LogisticRegression(solver='saga', penalty='none'))]),
                 'l2_1.0': Pipeline([('scaler', StandardScaler()), ('logreg', LogisticRegression(solver='saga', penalty='l2', C=1.0))]),
                 'l2_0.1': Pipeline([('scaler', StandardScaler()), ('logreg', LogisticRegression(solver='saga', penalty='l2', C=0.1))]),
                 'l2_0.01': Pipeline([('scaler', StandardScaler()), ('logreg', LogisticRegression(solver='saga', penalty='l2', C=0.01))])
                },
  'plot': True,
}


def get_default_cfg():
  return copy.deepcopy(default_config)

  
def experiment_baseline(cfg):
  
  model_cfg, data_cfg = cfg['model'], cfg['dataset']
  
  vec_model_name = model_cfg['vectorizer_model_name']
  prompt = cfg['prompt']
  
  dataset_name = data_cfg['dataset_name'].replace('/', '_')
  
  logging.info(f'loading {dataset_name} dataset')
  trainset, testset = prepare_dataset(**cfg['dataset'], prompt=prompt)
  label_names = get_dataset_label_names(trainset)
  
  train_y, test_y = np.array(trainset['label']), np.array(testset['label'])
  train_texts, test_texts = trainset['text'], testset['text']
  
  if model_cfg['finetune']:
    logging.info(f'load {vec_model_name} model')
    finetuning_dataset = DatasetDict({'train': trainset, 'validation': trainset})
    vec_tk, vec_lm = load_clf_model(vec_model_name, num_labels=len(np.unique(train_y)),
                                    device=cfg['vectorizer_model']['device'])
    logging.info(f'finetune {vec_model_name} as a sequence classifier')

    ft_kwargs = model_cfg['finetune']
    ft_kwargs = ft_kwargs if isinstance(ft_kwargs, dict) else {}
    if 'batch_size' not in ft_kwargs:
      if 'large' in vec_model_name:
        ft_kwargs['batch_size'] = 1 
      elif 'medium' in vec_model_name:
        ft_kwargs['batch_size'] = 2 
      else:
        ft_kwargs['batch_size'] = 4 

    finetune_clf_lm(vec_lm, vec_tk, finetuning_dataset, **ft_kwargs)
  else:
    logging.info(f'load {vec_model_name} language model')
    vec_tk, vec_lm = load_model(vec_model_name, device=cfg['vectorizer_model']['device'])

  logging.info(f'get features')
  feature_mode = model_cfg['feature_mode']
  
  train_x = vectorize_by_embedding(train_texts, vec_lm, vec_tk, mode=feature_mode)
  test_x = vectorize_by_embedding(test_texts, vec_lm, vec_tk, mode=feature_mode)
  
  logging.info(f'train features shape: {train_x.shape}')
  logging.info(f'test features shape: {test_x.shape}')

  classifier = cfg['classifier']
  
  train_acc, test_acc = 0, 0
  train_f1, test_f1 = 0, 0
  
  clf_results, classifiers = {}, {}
  main_clf_name = 'l2_1.0'
  
  for clf_name, clf in cfg['classifier'].items():
    clf_results[clf_name] = run_classifier(cfg['classifier'][clf_name], train_x, train_y, test_x, test_y)
    classifiers[clf_name] = clf_results[clf_name].pop('classifier')
    
    experiment_log = clf_results[clf_name].copy()
    experiment_log['classifier'] = clf_name
  
    for sub_cfg in [model_cfg, data_cfg]:
      experiment_log.update(sub_cfg)
    experiment_log['feature_mode'] = feature_mode

    with open(BASELINE_LOG_PATH, 'a') as logf:
      logf.write(json.dumps(experiment_log) + '\n')
      
  train_acc, test_acc = clf_results[main_clf_name]['train_acc'], clf_results[main_clf_name]['test_acc']
  train_f1, test_f1 = clf_results[main_clf_name]['train_f1'], clf_results[main_clf_name]['test_f1']
  
  if cfg['plot']:
    plot_features(train_x, train_y, names=label_names, tsne=False,
                  title=f'{dataset_name}/{vec_model_name}: train acc: {train_acc}\ttest acc: {test_acc}')
    plot_features(test_x, test_y, names=label_names, tsne=False, 
                  title=f'{dataset_name}/{vec_model_name}: train f1: {train_f1}\ttest f1: {test_f1}')
    
  logging.info(f'train acc: {train_acc}\t\ttest acc: {test_acc}')
  logging.info(f'train f1: {train_f1}\t\ttest f1: {test_f1}')

  results = {
    'trainset': trainset,
    'testset': testset,
    'label_names': label_names,
    'train_x': train_x,
    'test_x': test_x,
    'train_y': train_y,
    'test_y': test_y,
    'classifier': classifiers[main_clf_name],
    'train_acc': train_acc, 
    'test_acc': test_acc, 
    'train_f1': train_f1, 
    'test_f1': test_f1, 
    'clf_results': clf_results,
  }
    
  return results