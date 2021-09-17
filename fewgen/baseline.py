import re
import logging

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from matplotlib import pyplot as plt
from sklearn.preprocessing import StandardScaler

from fewgen.dataset import prepare_dataset
from fewgen.vectorizer import vectorize_by_embedding
from fewgen.description import Description
from fewgen.util import load_model
from fewgen.experiment import plot_features, sweep_classifiers


default_config = {
  'vectorizer_lm': {'model_name': 'gpt2', 'device': 'cuda'},
  
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
  'feature_mode': 'avg',  # avg_emb, last_emb, next_word
  'feature_reducer': PCA(n_components=2),
  'classifier': LogisticRegression(solver='newton-cg'),
  'plot': True,
}

  
def experiment_baseline(cfg):
  vec_model_name = cfg['vectorizer_lm']['model_name']
  prompt = cfg.get('prompt')
  
  dataset_name = cfg['dataset']['dataset_name']
  
  logging.info(f'load {vec_model_name} language model')
  vec_tk, vec_lm = load_model(vec_model_name, device=cfg['vectorizer_lm']['device'])
  
  logging.info(f'loading {dataset_name} dataset')
  trainset, testset = prepare_dataset(**cfg['dataset'])
  
  train_y, test_y = np.array(trainset['label']), np.array(testset['label'])

  logging.info(f'get features')

  train_texts, test_texts = trainset['text'], testset['text']
  if prompt is not None and len(prompt.strip()) > 0:
    prompt = ' ' + prompt.strip()
    train_texts = [t + prompt for t in train_texts]
    test_texts = [t + prompt for t in test_texts]

  feature_mode = cfg['feature_mode']
  train_x = vectorize_by_embedding(train_texts, vec_lm, vec_tk, mode=feature_mode)
  test_x = vectorize_by_embedding(test_texts, vec_lm, vec_tk, mode=feature_mode)
  
  logging.info(f'train features shape: {train_x.shape}')
  logging.info(f'test features shape: {test_x.shape}')
  
  reducer, classifier = cfg['feature_reducer'], cfg['classifier']
  
  if reducer is not None:
    train_x = reducer.fit_transform(train_x)
    test_x = reducer.transform(test_x)
    logging.info(f'train features reduced to: {train_x.shape}')
    logging.info(f'test features reduced to: {test_x.shape}')   
    
  classifier.fit(train_x, train_y)
  train_preds = classifier.predict(train_x)  
  test_preds = classifier.predict(test_x)
  
  train_acc = accuracy_score(train_y, train_preds)
  test_acc = accuracy_score(test_y, test_preds)  
  
  if cfg['plot']:
    plot_features(train_x, train_y, tsne=False,
                  title=f'{dataset_name}/{vec_model_name}: train acc: {train_acc}')
    plot_features(test_x, test_y, tsne=False,
                  title=f'{dataset_name}/{vec_model_name}: test acc: {test_acc}')
    
  logging.info(f'train acc: {train_acc}')
  logging.info(f'test acc: {test_acc}')
  
  return {'train_x': train_x,
          'test_x': test_x,
          'train_y': train_y,
          'test_y': test_y,
          'classifier': classifier,
          'train_acc': train_acc, 
          'test_acc': test_acc, 
          }
