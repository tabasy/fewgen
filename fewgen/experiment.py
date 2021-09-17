import re
import logging

import torch
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
from fewgen.vectorizer import vectorize_by_descriptions, vectorize_by_examples, vectorize_by_nli
from fewgen.description import Description
from fewgen.generator import DiverseDescriptionGenerator
from fewgen.util import load_model
from fewgen.nli import load_nli


sent_end_re = re.compile(r'[.!]')


def satisfaction(desc):
  text = desc.get_text()
  has_punc = len(sent_end_re.findall(text)) > 0
  return has_punc


default_config = {
  'generator_model': {'model_name': 'gpt2', 'device': 'cuda'},
  'vectorizer_model': {'model_name': 'gpt2', 'device': 'cuda'},
  
  'device': 'cuda',
  'dataset': {
    'dataset_name': 'glue/sst2',
    'shuffle': False, 
    'shuffle_seed': 0,
    'train_ex_per_class': 16,
    'test_ex_per_class': 64,
    'test_split_name': 'validation',
  },
  'description': {'eps': 5e-3},
  'generator': {
    'prompt': ' All in all, the movie was',
    'beam_size': 8,
    'num_beam_groups': 4,
    'min_len': 2,
    'max_len': 5,
    'satisfaction': satisfaction,
    'diversity_factor': 0.95,
    'stop_if_satisfied': False,
    'keep_longest': False,
    'group_best_only': True,
    'log': False
  },
  'feature_mode': 'ppl',
  'feature_reducer': PCA(n_components=2),
  'classifier': LogisticRegression(solver='newton-cg'),
  'plot': True,
}


def plot_features(features, labels, tsne=True, title=None):
  if title:
    plt.title(title)
  if tsne and features.shape[-1] > 2:
    features = TSNE(n_components=2, perplexity=len(features)/len(np.unique(labels))).fit_transform(features)
  
  if features.shape[-1] >= 2:
    plt.scatter(features[:, 0], features[:, 1], c=labels)
  else:
    for i in np.unique(labels):
      plt.hist(features[labels==i][:, 0], bins=min(16, len(features)//4), label=str(i), alpha=0.75)
  plt.show()

  
def experiment_fewgen(cfg):
  gen_model_name = cfg['generator_model']['model_name']
  vec_model_name = cfg['vectorizer_model']['model_name']
  prompt = cfg['generator']['prompt']
  
  dataset_name = cfg['dataset']['dataset_name']
  
  logging.info(f'load {gen_model_name} language model')
  gen_tk, gen_lm = load_model(gen_model_name, device=cfg['generator_model']['device'])
  
  logging.info(f'loading {dataset_name} dataset')
  trainset, testset = prepare_dataset(**cfg['dataset'])
  
  Description.set_hparams(**cfg['description'])
  
  generator = DiverseDescriptionGenerator(model=gen_lm, tokenizer=gen_tk)
  generator.set_hparams(**cfg['generator'])
  
  if cfg['feature_mode'].startswith('by_ex'):
    logging.info(f'load {vec_model_name} language model')
    vec_tk, vec_lm = load_model(vec_model_name, device=cfg['vectorizer_model']['device'])
    logging.info(f'vectorize examples using train examples')
    train_x = vectorize_by_examples(trainset['text'], trainset['text'], generator, vec_tk, vec_lm)
    test_x = vectorize_by_examples(testset['text'], trainset['text'], generator, vec_tk, vec_lm)
    
  else : # 'by_description'
    
    if 'manual_descriptions' in cfg:
      descriptions = [Description.from_text(desc, prompt, gen_tk) for desc in cfg['manual_descriptions']]
      print(descriptions)
    else:
      logging.info(f'generate descriptions')
      descriptions = generator.generate_class_descriptions(trainset['text'], trainset['label'])
      print(descriptions)
      descriptions = sum(descriptions.values(), [])
      del gen_lm
      torch.cuda.empty_cache()
    
    if cfg['feature_mode'] == 'ppl':
      logging.info(f'load {vec_model_name} language model')
      vec_tk, vec_lm = load_model(vec_model_name, device=cfg['vectorizer_model']['device'])
      descriptions = [desc.migrate(vec_tk) for desc in descriptions]

      logging.info(f'vectorize examples using descriptions')

      train_x = vectorize_by_descriptions(trainset['text'], vec_lm, descriptions)
      test_x = vectorize_by_descriptions(testset['text'], vec_lm, descriptions)
      
    else:   # nli
      logging.info(f'load {vec_model_name} nli model')
      vec_nli = load_nli(vec_model_name, device=cfg['vectorizer_model']['device'])
      
      logging.info(f'vectorize examples using nli')
      
      train_x = vectorize_by_nli(trainset['text'], vec_nli, descriptions)
      test_x = vectorize_by_nli(testset['text'], vec_nli, descriptions)
          
  logging.info(f'train features shape: {train_x.shape}')
  logging.info(f'test features shape: {test_x.shape}')  
  
  train_y, test_y = np.array(trainset['label']), np.array(testset['label'])
  
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
    plot_features(train_x, train_y, tsne=False, title=f'{dataset_name}/{gen_model_name}/{vec_model_name}: train acc: {train_acc}')
    plot_features(test_x, test_y, tsne=False, title=f'{dataset_name}/{gen_model_name}/{vec_model_name}: test acc: {test_acc}')
    
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


def run_classifier(clf, train_x, train_y, test_x, test_y):
  clf.fit(train_x, train_y)
  preds = clf.predict(test_x)
  return accuracy_score(preds, test_y)


def sweep_classifiers(results, reduce=2, scale=True, plot=False, tsne=False):

  train_x, test_x = results['train_x'], results['test_x']
  train_y, test_y = results['train_y'], results['test_y']

  if scale:
    scaler = StandardScaler()
    train_x = scaler.fit_transform(train_x)
    test_x = scaler.transform(test_x)         
  
  if reduce > 0:
    reducer = PCA(n_components=reduce)
    
    train_x = reducer.fit_transform(train_x)
    test_x = reducer.transform(test_x) 

  if plot:
    plot_features(test_x, test_y, tsne=tsne)
  
  logreg = LogisticRegression(solver='newton-cg')
  logreg_acc = run_classifier(logreg, train_x, train_y, test_x, test_y)
  logging.info(f'logistic regression test acc: {logreg_acc}')
  accs = []

  for k in range(1, 30):
    knn = KNeighborsClassifier(n_neighbors=k)
    test_acc = run_classifier(knn, train_x, train_y, test_x, test_y)
    accs.append(test_acc)
  
  plt.title('KNN')
  plt.plot(accs)
  plt.show()
  
  accs = []

  for c in [0.25, 0.5, 1, 2, 4, 8, 16, 32, 64, 128, 256]:
    svc = SVC(kernel='rbf', C=c)
    test_acc = run_classifier(svc, train_x, train_y, test_x, test_y)
    accs.append(test_acc)

  plt.title('SVM')
  plt.plot(accs)
  plt.show()