import os, re
from random import choice
import logging

import torch
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
from fewgen.vectorizer import vectorize_by_descriptions, vectorize_by_examples, vectorize_by_nli
from fewgen.description import Description
from fewgen.generator import DiverseDescriptionGenerator
from fewgen.finetune import finetune_lm
from fewgen.util import load_model, load_nli


sent_end_re = re.compile(r'[.!]')


def satisfaction(desc):
  text = desc.get_text()
  has_punc = len(sent_end_re.findall(text)) > 0
  return has_punc


default_config = {
  'generator_model': {'model_name': 'gpt2', 'device': 'cuda'},
  'vectorizer_model': {'model_name': 'gpt2', 'device': 'cuda', 'finetune': False},
  
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


def plot_features(features, labels, names=None, tsne=True, title=None):
  unique_labels = sorted(np.unique(labels).tolist())
  
  if title:
    plt.title(title)
  if tsne and features.shape[-1] > 2:
    features = TSNE(n_components=2, perplexity=len(features)/len(unique_labels)).fit_transform(features)
  
  if features.shape[-1] >= 2:
    for label, name in zip(unique_labels, names):
      plt.scatter(features[labels==label][:, 0], features[labels==label][:, 1], label=name, alpha=0.75)
  else:
    for label, name in zip(unique_labels, names):
      plt.hist(features[labels==label][:, 0], bins=min(16, len(features)//4), label=name, alpha=0.75)
  plt.legend()
  plt.show()

  
def experiment_fewgen(cfg):
  gen_model_name = cfg['generator_model']['model_name']
  vec_model_name = cfg['vectorizer_model']['model_name']
  prompt = cfg['generator']['prompt']
  
  dataset_name = cfg['dataset']['dataset_name']
  
  logging.info(f'loading {dataset_name} dataset')
  trainset, testset = prepare_dataset(**cfg['dataset'])
  label_names = get_dataset_label_names(trainset)
  
  Description.set_hparams(**cfg['description'])
    
  if 'manual_descriptions' in cfg:
    gen_model_name = 'manual'
    vec_tk = load_model(vec_model_name, tokenizer_only=True)
    
    class_descriptions = {}
    for k, man_descs in cfg['manual_descriptions'].items():
      class_descriptions[k] = []
      for man_desc in man_descs:
        if isinstance(man_desc, str):
          desc = Description.from_text(desc, prompt, vec_tk)
        else:
          prompt, desc = man_desc
          desc = Description.from_text(text=desc, prompt=prompt, tokenizer=vec_tk)
        class_descriptions[k].append(desc)
            
    descriptions = sum(class_descriptions.values(), [])
    logging.info('manual descriptions:')
    logging.info('\n'.join(map(str, descriptions)))
    
  else:
    logging.info(f'load {gen_model_name} language model')
    gen_tk, gen_lm = load_model(gen_model_name, device=cfg['generator_model']['device'])

    generator = DiverseDescriptionGenerator(model=gen_lm, tokenizer=gen_tk)
    generator.set_hparams(**cfg['generator'])

    logging.info(f'generate descriptions:')
    class_descriptions = generator.generate_class_descriptions(trainset['text'], trainset['label'])
    descriptions = sum(class_descriptions.values(), [])
    logging.info('\n'.join(map(str, descriptions)))
    
    if gen_model_name != vec_model_name:
      del gen_lm
      torch.cuda.empty_cache()

  if cfg['feature_mode'] == 'ppl':
    if vec_model_name == gen_model_name:
      logging.info(f'reuse {vec_model_name} language model')
      vec_tk, vec_lm = gen_tk, gen_lm
    else:
      logging.info(f'load {vec_model_name} language model')
      vec_tk, vec_lm = load_model(vec_model_name, device=cfg['vectorizer_model']['device'])
    descriptions = [desc.migrate(vec_tk) for desc in descriptions]
    
    if cfg['vectorizer_model']['finetune']:
      logging.info(f'finetune {vec_model_name} with our descriptions')
      class_descriptions_vec = {}
      for k, descs in class_descriptions.items():
        class_descriptions_vec[k] = [d.migrate(vec_tk) for d in descs]
      finetune_dataset = generate_finetuning_data({'trainset': trainset,
                                                   'testset': testset[:len(trainset)], 
                                                   'class_descriptions': class_descriptions_vec})
      batch_size = 2 if 'medium' in vec_model_name else 4
      finetune_lm(vec_lm, vec_tk, finetune_dataset, batch_size=batch_size)

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
    plot_features(train_x, train_y, names=label_names, tsne=False,
                  title=f'{dataset_name}/{gen_model_name}/{vec_model_name}: train acc: {train_acc}')
    plot_features(test_x, test_y, names=label_names, tsne=False, 
                  title=f'{dataset_name}/{gen_model_name}/{vec_model_name}: test acc: {test_acc}')
    
  logging.info(f'train acc: {train_acc}')
  logging.info(f'test acc: {test_acc}')
  
  return {
    'trainset': trainset,
    'testset': testset,
    'label_names': label_names,
    'class_descriptions': class_descriptions,
    'descriptions': descriptions,
    'train_x': train_x,
    'test_x': test_x,
    'train_y': train_y,
    'test_y': test_y,
    'classifier': classifier,
    'train_acc': train_acc, 
    'test_acc': test_acc, 
  }


def generate_finetuning_data(results, reverse_train=False, reverse_test=False, save_dir=None):   # use ppl change to select better descriptions for each example
  trainset, testset = results['trainset'], results['testset']
  class_descriptions = results['class_descriptions']

  train_data = {'text': [], 'prompt': [], 'description': [], 'full': []}
  test_data = {'text': [], 'prompt': [], 'description': [], 'full': []}
  
  lebel_set = set(trainset['label'])
  
  for text, label in zip(trainset['text'], trainset['label']):
    if reverse_train:
      label = choice(list(lebel_set - {label}))
    for desc in class_descriptions[label]:
      train_data['text'].append(text)
      train_data['prompt'].append(desc.get_text(prompt_only=True))
      train_data['description'].append(desc.get_text(prompt=False))
      train_data['full'].append(f'{text.strip()} {desc.get_text(prompt=True)}')
  
  for text, label in zip(testset['text'], testset['label']):
    if reverse_test:
      label = choice(list(lebel_set - {label}))
    for desc in class_descriptions[label]:
      test_data['text'].append(text)
      test_data['prompt'].append(desc.get_text(prompt_only=True))
      test_data['description'].append(desc.get_text(prompt=False))
      test_data['full'].append(f'{text.strip()} {desc.get_text(prompt=True)}')
      
  datasets = DatasetDict({
    'train': Dataset.from_dict(train_data),
    'validation': Dataset.from_dict(test_data),
    'test': Dataset.from_dict(test_data), 
    }
  )
  
  if save_dir:
    os.makedirs(save_dir, exist_ok=True)
    datasets.save_to_disk(save_dir)
      
  return datasets


def run_classifier(clf, train_x, train_y, test_x, test_y):
  clf.fit(train_x, train_y)
  train_preds = clf.predict(train_x)
  test_preds = clf.predict(test_x)
  
  return {
    'classifier': clf,
    'train_acc': accuracy_score(train_y, train_preds),
    'test_acc': accuracy_score(test_y, test_preds),
    'train_f1': f1_score(train_y, train_preds, average='macro'),
    'test_f1': f1_score(test_y, test_preds, average='macro'),
  }
          

def sweep_classifiers(results, reduce=2, scale=True, plot=False, tsne=False, conf_mat=True):

  train_x, test_x = results['train_x'], results['test_x']
  train_y, test_y = results['train_y'], results['test_y']
  label_names = results['label_names']

  if scale:
    scaler = StandardScaler()
    train_x = scaler.fit_transform(train_x)
    test_x = scaler.transform(test_x)         
  
  if reduce > 0:
    reducer = PCA(n_components=reduce)
    
    train_x = reducer.fit_transform(train_x)
    test_x = reducer.transform(test_x) 

  if plot:
    plot_features(test_x, test_y, tsne=tsne, names=label_names)
  
  logreg = LogisticRegression(solver='newton-cg')
  logreg_result = run_classifier(logreg, train_x, train_y, test_x, test_y)
  logging.info(f'logistic regression test acc: {logreg_result["test_acc"]}')
  logging.info(f'logistic regression test f1: {logreg_result["test_f1"]}')
  
  if conf_mat:
#     ConfusionMatrixDisplay.from_predictions(test_x, test_y, display_labels=label_names)  
    plot_confusion_matrix(logreg_result['classifier'], test_x, test_y, display_labels=label_names)  
    plt.show()
  
  ks, knn_results = list(range(1, 30)), {'test_acc': [], 'test_f1': []}

  for k in ks:
    knn = KNeighborsClassifier(n_neighbors=k)
    knn_result = run_classifier(knn, train_x, train_y, test_x, test_y)
    for name, values in knn_results.items():
      values.append(knn_result[name])
  
  plt.title('KNN')
  for name, values in knn_results.items():
    plt.plot(ks, values, label=name)
  plt.legend()
  plt.show()
  
  cs = [0.25, 0.5, 1, 2, 4, 8, 16, 32, 64, 128, 256]
  svm_results = {'test_acc': [], 'test_f1': []}

  for c in cs:
    svc = SVC(kernel='rbf', C=c)
    svm_result = run_classifier(svc, train_x, train_y, test_x, test_y)
    for name, values in svm_results.items():
      values.append(svm_result[name])
  
  plt.title('KNN')
  for name, values in svm_results.items():
    plt.plot(cs, values, label=name)
  plt.legend()
  plt.show()
