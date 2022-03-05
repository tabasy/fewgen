import re, copy
import os, re, json
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
from sklearn.pipeline import Pipeline
from sklearn.base import clone
from matplotlib import pyplot as plt

from fewgen.dataset import prepare_dataset, get_dataset_label_names
from fewgen.vectorizer import vectorize_by_descriptions, vectorize_by_examples, vectorize_by_nli
from fewgen.description import Description, save_descriptions, load_descriptions
from fewgen.generator import DiverseDescriptionGenerator
from fewgen.finetune import finetune_lm, generate_finetuning_data
from fewgen.util import load_model, load_nli, run_classifier, plot_features
from fewgen.explain import log_results_by_example


DESC_DIR = 'descriptions'
EXP_DIR = 'experiments'
EXP_LOG_PATH = os.path.join(EXP_DIR, 'fewgen.jsonl')

os.makedirs(DESC_DIR, exist_ok=True)
os.makedirs(EXP_DIR, exist_ok=True)


default_config = {
  'model': {
    'generator_model_name': 'gpt2',
    'vectorizer_model_name': 'gpt2', 
    'generator_device': 'cuda',
    'vectorizer_device': 'cuda',
    'finetune': False
  },
  
  'dataset': {
    'dataset_name': 'ag_news',
    'shuffle': True, 
    'shuffle_seed': 110,
    'train_ex_per_class': 16,
    'test_ex_per_class': 64,
    'test_split_name': 'test',
  },
  
  'description': {
    'manual': False,
    'num_descriptions': 4,
  },
  
  'generator': {
    'prompt': ' This is all about',
    'beam_size': 16,
    'num_beam_groups': 4,
    'max_len': 5,
    'diversity_factor': 0.9,
    'smooth_value': 1e-3,
    'stop_if_satisfied': False,
    'keep_longest': False,
    'group_best_only': True,
    'log': False
  },
  'feature_mode': 'ppl',
  'classifier': {'no_l2': Pipeline([('scaler', StandardScaler()), ('logreg', LogisticRegression(solver='saga', penalty='none'))]),
                 'l2_1.0': Pipeline([('scaler', StandardScaler()), ('logreg', LogisticRegression(solver='saga', penalty='l2', C=1.0))]),
                 'l2_0.1': Pipeline([('scaler', StandardScaler()), ('logreg', LogisticRegression(solver='saga', penalty='l2', C=0.1))]),
                 'l2_0.01': Pipeline([('scaler', StandardScaler()), ('logreg', LogisticRegression(solver='saga', penalty='l2', C=0.01))])
                },
  'plot': True,
}
  

def get_default_cfg():
  return copy.deepcopy(default_config)

  
def experiment_fewgen(cfg):
  
  model_cfg, data_cfg = cfg['model'], cfg['dataset']
  desc_cfg, gen_cfg = cfg['description'], cfg['generator']
  
  gen_model_name = model_cfg['generator_model_name']
  vec_model_name = model_cfg['vectorizer_model_name']
  prompt = gen_cfg['prompt']
  
  dataset_name = data_cfg['dataset_name'].replace('/', '_')
  os.makedirs(os.path.join(DESC_DIR, dataset_name), exist_ok=True)
  os.makedirs(os.path.join(EXP_DIR, dataset_name), exist_ok=True)

  logging.info(f'loading {dataset_name} dataset')
  trainset, testset = prepare_dataset(**cfg['dataset'])
  label_names = get_dataset_label_names(trainset)
  train_size = len(trainset['text'])
      
  if desc_cfg['manual']:
    gen_model_name = 'manual'
    vec_tk = load_model(vec_model_name, tokenizer_only=True)
    
    num_desc = desc_cfg['num_descriptions']

    desc_path = os.path.join(DESC_DIR, dataset_name, 'manual.json')
    manual_descriptions = load_descriptions(desc_path, label_names, num_desc)
    
    class_descriptions = {}
    for k, man_descs in manual_descriptions.items():
      class_descriptions[k] = []
      for man_desc in man_descs:
        desc = Description.from_text(tokenizer=vec_tk, full=man_desc)
        class_descriptions[k].append(desc)
            
    descriptions = sum(class_descriptions.values(), [])
    logging.info('manual descriptions:')
    logging.info('\n'.join(map(str, descriptions)))
    
  else:
    logging.info(f'load {gen_model_name} language model')
    gen_cfg['num_beam_groups'] = desc_cfg['num_descriptions']
    
    gen_tk, gen_lm = load_model(gen_model_name, device=model_cfg['generator_device'])

    generator = DiverseDescriptionGenerator(model=gen_lm, tokenizer=gen_tk)
    generator.set_hparams(**gen_cfg)
    
    logging.info(f'generate descriptions:')
    class_descriptions = generator.generate_class_descriptions(trainset['text'], trainset['label'])
    prompt_hash = generator.prompt.replace(' ', '_')
    desc_path = os.path.join(DESC_DIR, dataset_name, f'{gen_model_name.split("/")[-1].replace("-", "")}'\
                             f'-{prompt_hash}-{data_cfg["shuffle_seed"]}-{generator.beam_size}-{generator.smooth_value}-{generator.max_len}.json')
    save_descriptions(class_descriptions, desc_path, label_names, generator.get_hparams())
    
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
      vec_tk, vec_lm = load_model(vec_model_name, device=model_cfg['vectorizer_device'])
    descriptions = [desc.migrate(vec_tk) for desc in descriptions]
    
    if model_cfg['finetune']:
      logging.info(f'finetune {vec_model_name} with our descriptions')
      class_descriptions_vec = {}
      for k, descs in class_descriptions.items():
        class_descriptions_vec[k] = [d.migrate(vec_tk) for d in descs]
      finetune_dataset = generate_finetuning_data({'trainset': trainset,
                                                   'testset': trainset, # testset[:len(trainset)], 
                                                   'class_descriptions': class_descriptions_vec})
      
      ft_kwargs = model_cfg['finetune']
      ft_kwargs = ft_kwargs if isinstance(ft_kwargs, dict) else {}
      if 'batch_size' not in ft_kwargs:
        if 'large' in vec_model_name:
          ft_kwargs['batch_size'] = 1 
        elif 'medium' in vec_model_name:
          ft_kwargs['batch_size'] = 2 
        else:
          ft_kwargs['batch_size'] = 4 
          
      finetune_lm(vec_lm, vec_tk, finetune_dataset, **ft_kwargs)

    logging.info(f'vectorize examples using descriptions')

    train_x = vectorize_by_descriptions(trainset['text'], vec_lm, descriptions)
    test_x = vectorize_by_descriptions(testset['text'], vec_lm, descriptions)

  else:   # nli
    logging.info(f'load {vec_model_name} nli model')
    vec_nli = load_nli(vec_model_name, device=model_cfg['vectorizer_device'])

    logging.info(f'vectorize examples using nli')

    train_x = vectorize_by_nli(trainset['text'], vec_nli, descriptions)
    test_x = vectorize_by_nli(testset['text'], vec_nli, descriptions)
          
  logging.info(f'train features shape: {train_x.shape}')
  logging.info(f'test features shape: {test_x.shape}')  
  
  train_y, test_y = np.array(trainset['label']), np.array(testset['label'])
  
  classifier = cfg['classifier']
  
  train_acc, test_acc = 0, 0
  train_f1, test_f1 = 0, 0
  
  clf_results, classifiers = {}, {}
  main_clf_name = 'l2_1.0'
  
  for clf_name, clf in cfg['classifier'].items():
    clf_results[clf_name] = run_classifier(cfg['classifier'][clf_name], train_x, train_y, test_x, test_y)
    classifiers[clf_name] = clf_results[clf_name].pop('classifier')

    single_results = get_single_performance(clone(cfg['classifier'][clf_name]), train_x, train_y, test_x, test_y)
    
    experiment_log = clf_results[clf_name].copy()
    experiment_log['classifier'] = clf_name
    experiment_log['single_f1_min'] = single_results['single_f1'].min()
    experiment_log['single_f1_max'] = single_results['single_f1'].max()
    experiment_log['single_f1_mean'] = single_results['single_f1'].mean()
    experiment_log['single_acc_min'] = single_results['single_acc'].min()
    experiment_log['single_acc_max'] = single_results['single_acc'].max()
    experiment_log['single_acc_mean'] = single_results['single_acc'].mean()
    
    for sub_cfg in [model_cfg, data_cfg, desc_cfg, gen_cfg]:
      experiment_log.update(sub_cfg)

    with open(EXP_LOG_PATH, 'a') as logf:
      logf.write(json.dumps(experiment_log) + '\n')
      
  train_acc, test_acc = clf_results[main_clf_name]['train_acc'], clf_results[main_clf_name]['test_acc']
  train_f1, test_f1 = clf_results[main_clf_name]['train_f1'], clf_results[main_clf_name]['test_f1']
      
  logging.info(f'train acc: {train_acc}\t\ttest acc: {test_acc}')
  logging.info(f'train f1: {train_f1}\t\ttest f1: {test_f1}')
  
  results = {
    'trainset': trainset,
    'testset': testset,
    'label_names': label_names,
    'class_descriptions': class_descriptions,
    'descriptions': descriptions,
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

  model_name = vec_model_name.split("/")[-1].replace("-", "")
  manual = 'manual' if  desc_cfg['manual'] else 'auto'
  finetune = 'tuned' if  model_cfg['finetune'] else 'raw'
  
  train_log_path = os.path.join(EXP_DIR, dataset_name, f'{model_name}-{manual}-{finetune}-{data_cfg["shuffle_seed"]}-train.tsv')
  test_log_path = train_log_path.replace('train.tsv', 'test.tsv')
  
  log_results_by_example(results, train=True, path=train_log_path)
  log_results_by_example(results, train=False, path=test_log_path)
  
  return results


def multi_experiment(experiment_func, cfg, seeds):
  
  all_results = []
  n = len(seeds)
  
  for seed in seeds:
    cfg['dataset']['shuffle_seed'] = seed
    all_results.append(experiment_func(cfg))
    
  # list of dicts -> dict of lists
  agg_results = {k: [results[k] for results in all_results] for k in all_results[0]}
  
  # aggregate metrics using mean, std 
  for split_name in ['train', 'test']:
    for metric_name in ['acc', 'f1']:
      key = f'{split_name}_{metric_name}'
      agg_results[f'{key}_mean'] = np.mean(agg_results[key])
      agg_results[f'{key}_std'] = np.std(agg_results[key])
  
  return agg_results


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
    plot_confusion_matrix(logreg_result['classifier'], test_x, test_y, display_labels=label_names, normalize='true', xticks_rotation='vertical')  
    plt.show()
  
  ks, knn_results = list(range(1, min(32, len(train_x)-1))), {'test_acc': [], 'test_f1': []}

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
  
  plt.title('SVM')
  for name, values in svm_results.items():
    plt.plot(cs, values, label=name)
  plt.legend()
  plt.show()
            

def get_single_performance(clf=None, train_x=None, train_y=None, test_x=None, test_y=None, results=None):
    
  if results:
    train_x, test_x = results['train_x'], results['test_x']
    train_y, test_y = results['train_y'], results['test_y']

    clf = clone(results['classifier'])

  accs, f1s = [], []

  for i in range(4):
    num_cls = len(np.unique(train_y))
    num_desc = train_x.shape[1] // num_cls
    indices = np.arange(num_cls) * num_desc + i

    train_xi, test_xi = train_x[:, indices], test_x[:, indices]

    single_res = run_classifier(clf, train_xi, train_y, test_xi, test_y)
    accs.append(single_res['test_acc'])
    f1s.append(single_res['test_f1'])

  return {'single_acc': np.array(accs), 'single_f1': np.array(f1s)}