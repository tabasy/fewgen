import os
import logging

import torch
import numpy as np
from transformers import AutoTokenizer, AutoModel, GPT2LMHeadModel, AutoModelForSequenceClassification, ZeroShotClassificationPipeline
from transformers.modeling_utils import PreTrainedModel
from datasets import load_from_disk
from datasets import load_dataset as origin_load_dataset
from sklearn.manifold import TSNE
from sklearn.metrics import accuracy_score, f1_score, ConfusionMatrixDisplay, plot_confusion_matrix
from matplotlib import pyplot as plt

from collections import OrderedDict
from cachetools.keys import hashkey
from cachetools import cached, LRUCache

from datasets.utils.logging import set_verbosity_error

set_verbosity_error()


def pad_lists(lists, pad_value=0):
  max_len = max(map(len, lists))
  
  padded_lists = []
  
  for list_ in lists :
    padded_lists.append(list_ + [pad_value] * (max_len-len(list_)))
    
  return padded_lists


def set_cache_dir(directory):
  previous_dir = os.environ.get('hugging_cache_dir')
  os.environ['hugging_cache_dir'] = directory
  
  if previous_dir is None:
    logging.info(f'cache directory is set to {directory}')
  else:
    logging.info(f'cache directory is changed from {previous_dir} to {directory}')


def tensor_hashkey(*args):
  keys = []
  for arg in args:
    if torch.is_tensor(arg):
      keys.append(arg.sum().item())
    elif isinstance(arg, PreTrainedModel):
      keys.append(arg.config._name_or_path)
    elif isinstance(arg, (tuple, list)):
      keys.append(tensor_hashkey(*arg))
    else:
      keys.append(hashkey(arg)[0])
  
  return tuple(keys)


def tokenize_texts(texts, tokenizer):
  tokenized = tokenizer(texts, return_tensors='pt', add_special_tokens=True, padding='longest')
  return (tokenized['input_ids'], tokenized['attention_mask'])

  
def all_to(inputs, device):
  if torch.is_tensor(inputs):
    return inputs.to(device)
  if isinstance(inputs, tuple):
    return tuple([all_to(i, device) for i in inputs])
  if isinstance(inputs, list):
    return [all_to(i, device) for i in inputs]
  if isinstance(inputs, dict):
    return {k:all_to(v, device) for k, v in inputs.items()}

  
def resolve_model_path(model_name):
  
  cache_root = os.environ.get('hugging_cache_dir')
  
  path_to_check = os.path.join(cache_root, 'model', model_name, 'pytorch_model.bin') 
  
  if os.path.exists(path_to_check):
    tokenizer_path = os.path.join(cache_root, 'tokenizer', model_name)
    model_path = os.path.join(cache_root, 'model', model_name) 
  else:
    tokenizer_path = model_name
    model_path = model_name

  return tokenizer_path, model_path


# @cached(LRUCache(maxsize=8))
def load_model(model_name, device='cuda', tokenizer_only=False):

  tokenizer_path, model_path = resolve_model_path(model_name)  

  tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
  if tokenizer_only:
    return tokenizer
  
  if 'gpt' in model_name:
    tokenizer.pad_token = tokenizer.bos_token
    tokenizer.mask_token = tokenizer.bos_token

    model = GPT2LMHeadModel.from_pretrained(model_path, pad_token_id=tokenizer.bos_token_id)
  else:
      model = AutoModel.from_pretrained(model_path)
  return tokenizer, model.to(device)


def load_dataset(dataset_name, subtask=None):
    
  cache_root = os.environ.get('hugging_cache_dir')
  
  try:
    dataset_dir = os.path.join(dataset_name, subtask) if subtask else dataset_name
    dataset = load_from_disk(os.path.join(cache_root, 'dataset', dataset_dir))
  except FileNotFoundError:
    if subtask is None:
      dataset = origin_load_dataset(*dataset_name.split('/'))
    else:
      dataset = origin_load_dataset(dataset_name, subtask)

  return dataset


def load_nli(model_name, device='cpu'):
  tokenizer, model = load_model(model_name, device)
  
  return ZeroShotClassificationPipeline(
    model=model, tokenizer=tokenizer,
    device=-1 if device == 'cpu' else 0
  )


def load_clf_model(model_name, num_labels=2, device='cuda'):
  
  tokenizer_path, model_path = resolve_model_path(model_name)

  tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
  
  tokenizer.pad_token = tokenizer.eos_token
  tokenizer.mask_token = tokenizer.bos_token

  model = AutoModelForSequenceClassification.from_pretrained(model_path, num_labels=num_labels,
                                                            pad_token_id=tokenizer.eos_token_id)
  return tokenizer, model.to(device)


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

  
def boxplot_dataframe(df, column=None, linewidth=3, layout=(1, 2), figsize=(16, 6), fontsize=14, 
            showfliers=True, showmeans=True, **kwargs):
  boxprops = dict(linestyle='-', linewidth=linewidth, color='C0')
  medianprops = dict(linestyle='-', linewidth=linewidth, color='C3')
  capprops = dict(linestyle='-', linewidth=linewidth, color='C7')
  whiskerprops = dict(linestyle='--', linewidth=linewidth, color='C7')
  meanprops = dict(linestyle='-', linewidth=linewidth, color='C2')

  df.boxplot(by='label_name', column=column, 
             layout=layout, figsize=figsize, fontsize=fontsize, showfliers=showfliers, showmeans=True,
             capprops=capprops, whiskerprops=whiskerprops, boxprops=boxprops, medianprops=medianprops, meanprops=meanprops, **kwargs)
  plt.show()
  
  
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