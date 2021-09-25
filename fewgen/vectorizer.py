import torch

from tqdm.auto import tqdm

from fewgen.description import Description
from fewgen.inference import compute_ppl_changes, compute_batch_ppl_change, get_embedding, get_next_probs
from fewgen.util import *


def vectorize_by_descriptions(texts, model, descriptions, log=True):
  
  vectors = []
  for text in tqdm(texts, disable=not log):
    
    input_ = tokenize_texts(text, descriptions[0].tokenizer)
    ppl_changes = compute_ppl_changes(model, input_, descriptions)
    vectors.append(ppl_changes)
      
  return torch.stack(vectors, dim=0).numpy()


def vectorize_by_examples(texts, train_texts, generator, vec_tk=None, vec_lm=None, log=True):
  
  train_inputs = tokenize_texts(train_texts, generator.tokenizer)
  
  vectors = []
  for text in tqdm(texts, disable=not log):
    
    descriptions = generator.generate_example_descriptions(text)
    
    if vec_tk:
      descriptions = [desc.migrate(vec_tk) for desc in descriptions]
      
    vec_lm = vec_lm or generator.lm
    
    print(text, '\t'.join(map(str, descriptions)))
        
    vector = []
    for desc in descriptions:
      ppl_changes = compute_batch_ppl_change(vec_lm, train_inputs, desc)
      vector.append(ppl_changes)
      
    vectors.append(torch.stack(vector, dim=0).mean(dim=0))
  return torch.stack(vectors, dim=0).numpy()


def vectorize_nli_pred(pred, desc_map):
  scores = torch.zeros((len(desc_map),))
  for label, score in zip(pred['labels'], pred['scores']):
    scores[desc_map[label]] += score
  return scores


def vectorize_by_nli_(texts, model, descriptions):
  
  desc_map = {s: i for i, s in enumerate(descriptions)}
  
  nli_preds = model(texts, candidate_labels=descriptions,
                    hypothesis_template='{}', multi_label=True)
  vectors = []
  for pred in nli_preds:
    vectors.append(vectorize_nli_pred(pred, desc_map))
  vectors = torch.stack(vectors)

  return vectors


def vectorize_by_nli(texts, model, descriptions, batch_size=8):
  
  descriptions = [desc.get_text(prompt=True) for desc in descriptions]
  descriptions = list(set(descriptions))
    
  vectors = []
  for i in range(0, len(descriptions), batch_size):
    batch_descriptions = descriptions[i:i+batch_size]
    vectors.append(vectorize_by_nli_(texts, model, batch_descriptions))

  vectors = torch.cat(vectors, dim=1)

  return vectors


def vectorize_by_embedding(texts, model, tokenizer, mode='avg_emb', batch_size=16):
  
  vectors = []
  
  for i in tqdm(range(0, len(texts), batch_size)):
    batch_texts = texts[i: i+batch_size]
    batch_inputs = tokenize_texts(batch_texts, tokenizer)
  
    if mode in ['avg_emb', 'last_emb']:
      vectors.append(get_embedding(model, batch_inputs, mode=mode))
    elif mode == 'next_probs':
      vectors.append(get_next_probs(model, batch_inputs)[0])
    else:
      raise ValueError(f'invalid mode: {mode} expcted one of `last_emb`, `avg_emb`, `next_probs`')
  return torch.cat(vectors, dim=0)


def vectorize_by_sequence_classifier(texts, model, tokenizer, mode='avg_emb', batch_size=16):
  
  vectors = []
  
  for i in tqdm(range(0, len(texts), batch_size)):
    batch_texts = texts[i: i+batch_size]
    batch_inputs = tokenize_texts(batch_texts, tokenizer)
  
    if mode in ['avg_emb', 'last_emb']:
      vectors.append(get_embedding(model, batch_inputs, mode=mode))
    elif mode == 'next_probs':
      vectors.append(get_next_probs(model, batch_inputs)[0])
    else:
      raise ValueError(f'invalid mode: {mode} expcted one of `last_emb`, `avg_emb`, `next_probs`')
  return torch.cat(vectors, dim=0)