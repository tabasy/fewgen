import numpy as np
import torch

from matplotlib import pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as clr
from pylab import rcParams

from fewgen.inference import compute_all_probs, compute_ppl


def visualize_scores(sentences, score_lists, color_maps='RdYlGn', rtl=False,
                        alpha=0.75, font_size=14, token_sep=' ', sentence_sep='<br/><br/>'):

  if type(color_maps) is str:
    color_maps = [color_maps] * len(sentences)

  span_sentences, style_sentences = [], []

  for s, tokens in enumerate(sentences):

    scores = score_lists[s]
    cmap = cm.get_cmap(color_maps[s])
    
    max_value = max(abs(min(scores)), abs(max(scores)))
#     normer = clr.Normalize(vmin=-max_value/alpha, vmax=max_value/alpha)
    normer = clr.Normalize(vmin=-1, vmax=+1)

    colors = [clr.to_hex(cmap(normer(x))) for x in scores]

    if len(tokens) != len(colors):
      raise ValueError(f"number of tokens and colors don't match: {len(tokens)}, {len(colors)}")
    
    style_elems, span_elems = [], []
    for i in range(len(tokens)):
      hash_value = colors[i][1:]
      style_elems.append(f'.c{s}-{i}-{hash_value} {{ background-color: {colors[i]}; }}')
      span_elems.append(f'<span class="vizatt c{s}-{i}-{hash_value}">{tokens[i]} </span>')

    span_sentences.append(token_sep.join(span_elems))
    style_sentences.append(' '.join(style_elems))
    text_dir = 'rtl' if rtl else 'ltr'

  return (f"""<html><head><link href="https://fonts.googleapis.com/css?family=Roboto+Mono&display=swap" rel="stylesheet">
               <style>span.vizatt {{ font-family: "Roboto Mono", monospace; font-size: {font_size}px; padding: 2px}} {' '.join(style_sentences)}</style>
               </head><body><p dir="{text_dir}">{sentence_sep.join(span_sentences)}</p></body></html>""")


def get_tokens_probs(model, tokenizer, text=None, ids=None):
  if text:
    tokens = tokenizer.tokenize(text, add_special_tokens=True)
    inputs = tokenizer([text], return_tensors='pt')
    inputs = inputs['input_ids'], inputs['attention_mask']
  else:
    tokens = tokenizer.convert_ids_to_tokens(ids)
    input_ids = torch.tensor([ids]).long()
    att_mask = torch.ones_like(input_ids)
    inputs = input_ids, att_mask
    
  probs = compute_all_probs(model, inputs)
  tokens = [tokenizer.convert_tokens_to_string([t]) for t in tokens]
  
  return tokens, torch.cat([torch.zeros(1), probs[0]])
  

def show_prob_changes(model, tokenizer, description, text, smooth_value=0.001, **kwargs):
  d_tokens, d_probs = get_tokens_probs(model, tokenizer, ids=description.prompt + description.ids)
  td_tokens, td_probs = get_tokens_probs(model, tokenizer, ids=tokenizer.encode(text) + description.prompt + description.ids)
  
  d_len = len(description)

  d_scores = ((td_probs[-d_len:] + smooth_value) / (d_probs[-d_len:] + smooth_value)).log()

  td_prob_changes = torch.zeros_like(td_probs)
  td_prob_changes[-d_len:] = d_scores
  
#   print('prob_changes:', d_scores.exp(), 'ppl change:', d_scores.mean().exp() ** -1, '\n')

  return visualize_scores([td_tokens], [td_prob_changes], **kwargs)
  
def compare_probs(model, tokenizer, description, texts, smooth_value=0.001):
  td_tokens, td_probs = [], []
  for text in texts:
    tokens, probs = get_tokens_probs(model, tokenizer, ids=tokenizer.encode(text) + description.prompt + description.ids)
    td_tokens.append(tokens)
    td_probs.append(probs)
  
  return visualize_scores(td_tokens, td_probs)


def log_results_by_example(results, train=False, path=None):
  
  label_names = results['label_names']
  descriptions = results['descriptions']
  
  if train:
    dataset = results['trainset']
    features = results['train_x']
  else:
    dataset = results['testset']
    features = results['test_x']
  
  df = dataset.to_pandas()
  preds = results['classifier'].predict(features)
  
  df['pred'] = preds
  df['correct'] = df['pred'] == df['label']
  df['label_name'] = [label_names[i] for i in df['label']]
  df['pred_name'] = [label_names[i] for i in df['pred']]
     
  for i, desc in enumerate(descriptions):
    desc_str = desc.get_text(prompt=True, prompt_only=False)
    df[desc_str] = features[:, i]

  if path:
    df.to_csv(path, sep='\t', index=False)
  
  return df

def show_description_importances(results, markdown=True, k=3):
  
  report = ''
  
  class_coefs = results['classifier'].coef_
  if class_coefs.shape[0] == 1:
    class_coefs = np.concatenate([-class_coefs, class_coefs])
    
  descriptions = results['descriptions']
  label_names = results['label_names']

  for name, coefs in zip(label_names, class_coefs):
    if markdown:
      report += f'### {name}\n\n'
    else:
      report += f'{name}\n'
      
    for idx in coefs.argsort()[-1:-(k+1):-1]:
      if markdown:
        report += f'`{descriptions[idx]}`: **{coefs[idx]:.3f}**\n\n'
      else:
        report += f'{descriptions[idx]}:\t {coefs[idx]:.3f}\n'
  return report