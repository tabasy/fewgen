
class DiverseDescriptionGenerator:
  def __init__(self, base_class_name=None, tokenizer=None, lm=None, device='cuda'):
    self.device = device

    tokenizer = tokenizer or AutoTokenizer.from_pretrained(clm_model_name)
    tokenizer.pad_token = tokenizer.bos_token
    tokenizer.mask_token = tokenizer.bos_token

    self.tokenizer = tokenizer
    self.lm = lm or GPT2LMHeadModel.from_pretrained(clm_model_name, pad_token_id=tokenizer.bos_token_id)
    self.beam_size = 16
    self.num_beam_groups = 4
    self.diversity_factor = 0.95


  def set_hparams(self, prompt='',
                  beam_size=16, min_len=2, max_len=5,
                  num_beam_groups=4, diversity_factor=0.95,
                  satisfaction=None, keep_longest=False, stop_if_satisfied=True, group_best_only=True,
                  log=True
                  ):
    self.prompt = prompt
    self.beam_size = beam_size
    self.num_beam_groups = num_beam_groups
    self.min_len = min_len
    self.max_len = max_len
    self.satisfaction = satisfaction
    self.diversity_factor = diversity_factor
    self.stop_if_satisfied = stop_if_satisfied
    self.keep_longest = keep_longest
    self.group_best_only = group_best_only
    self.log = log

  def generate(self, inputs, neg_inputs):
    descriptions = []
    Description.initialize(self.lm, self.tokenizer, self.prompt)
    roots = [Description() for i in range(self.num_beam_groups)]
    group_iters = [LevelOrderGroupIter(root) for root in roots]

    for group_levels in tqdm(zip(*group_iters), total=self.max_len, disable=not self.log):

      group_beams = []

      for glevel in group_levels:
        
        for gbeam in group_beams:  # just previous beams
          for desc in glevel:
            desc.penalize(gbeam, factor=self.diversity_factor)

        sorted_glevel = sorted(glevel, reverse=True)
        new_gbeam = sorted_glevel[:self.beam_size]
        group_beams.append(new_gbeam)

        for desc in sorted_glevel[self.beam_size:]:
          desc.forget_past()
          desc.delete()

        for i, desc in enumerate(new_gbeam):
          if self.log and len(desc) > 0:
            print(repr(desc))

          satisfied = self.satisfaction(desc)
          last_step = len(desc) == self.max_len
          is_group_best = last_step and i == 0
          keep_free = not self.group_best_only

          keep = self.group_best_only and is_group_best
          keep = keep or (satisfied and keep_free)
          keep = keep or (last_step and self.keep_longest and keep_free)
          skip = last_step or (satisfied and self.stop_if_satisfied)

          if keep:
              descriptions.append(desc)                     

          if skip:
            desc.forget_past()
            continue

          desc.generate(inputs, neg_inputs, self.beam_size)

        if self.log:
          print('-'*50)

      if self.log:
        print('='*50)

    for desc in descriptions:
      desc.absolve()   # clear penalties

    descriptions = sorted(descriptions, reverse=True)[:self.beam_size]
    return descriptions

  def generate_all(self, texts, labels):

    ids, masks = tokenize_texts(texts, self.tokenizer)
    labels = torch.tensor(labels)
    descriptions = {}

    for cls in torch.unique(labels):
      pos_inputs = ids[labels==cls], masks[labels==cls]
      all_neg_inputs = ids[labels!=cls], masks[labels!=cls]
      neg_indices = torch.randperm(len(all_neg_inputs[0]))[:len(pos_inputs[0])]
      neg_inputs = all_neg_inputs[0][neg_indices], all_neg_inputs[1][neg_indices]

      descriptions[cls.item()] = self.generate(pos_inputs, neg_inputs)
      
    return descriptions
