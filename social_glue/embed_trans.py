# Chargeons un modèle générique 
import torch
import numpy as np

def embedding(sentence, model, tokenizer):
  """
  last hidden layer
  """
  # tokenisation de la phrase
  input_ids1 = torch.tensor(tokenizer.encode(sentence)).unsqueeze(0)  # Batch size 1
  # Calcul des embeddings pour chaque token
  outputs1 = model(input_ids1.to('cuda'), return_dict=True) # output_hidden_states=True)
  embed = outputs1['last_hidden_state'][0] # shape 1  x number of words x latent_space
  embed = np.mean(embed.detach().cpu().numpy(), axis=0)
  return embed

def embedding_pool(sentence, model, tokenizer):
  """
  pooler
  """
  # tokenisation de la phrase
  input_ids1 = torch.tensor(tokenizer.encode(sentence)).unsqueeze(0)  # Batch size 1
  # Calcul des embeddings pour chaque token
  outputs1 = model(input_ids1.to('cuda'), return_dict=True) # output_hidden_states=True)
  out = outputs1['pooler_output'].detach().cpu().numpy()[0]
  return out 

def embedding_all_l(sentence, model, tokenizer, device='cuda'):
  """
  return all layers
  """
  input_ids1 = torch.tensor(tokenizer.encode(sentence)).unsqueeze(0)  # Batch size 1
  # Calcul des embeddings pour chaque token
  outputs = model(input_ids1.to('cuda'), return_dict=True, output_hidden_states=True)
  out = {}
  for i, l in enumerate(outputs['hidden_states']):
      out[i] = l.detach().cpu().numpy()
      out[i] = np.mean(out[i][0], axis=0)
  return out 

def embedding_last_n(sentence, model, tokenizer, device='cuda', n=4):
  """
  last n layers
  """
  input_ids1 = torch.tensor(tokenizer.encode(sentence)).unsqueeze(0)  # Batch size 1
  # Calcul des embeddings pour chaque token
  outputs = model(input_ids1.to(device), return_dict=True, output_hidden_states=True)
  embed_dim = outputs['hidden_states'][0].shape[-1]
  out = np.zeros((n,embed_dim))
  for i,l in enumerate(range(len(outputs['hidden_states'])-1,len(outputs['hidden_states'])-1-n,-1)):

      out_l = outputs['hidden_states'][l].detach().cpu().numpy()
      out[i] = np.mean(out_l[0], axis=0)
  final_pool = np.mean(out, axis=0)
  return final_pool 




def embedding_1_l(sentence, model, tokenizer, layer_idx=10):
  """
  return one particular layer
  """
  input_ids1 = torch.tensor(tokenizer.encode(sentence)).unsqueeze(0)  # Batch size 1
  # Calcul des embeddings pour chaque token
  outputs1 = model(input_ids1, return_dict=True, output_hidden_states=True)
  return output.hidden_states[layer_ids].detach().numpy()

def embed_sentence(sentence, model):
    """
    sentence embedding
    """
    e = model.encode(sentence)
    return e #e.cpu().detach().numpy()
