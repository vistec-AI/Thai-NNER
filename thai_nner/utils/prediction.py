import torch

from thai_nner.utils.span2json import span2json
from thai_nner.utils.conll2span import conll2span
from thai_nner.utils.correcting_labels import fix_labels

from thai_nner.data_loader.features_lm_input import InputLM
from pythainlp.tokenize import word_tokenize
from thai_nner.utils.correcting_labels import fix_labels, remove_incorrect_tag

def show(x):
    text = f"{str(x['span']):<15}"
    text+= f"{x['entity_type']:<15}"
    text+= f"{''.join(x['text'])}"
    print(text)
    
def get_dict_prediction(tokens, preds, attention_mask, ids2tag):
    temp_preds=[]
    for index in range(len(preds)):    
        if attention_mask[index]==1:
            Ptag = ids2tag.get(preds[index].item())
            temp_preds.append(Ptag)
            
    temp_preds = remove_incorrect_tag(temp_preds, "BIOES")
    temp_preds = fix_labels(temp_preds, "BIOES")    
    temp_preds = conll2span(temp_preds)
    temp_preds = span2json(tokens, temp_preds)   
    return temp_preds

def predict(model, text, data_loader, max_sent_length, lm_path="airesearch/wangchanberta-base-att-spm-uncased"):
    # Setup
    ids2tag = data_loader.ids2tag
    lm_path = lm_path
    max_sent_length = max_sent_length
    
    tokens = word_tokenize(text, engine='newmm')
    out = InputLM(lm_path, max_sent_length)(tokens,[])
    lm_tokens = out['lm_tokens']
    input_ids = out['input_ids']
    mask = out['attention_mask']
    
    input_ids = torch.tensor([input_ids])
    mask = torch.tensor([mask])
    logits = model(input_ids, mask)
    
    # Pre-processing
    preds = []
    for l in range(len(logits)):
        layer = logits[l][0]
        layer = layer.argmax(axis=0)
        entity = get_dict_prediction(
            lm_tokens, layer, 
            mask[0], ids2tag)
        if len(entity)>0:
            preds.extend(entity)  
    return lm_tokens, sorted(preds, key=lambda t: t['span'])