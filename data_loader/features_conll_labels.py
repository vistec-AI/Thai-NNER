import sys
import numpy as np
from data_loader.span2conll import Span2conll
from utils.correcting_labels import fix_labels, remove_incorrect_tag     

class CONLLLabels():
    def __init__(self, max_layers, boundary_type, debug):
        self.max_layers = max_layers
        self.boundary_type = boundary_type
        self.span_conll = Span2conll(
            visualize=debug, 
            max_depth=self.max_layers)
    
    def __call__(self, tokens, entities):
        labels = self.span_conll.processed_entities(entities)
        labels = self.span_conll(tokens, labels)
        labels = np.array(labels)
        targets = [list(labels[:,i+1]) for i in range(self.max_layers)]
        
        temp_target = {}
        for layer, label in enumerate(targets):
            boundary_tag = self.span_conll.tag_bio(label)
            if set(self.boundary_type)==set("BIOES"):  # BIOES tagging
                boundary_tag = fix_labels(boundary_tag,"BIOES")
            temp_target[layer]=boundary_tag
        return temp_target
    
    def map_srt_ids(self, data, dict_data, unk):
        resutls = []
        for d in data:
            if d in dict_data:
                resutls.append(dict_data.get(d))
            else:
                resutls.append(dict_data.get(unk))
        return resutls