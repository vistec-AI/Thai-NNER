from itertools import count
import json
import copy
from torch.utils.data import DataLoader
from data_loader.features_lm_input import InputLM
from data_loader.features_conll_labels import CONLLLabels

class NERDataloader():
    def __init__(self, 
            path_data, lm_path, max_sent_length, 
            boundary_type, batch_size, max_layers, 
            shuffle, sample_data,debug):
        self.shuffle = shuffle
        self.batch_size = batch_size
        self.max_layers = max_layers
        self.sample_data = "sample_" if sample_data else ""
        # Dataset setting
        self.path_data = path_data
        self.train = self._load("train")
        self.test = self._load("test")
        self.dev = self._load("dev")
        self.config = self._load("config")
        # Tag setting
        self.debug = debug
        self.boundary_type = boundary_type
        self.tag2ids = None
        self.ids2tag = None
        self.num_tag = None
        self.word2ids = None
        self.ids2word = None
        # Features LM input setting
        self.lm_path = lm_path
        self.max_sent_length = max_sent_length
        self.input_lm = InputLM(self.lm_path,self.max_sent_length)
        # Features flat-ner
        self.conll_labels = CONLLLabels(
            max_layers=self.max_layers, 
            boundary_type=self.boundary_type,
            debug=self.debug)
        # Add features
        self._gen_dict()
        self._features()
        
    def _collate_batch(self, batch):
        # Preprocessing batch
        keys = batch[0].keys()
        keys_nested_lm_conll = batch[0]['nested_lm_conll'].keys()
        keys_nested_lm_conll_ids = batch[0]['nested_lm_conll_ids'].keys()

        features = {key:[] for key in keys}
        features['nested_lm_conll'] = {
            key:[] for key in keys_nested_lm_conll}
        features['nested_lm_conll_ids'] = {
            key:[] for key in keys_nested_lm_conll_ids}

        for instance in batch:
            for key, val in instance.items():
                # Feature nested_lm_conll
                if key == "nested_lm_conll":
                    for layer, temp_val in val.items():
                        features['nested_lm_conll'][layer].append(temp_val)
                # Feature nested_lm_conll_ids
                elif key == "nested_lm_conll_ids":
                    for layer, temp_val in val.items():
                        features['nested_lm_conll_ids'][layer].append(temp_val)
                # Other Features
                else:
                    features[key].append(val)
        return features
    
    def get_train(self):
        return DataLoader( self.train, batch_size=self.batch_size, collate_fn=self._collate_batch)

    def get_test(self):
        return DataLoader( self.test, batch_size=self.batch_size, collate_fn=self._collate_batch)

    def get_validation(self):
        return DataLoader( self.dev, batch_size=self.batch_size, collate_fn=self._collate_batch)
    
    def _add_features(self, data):
        
        features = []
        for instance in data:
            temp_features = {}
            item = copy.deepcopy(instance)
            
            ### Using data ###
            tokens=item['tokens']
            entities=item['entities']
            
            temp_features['sentence_id']=item['sentence_id']
            temp_features['tokens']=item['tokens']
            temp_features['entities']=item['entities']
            
            ### Add LM features ###
            item = self.input_lm(tokens, entities)
            lm_tokens = item['lm_tokens']
            lm_entities = item['lm_entities']
            
            temp_features['lm_tokens']=lm_tokens
            temp_features['lm_entities']=lm_entities
            temp_features['input_ids']=item['input_ids']
            temp_features['encode_dict']=item['encode_dict']
            temp_features['attention_mask']=item['attention_mask']
            
            ### Add conll labels ###
            item = self.conll_labels(lm_tokens, lm_entities)
            nested_lm_conll = item
            flat_lm_conll = item[0]
            temp_features['nested_lm_conll']=nested_lm_conll
            temp_features['flat_lm_conll']=flat_lm_conll
            
            ### Add conll label ids ###
            flat_lm_conll_ids = self.conll_labels.map_srt_ids(
                flat_lm_conll, self.tag2ids, unk="PAD")
            
            nested_lm_conll_ids = {}
            for i, item in nested_lm_conll.items():
                temp_item_ids = self.conll_labels.map_srt_ids(
                    item, self.tag2ids,unk="PAD")
                nested_lm_conll_ids[i] = temp_item_ids
                
            temp_features['flat_lm_conll_ids']=flat_lm_conll_ids
            temp_features['nested_lm_conll_ids']=nested_lm_conll_ids
            features.append(temp_features)
        return features
    
    def _features(self):
        "Add features"
        self.train = self._add_features(self.train)
        self.test = self._add_features(self.test)
        self.dev = self._add_features(self.dev)
    
    def _load(self, data_type):
        if data_type in ["train", "test", 'dev']: # Load dataset
            path = f"{self.path_data}/{self.sample_data}{data_type}.json"
            data = json.load(open(path))
            data = [item for item in data if len(item['tokens'])>0] 
        elif data_type=="config": # Load config
            path = f"{self.path_data}/config.json"
            data = json.load(open(path))
        else:
            log = f"FlatLMDataloader, input data_type error {data_type}"
            raise log
        return data
    
    def _gen_dict(self):
        unique_labels = self.config['mentions']['unique_labels']
        boundary_type=copy.deepcopy(self.boundary_type)  # Generate boundary tags
        boundary_type=boundary_type.replace("O","")
        tags = [f"{span}-{tag}" for tag in unique_labels 
            for span in boundary_type]

        tags = ['O'] + tags
        tag2ids = {tag:ids for ids, tag in enumerate(tags)}
        ids2tag = {ids:tag for tag, ids in tag2ids.items()}

        self.tag2ids = tag2ids
        self.ids2tag = ids2tag
        self.num_tag = len(ids2tag)