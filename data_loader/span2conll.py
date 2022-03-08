from ast import Raise
import pdb
import numpy as np

class Span2conll():
    def __init__(self, visualize=True, max_depth=8, SEP='[SEP]'):
        self.max_depth = max_depth
        self.visualize = visualize
        self.SEP = '@'
    
    def __call__(self, tokens, labels):
        """
        inputs : 
            tokens is a list of words
            labels is a list of tuple(start, end, tag)
        output :
            a sentence in conll format
        """
        output = self.span2conll(tokens, labels)
        return output
    
    def index_in_span(self, idx, entity_list, mode='start'):
        if   mode=='start': mode=0
        elif mode=='end':   mode=1
        else:
            raise "Check mode"
        idx_entity_list = [p[mode] for p in entity_list]
        idx_entities =  np.where(np.array(idx_entity_list) == idx)[0]
        if len(idx_entities) == 0:
            return False
        return [ entity for idx, entity in enumerate(entity_list) 
                if idx in idx_entities]
    

    def span2conll(self, words, labels):
        max_token=self.max_depth
        result_conll  = []
        entity_queue  = []
        labels = sorted(labels, key=lambda x:(x[0], -x[1]))
        labels = [(e[0], e[1], f"{str(idx+1)}{self.SEP}{e[2]}") 
                          for idx, e in enumerate(labels)]
        for idx, word in enumerate(words):
            start_entities = self.index_in_span(idx, labels)
            if start_entities :
                entity_queue.extend(start_entities)
                entity_queue = sorted(entity_queue, key=lambda x:(x[0], -x[1]))
            end_entities = self.index_in_span(idx, labels, 'end')
            if end_entities:
                entity_queue = [end_en for end_en in entity_queue 
                                if end_en not in end_entities]
            temp_result = [ x[-1] for x in entity_queue]
            temp_result+=['O']*(max_token-len(temp_result))
            result_conll.append([word]+temp_result)

            if self.visualize:
                token = word[0:max_token] \
                        if len(word) <= max_token \
                        else word[0:max_token]+'...' 
                print(f"\n{idx:<3} {token:<15} \t", end=' ')
                for label in entity_queue:
                    label = label[-1]
                    label = label[0:max_token+3] \
                            if len(label) <= max_token \
                            else label[0:max_token]+'...' 
                    print(f"{label:<15}", end=' ')
        entity_queue = [e for e in entity_queue if e[1]!=idx+1]
        if len(entity_queue) != 0:
            raise "Error span2conll"
        return result_conll

    @staticmethod
    def processed_entities(label):
        temp_label = []
        for index in range(len(label)):
            _start, _end = label[index]['span']
            _tag = label[index]['entity_type']
            temp = (_start, _end, _tag)
            temp_label.append(temp)
        return temp_label
    
    @staticmethod
    def tag_bio(entities):
        num_tokens = len(entities)
        results = []
        for index in range(num_tokens):
            tag = entities[index]
            if tag=="O": 
                results.append(tag)
            elif tag!=entities[index-1]:
                tag = tag.split("@")[-1]
                results.append(f"B-{tag}")
            elif tag!="O":
                tag = tag.split("@")[-1]
                results.append(f"I-{tag}")
            else:
                raise "Error"
        return results