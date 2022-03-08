import sys
import numpy as np
# sys.path.append('/ist/users/weerayutb/projects/nner_helper/scripts/')
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
            
            # BIO tagging
            boundary_tag = self.span_conll.tag_bio(label)
            
            # BIOES tagging
            if set(self.boundary_type)==set("BIOES"):
                # boundary_tag = remove_incorrect_tag(boundary_tag,"BIOES") # Will remove all tags in BIO tag
                boundary_tag = fix_labels(boundary_tag,"BIOES")
                
            # Keep each layer
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
        
if __name__ == "__main__":
    tag2ids = {"person":0, "title":1, "firstname":2, "media":3, "O":4}

    tokens = [
    'การ', 'พบปะ', 'กับ', 'บรรดา', 'สื่อมวลชน', 'ใน', 'ครั้งนี้', 
    'ถือว่า', 'ปรับ', 'กระบวนการ', 'เชิงรุก', 'ของ', 'รัฐบาล', 
    'ภายใต้', 'การนำ', 'ของ', '_', 'พล', '.', 'อ.', 'สุรยุทธ์',
    '_', 'อีกครั้ง', '_', 'หลังจากที่', 'ได้', 'เปลี่ยนแปลง', 'การทำงาน',
    'เพื่อ', 'ลด', 'แรงกดดัน', 'จาก', 'สังคม', 'มากขึ้น', '_', 'ที่มา',
    '_', '_', 'ผู้จัดการ', 'ออนไลน์']

    entities = [
    {'entity_type': 'person', 'span': [33, 40], 'text': 'พล.อ.สุรยุทธ์'},
    {'entity_type': 'title', 'span': [33, 38], 'text': 'พล.อ.'},
    {'entity_type': 'firstname', 'span': [38, 40], 'text': 'สุรยุทธ์'},
    {'entity_type': 'media', 'span': [73, 77], 'text': 'ผู้จัดการออนไลน์'},
    {'entity_type': 'media', 'span': [73, 75], 'text': 'ผู้จัดการ'}]

    max_layers=3
    conll_labels = CONLLLabels(
        max_layers=max_layers, 
        boundary_type="BIO", 
        debug=True
    )
    item = conll_labels(tokens, entities)

    print("\n\nkeys, layers\n")
    print(item.keys())
    for x in range(len(item)):
        print(f"Layers {x}:", item[x][0:5],"...")

    breakpoint()