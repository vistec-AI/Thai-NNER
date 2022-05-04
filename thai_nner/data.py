import json
import os
import thai_nner
thai_nner_path = os.path.dirname(thai_nner.__file__)


class Data:
    def __init__(self, path_data=os.path.join(thai_nner_path,"config.json"), boundary_type="BIESO") -> None:
        with open(path_data) as fh:
            self.data = json.load(fh)
        unique_labels = self.data['mentions']['unique_labels']
        self.boundary_type=boundary_type.replace("O","")
        tags = [f"{span}-{tag}" for tag in unique_labels 
            for span in self.boundary_type]

        tags = ['O'] + tags
        tag2ids = {tag:ids for ids, tag in enumerate(tags)}
        ids2tag = {ids:tag for tag, ids in tag2ids.items()}

        self.tag2ids = tag2ids
        self.ids2tag = ids2tag
        self.num_tag = len(ids2tag)