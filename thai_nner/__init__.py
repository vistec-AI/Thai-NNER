from typing import List, Tuple
import os
import pathlib
from collections import OrderedDict
import torch
from thai_nner.model import NNEModel
from thai_nner.data import Data
from thai_nner.utils.prediction import predict, show
import thai_nner
thai_nner_path = os.path.dirname(thai_nner.__file__)


class NNER:
    def __init__(self, path_model: str, num_classes: int=417, num_layers: int=8, max_sent_length: int=512, path_lm: str="airesearch/wangchanberta-base-att-spm-uncased", path_data: str=os.path.join(thai_nner_path,"config.json"), boundary_type: str="BIESO", device="cpu") -> None:
        self.path_model = path_model
        self.max_sent_length = max_sent_length
        self.path_lm = path_lm
        self.model = NNEModel(num_classes=num_classes,num_layers=num_layers,path_lm=path_lm)
        self.model.to(torch.device(device))
        self.state_dict_new = OrderedDict()
        try:
            self.state_dict = torch.load(self.path_model, map_location=torch.device(device))
        except NotImplementedError:
            temp = pathlib.PosixPath
            pathlib.PosixPath = pathlib.WindowsPath
            self.state_dict = torch.load(self.path_model, map_location=torch.device(device))
        self.model.load_state_dict(self.state_dict)
        self.PAD = '<pad>'
        self.data = Data(path_data=path_data, boundary_type=boundary_type)

    def get_tag(self, text: str, show_result: bool=False) -> Tuple[List[str], dict]:
        tokens, out = predict(self.model, text, self.data, self.max_sent_length, lm_path=self.path_lm)
        tokens = [tk for tk in tokens if tk!=self.PAD]
        if show_result:
            print("|".join(tokens), "\n")
            [show(x) for x in out]
        return tokens,out
