import torch
import torch.nn as nn
from transformers import AutoModel
import numpy as np
from abc import abstractmethod

class BaseModel(nn.Module):
    """
    Base class for all models
    """
    @abstractmethod
    def forward(self, *inputs):
        """
        Forward pass logic
        :return: Model output
        """
        raise NotImplementedError

    def __str__(self):
        """
        Model prints with number of trainable parameters
        """
        model_parameters = filter(lambda p: p.requires_grad, self.parameters())
        params = sum([np.prod(p.size()) for p in model_parameters])
        return super().__str__() + '\nTrainable parameters: {}'.format(params)

class NNEModel(BaseModel):
    def __init__(self, num_classes=417, num_layers=8, path_lm="airesearch/wangchanberta-base-att-spm-uncased"):
        super(NNEModel, self).__init__()
        self.num_layers = num_layers
        self.num_classes = num_classes
        self.lm = AutoModel.from_pretrained(
            path_lm, output_hidden_states=True)
        self.hidden_size = self.lm.config.hidden_size
        self.decoder = nn.ModuleList([
            Decoder(self.hidden_size, self.num_classes) 
            for i in range(self.num_layers)])
 
    def forward(self, input_ids, mask):
        x = self.lm(input_ids=input_ids, attention_mask=mask)
        x = torch.stack(x[2][-4:],dim=-1).mean(-1)
        x = [self.decoder[i](x) for i in range(self.num_layers)]
        return x

class Decoder(nn.Module):
    def __init__(self, hidden_size, num_classes) -> None:
        super(Decoder, self).__init__()
        self.hidden_size = hidden_size
        self.num_classes = num_classes
        self.fc1 = nn.Linear(self.hidden_size, self.num_classes)

    def forward(self, encoder):
        x = self.fc1(encoder)
        logits = x.transpose(1,2)
        return logits