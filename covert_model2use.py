import pathlib
import argparse
import torch
from parse_config import ConfigParser
from model.model import NNEModel
from collections import OrderedDict

parser = argparse.ArgumentParser(description='Description of your program')
parser.add_argument('-i','--input', help='Path of model', required=True)
parser.add_argument('-o','--output', help='Path for output model', required=True)
args = vars(parser.parse_args())

try:
    checkpoint = torch.load(args['input'], map_location=torch.device('cpu'))

except NotImplementedError:
    temp = pathlib.PosixPath
    pathlib.PosixPath = pathlib.WindowsPath
    checkpoint = torch.load(args['input'], map_location=torch.device('cpu'))

state_dict_new = OrderedDict()
state_dict = checkpoint['state_dict']
for key, value in state_dict.items():
    state_dict_new[str(key).replace('module.','',1)] = value
torch.save(state_dict_new, args['output'])