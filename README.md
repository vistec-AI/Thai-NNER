# Thai-NNER (Thai Nested Named Entity Recognition Corpus)
Code associated with the paper [Thai Nested Named Entity Recognition Corpus](https://github.com/vistec-AI/Thai-NNER/files/8497522/thai_nested_named_entity_recognition_corpus.pdf) at ACL 2022.

## Abstract / Motivation
This work presents the first Thai Nested Named Entity Recognition (N-NER) dataset. Thai N-NER consists of 264,798 mentions, 104 classes, and a maximum depth of 8 layers obtained from news articles and restaurant reviews, a total of 4894 documents. Our work, to the best of our knowledge, presents the largest non-English N-NER dataset and the first non-English one with fine-grained classes.

# How to use?

## Install

> pip install thai_nner

## Usage

You needs to download model from "data/[checkpoints]": 
[Download](https://drive.google.com/drive/folders/1t71ljTPO1W7xmVquyFhDVynHixlLWQ-J?usp=sharing)

Example: 0906_214036/model_best.pth

and use ```covert_model2use.py``` script by

> python -i 0906_214036/model_best.pth -o model.pth

### Usage Example

```python
from thai_nner import NNER
nner = NNER("model.pth")
nner.get_tag("วันนี้วันที่ 5 เมษายน 2565 เป็นวันที่อากาศดีมาก")
# output: (['<s>', 'วันนี้', 'วันที่', '', '', '5', '', '', 'เมษายน', '', '', '25', '65', '', '', 'เป็น', 'วันที่', '', 'อากาศ', '', 'ดีมาก', '</s>'], [{'text': ['วันนี้'], 'span': [1, 2], 'entity_type': 'rel'}, {'text': ['วันที่', '', '', '5'], 'span': [2, 6], 'entity_type': 'day'}, {'text': ['วันที่', '', '', '5', '', '', 'เมษายน', '', '', '25', '65'], 'span': [2, 13], 'entity_type': 'date'}, {'text': ['', '5'], 'span': [4, 6], 'entity_type': 'cardinal'}, {'text': ['', 'เมษายน'], 'span': [7, 9], 'entity_type': 'month'}, {'text': ['', '25', '65'], 'span': [10, 13], 'entity_type': 'year'}])
```


## Example (test and useage)
[Colabs](https://colab.research.google.com/drive/16m7Vx0ezLpPY2PQLlIMlbfmI9KBO5o7A?usp=sharing)

# Dataset and Models
## Model's Checkpoint
Download and save  models' checkpoints at the following path "data/[checkpoints]": 
[Download](https://drive.google.com/drive/folders/1t71ljTPO1W7xmVquyFhDVynHixlLWQ-J?usp=sharing)

## Dataset 
Download and save the dataset at the following path "data/[scb-nner-th-2022]": 
[Download](https://drive.google.com/drive/folders/1lp3ZK4i2Q2SC77AoVTEPy9CHB8lAGFEK?usp=sharing)

## Pre-trained Language Model
Download and save the pre-trained language model at the following path "data/[lm]": 
[Download](https://drive.google.com/drive/folders/1tkkTTMx0iFm1DA8SFsGQiXZy1TuDBTv_?usp=sharing)

# Training/Testing
## Train
```
python train.py --device 0,1 -c config.json
```
## Test
```
python test_nne.py --resume [PATH]/checkpoint.pth
```
## Tensorboard
```
tensorboard --logdir [PATH]/save/log/
```

# Results
![Experimental results](/img/results.png)


# Citation
```
@inproceedings{Buaphet-etal-2022-thai-nner,
    title = "Thai Nested Named Entity Recognition Corpus",
    author = "Buaphet, Weerayut  and
      Udomcharoenchaikit, Can  and
      Limkonchotiwat, Peerat and
      Rutherford, Attapol  and 
      Nutanong, Sarana",
    booktitle = "Findings of the Association for Computational Linguistics: ACL 2022"
    year = "2022",
    publisher = "Association for Computational Linguistics",
}
```

## License
CC-BY-SA 3.0

## Acknowledgements
- Dataset information: The Thai N-NER corpus is supported in part by the Digital Economy Promotion Agency (depa) Digital Infrastructure Fund MP-62-003 and Siam Commercial Bank. This dataset is released as scb-nner-th-2022.
- Training code: [Tensorflow-Project-Template](https://github.com/MrGemy95/Tensorflow-Project-Template) by [Mahmoud Gemy](https://github.com/MrGemy95)
