# Thai-NNER (Thai Nested Named Entity Recognition Corpus)
Code associated with the paper [Thai Nested Named Entity Recognition Corpus](https://github.com/vistec-AI/Thai-NNER/files/8497522/thai_nested_named_entity_recognition_corpus.pdf) at ACL 2022.

## Abstract / Motivation
This work presents the first Thai Nested Named Entity Recognition (N-NER) dataset. Thai N-NER consists of 264,798 mentions, 104 classes, and a maximum depth of 8 layers obtained from news articles and restaurant reviews, a total of 4894 documents. Our work, to the best of our knowledge, presents the largest non-English N-NER dataset and the first non-English one with fine-grained classes.

## Example
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
[Download](https://drive.google.com/drive/folders/1tkCQbksNhnGPNXez1QUc7NA0VQ5IdkMb?usp=sharing)

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
    booktitle = "Findings of the Association for Computational Linguistics: AACL 2022"
    year = "2022",
    publisher = "Association for Computational Linguistics",
}
```

## License
CC-BY-SA 3.0

## Acknowledgements
- Dataset information: The Thai N-NER corpus is supported in part by the Digital Economy Promotion Agency (depa) Digital Infrastructure Fund MP-62-003 and Siam Commercial Bank. This dataset is released as scb-nner-th-2022.
- Training code: [Tensorflow-Project-Template](https://github.com/MrGemy95/Tensorflow-Project-Template) by [Mahmoud Gemy](https://github.com/MrGemy95)
