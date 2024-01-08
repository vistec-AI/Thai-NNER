# Thai-NNER (Thai Nested Named Entity Recognition Corpus)
This repository contains the code associated with the paper [Thai Nested Named Entity Recognition Corpus](https://aclanthology.org/2022.findings-acl.116) presented at ACL2022(findings).

<p align="center">
  <img src="/img/classes.png" alt="Classes" width="300"/>
</p>


## Abstract / Motivation
This work presents the first Thai Nested Named Entity Recognition (N-NER) dataset. Thai N-NER consists of 264,798 mentions, 104 classes, and a maximum depth of 8 layers obtained from news articles and restaurant reviews, a total of 4894 documents. Our work, to the best of our knowledge, presents the largest non-English N-NER dataset and the first non-English one with fine-grained classes.


## How to use?

### Installation

To get started, install the library:

```
pip install thai_nner
```

### Model Preparation

First, download the necessary resources (models, datasets, and pre-trained language models) from [here](https://drive.google.com/drive/folders/1Dy-360iZ9hIA-xA0yizSwmpM8sx6rrjJ?usp=share_link) and use the `convert_model2use.py` script to prepare it. 


```
python convert_model2use.py -i 0906_214036/checkpoint.pth -o model.pth
```

### Sample Usage

```python
import os
os.environ['CUDA_VISIBLE_DEVICES'] = "0"
from thai_nner import NNER

nner = NNER("model.pth")
tags = nner.get_tag("วันนี้วันที่ 5 เมษายน 2565 เป็นวันที่อากาศดีมาก")
print(tags)
```

### Examples & Testing

- [Python Library Demo](https://colab.research.google.com/drive/1SEazoGm9tZSElTxIhdyi7DwNMDO-YtJY?usp=sharing)
- [Functionality Testing](https://colab.research.google.com/drive/16m7Vx0ezLpPY2PQLlIMlbfmI9KBO5o7A?usp=sharing)


## Training and Testing

### Train

```
python train.py --device 0,1 -c config.json
```

### Test

```
python test_nne.py --resume [PATH]/checkpoint.pth
```

### Tensorboard

```
tensorboard --logdir [PATH]/save/log/
```

## Results
![Experimental results](/img/results.png)


## Citation

If you find our work useful, please consider citing:

```
@inproceedings{buaphet-etal-2022-thai,
    title = "{T}hai Nested Named Entity Recognition Corpus",
    author = "Buaphet, Weerayut  and
      Udomcharoenchaikit, Can  and
      Limkonchotiwat, Peerat  and
      Rutherford, Attapol  and
      Nutanong, Sarana",
    booktitle = "Findings of the Association for Computational Linguistics: ACL 2022",
    month = may,
    year = "2022",
    address = "Dublin, Ireland",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2022.findings-acl.116",
    doi = "10.18653/v1/2022.findings-acl.116",
    pages = "1473--1486",
    abstract = "",
}
```

## License

The project is licensed under CC-BY-SA 3.0.

## Acknowledgements

- **Dataset Credits**: The Thai N-NER corpus owes its inception partly to the Digital Economy Promotion Agency (depa) Digital Infrastructure Fund MP-62-003 and the Siam Commercial Bank. The dataset is named as scb-nner-th-2022.
- **Training Code Inspiration**: Adapted from [Tensorflow-Project-Template](https://github.com/MrGemy95/Tensorflow-Project-Template) by [Mahmoud Gemy](https://github.com/MrGemy95).
