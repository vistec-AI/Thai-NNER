# Thai-NNER (Paper Name)

## Abstract / Motivation
This work presents the first Thai Nested Named Entity Recognition (N-NER) dataset. Thai N-NER consists of 264,798 mentions, 104 classes, and a maximum depth of 8 layers obtained from news articles and restaurant reviews, a total of 4894 documents. Our work, to the best of our knowledge, presents the largest non-English N-NER dataset and the first non-English one with fine-grained classes.


## Example
Example: [Colabs](https://colab.research.google.com/drive/16m7Vx0ezLpPY2PQLlIMlbfmI9KBO5o7A?usp=sharing) <br>
## Model's Checkpoint
XXX: []()
## Dataset
XXX: [Checkpoints and Data](https://drive.google.com/drive/folders/1hQ3HYI3sBJqpeabUMSVGGMdTHalCEv-5?usp=sharing)


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

## License
CC-BY-SA 3.0

## Acknowledgements
- Dataset information: XXXXXXXXXXXXXX
- Training code: [Tensorflow-Project-Template](https://github.com/MrGemy95/Tensorflow-Project-Template) by [Mahmoud Gemy](https://github.com/MrGemy95)
