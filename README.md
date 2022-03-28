# Thai-NNER (Paper Name)

## Abstract / Motivation
(In this work, we propose XXXXXXXXXXXX)


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
This project is inspired by the project [Tensorflow-Project-Template](https://github.com/MrGemy95/Tensorflow-Project-Template) by [Mahmoud Gemy](https://github.com/MrGemy95)
