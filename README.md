# Thai-NNER

[Colabs](https://colab.research.google.com/drive/16m7Vx0ezLpPY2PQLlIMlbfmI9KBO5o7A?usp=sharing) <br>
[Checkpoints and Data](https://drive.google.com/drive/folders/1hQ3HYI3sBJqpeabUMSVGGMdTHalCEv-5?usp=sharing)

## Train/Test
```
python train.py --device 0,1 -c config.json
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
