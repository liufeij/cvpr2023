# A Simple and Fast Implementation of Faster R-CNN

## 1. Introduction

**[Update:]** I've further simplified the code to pytorch 1.5, torchvision 0.6, and replace the customized ops roipool and nms with the one from torchvision.  if you want the old version code, please checkout branch [v1.0](https://github.com/chenyuntc/simple-faster-rcnn-pytorch/tree/v1.0)



This project is a **Simplified** Faster R-CNN implementation based on [chainercv](https://github.com/chainer/chainercv) and other [projects](#acknowledgement) . I hope it can serve as an start code for those who want to know the detail of Faster R-CNN.  It aims to:

- Simplify the code (*Simple is better than complex*)
- Make the code more straightforward (*Flat is better than nested*)
- Match the performance reported in [origin paper](https://arxiv.org/abs/1506.01497) (*Speed Counts and mAP Matters*)

And it has the following features:
- It can be run as pure Python code, no more build affair. 
- It's a minimal implemention in around 2000 lines valid code with a lot of comment and instruction.(thanks to chainercv's excellent documentation)
- It achieves higher mAP than the origin implementation (0.712 VS 0.699)
- It achieve speed compariable with other implementation (6fps and 14fps for train and test in TITAN XP)
- It's memory-efficient (about 3GB for vgg16)


![img](imgs/faster-speed.jpg)



## 2. Performance

### 2.1 mAP

Currently, we split the dataset into train, validation and testing with raito of 7:1:2;
The mAP of the Hospital_1 in four_chamber_heart is around 0.6216;
You can do more experiments if you like;

## 3. Install dependencies


Here is an example of create environ **from scratch** with `anaconda`

```sh
# create conda env
# It is OK for you that python version > 3.7
conda create --name simp python=3.7
conda activate simp
# install pytorch
# It is also OK for you to install the up-to-date cuda and pytorch
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia

# install other dependancy
pip install visdom scikit-image tqdm fire ipdb pprint matplotlib torchnet

# start visdom
# Currently you do not need to use visdom, or you can use visdom after
# nohup python -m visdom.server &
```

If you don't use anaconda, then:

- install PyTorch with GPU (code are GPU-only), refer to [official website](http://pytorch.org)

- install other dependencies:  `pip install visdom scikit-image tqdm fire ipdb pprint matplotlib torchnet`

- start visdom for visualization

```Bash
nohup python -m visdom.server &
```



## 4. Train

### 4.1 Prepare data

#### Pascal VOC2007

1. Download the Dataset of Fetus detection by following link;
   link : 

2. Extract the 'zip' file into one directory

3. It should have this basic structure

   ```Bash
   $Dataset_Fetus_Object_Detection/                           # development kit
      $VOCdevkit/Hospital_1/                                     # VOC utility code
         $VOCdevkit/Hospital_1/annotations
            $four_chamber_heart_annotations.json
            $left_ventricular_outflow_tract.json
            $...
         $VOCdevkit/Hospital_1/four_chamber_heart
            $image1.jpg
            $image2.jpg
         $VOCdevkit/Hospital_1/left_ventricular_outflow_tract
            $...
      $VOCdevkit/Hospital_2                                      # image sets, annotations, etc.
      $...
   # ... and several other directories ...
   ```

4. modify `utils/config.py`, entering the dataset path .

### 5.3 begin training


```bash
python train.py train --env='fasterrcnn'
```

you may refer to `utils/config.py` for more argument.

Some Key arguments:

- `--dataset_path`: set the dataset path
- `--slices`: select slices that you would like to train, support one or more slice(s), should be a list, .e.g '['four_chamber_heart', 'left_ventricular_outflow_tract]'
- `--selected_hospital`: select hospital centers that you would like to train, support one or more center(s), should be a list, .e.g '['Hospital_1', 'Hospital_2]'
- `--env`: visdom env for visualization
- `--use-drop`: use dropout in RoI head, default False
- `--use-Adam`: use Adam instead of SGD, default SGD. (You need set a very low `lr` for Adam)
- `--load-path`: pretrained model path, default `None`, if it's specified, it would be loaded.

Licensed under MIT, see the LICENSE for more detail.

Contribution Welcome.

If you encounter any problem, feel free to open an issue, but too busy lately.

Correct me if anything is wrong or unclear.
