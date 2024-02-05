## Reproduction of "Language-Image COnsistency"
**Authors:** Patrik Bartak, Fanmin Shi, Konrad Szewczyk, Mikhail Vlasenko

This repository contains the codebase used in our reproduction study of "Language-Image COnsistency" 
by Lei et al. (https://arxiv.org/abs/2310.09821).

### Training

```bash
python main.py 
  --dataset $dataset
  --data $path_to_data
  --arch $arch 
  --epochs $epochs
  --lr $lr
  --training-method $training_method
  --alpha $alpha
  --beta $beta
  --seed $seed
  --workers $workers
  --context_position $context_position
  --enable_cls_prompts (optional)
```
where:
- ```$dataset``` - type of the dataset to use [allowed values: 'imagenet', 'imagenet-s50', 'cifar100']
- ```$path_to_data``` - path to the folder with data, containing 'train' and 'val' folders
- ```$arch``` - model architecture to use as the of visual encoder [allowed values: 'resnet18', 'resnet50']
- ```$epochs``` - number of training epochs (default: 40)
- ```$lr``` - initial learning rate (default: 3e-2)
- ```$training_method``` - training method to use for training [allowed value: 'baseline', 'LICO']
- ```$alpha``` - alpha hyperparameter of LICO
- ```$beta``` - bet hyperparameter of LICO
- ```$seed``` - seed used for training
- ```$workers``` - number of data loading workers
- ```$context_position``` - position of **CLASS LABEL** in the text prompt of LICO method [allowed value: 'front', 'end']

### Evaluation

#### Salience Equivariance Score

TO BE ADDED

#### Segmentation Content Heatmap

TO BE ADDED

#### Multi-Object Salience Uniformity

TO BE ADDED



