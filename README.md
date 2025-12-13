
# Holmes: Towards Effective and Harmless Model Ownership Verification to Personalized Large Vision Models via Decoupling Common Features
This is the official implementation of our paper 'Holmes: Towards Effective and Harmless Model Ownership Verification to Personalized Large Vision Models via Decoupling Common Features'. 


## Pipeline
![Pipeline](https://github.com/zlh-thu/Holmes/blob/main/figure/pipeline.png)


## Get victim model
```
bash victim_model.sh
```

## Get loss order of the training data
```
bash get_loss_order.sh
```

## Get poisoned shadow model and benign shadow model
```
bash poisoned_model.sh
bash benign_model.sh
```

## Get output dataset
```
bash output_dataset.sh
```

## Train Meta-Classifier & Ownership Verification
Meta-classifier training and ownership verification are implemented in `ownership_verification.ipynb`. Download checkpoints to `./outputset-ckpt/` (or update the notebook path) to reproduce similar results to our paper.
