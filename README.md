
# Holmes: Towards Effective and Harmless Model Ownership Verification to Personalized Large Vision Models via Decoupling Common Features
This is the official implementation of our paper 'Holmes: Towards Effective and Harmless Model Ownership Verification to Personalized Large Vision Models via Decoupling Common Features'. 


## Pipeline
![Pipeline](https://github.com/zlh-thu/Holmes/blob/main/figure/pipeline.pdf)


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

## Get output dataset of victim model, poisoned shadow model and benign shadow model
```
bash output_dataset.sh
```

## Train meta-classifier
```
bash train_clf.sh
```

## Ownership Verification
```
bash ownership_verification.sh
```

