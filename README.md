# SELFOOD: Self-Supervised Out-Of-Distribution Detection via Learning to Rank
This repository provides the code for training SELFOOD with *IDIL Loss* presented in the paper "[SELFOOD: Self-Supervised Out-Of-Distribution Detection via Learning to Rank](https://arxiv.org/abs/2305.14696)" accepted to EMNLP 2023.

## Getting Started
### Requirements
```
* ubuntu 20.04.4 LTS, cuda 11.6
* python 3.8.16
* huggingface-hub==0.12.1
* torch==1.13.1
* torchvision==0.14.1 
```
### Datasets
* Clima, NYT, TREC, Yelp

## How to Run
### Arguments (TODO)
| Args 	| Type 	| Description 	| Default|
|---------|--------|----------------------------------------------------|:-----:|
| seed 	| [int] 	| experiment seed | 42 |
| epochs 	| [int] 	| epochs | 8 |
| batch_size 	| [int] 	| batch size| 128|
| model 	| [str]	| bert-base-uncased, roberta-base | 	 bert-base-uncased |
| idil_loss_lambda 	| [float] 	| proportion of IDIL loss | 1.0	|
| ce_loss_lambda 	| [float] 	| proportion of CE loss | 0.0|
| output_dir 	| [str] 	| save files path	|  - |
| train_file 	| [str] 	| training file path	|  - |
| validation_file 	| [str] 	| validation file path	|  - |
| test_file 	| [str] 	| evaluation file path	|  - |

## Train with IDIL Loss
 
### Examples 
* Set the  `idil_loss` flag to train model with IDIL Loss.
``` 
python3 train_classifier.py --seed 21   --model  bert-base-uncased --train_file data/clima/clima_indomain/train.pkl --validation_file data/clima/clima_indomain/validation.pkl --max_length 512 --train_batch_size 8 --output_dir models/clima/clima_indomain/idil_loss/   --idil_loss --num_train_epochs 8 
```

## Train baseline models
* Remove the  `idil_loss` flag to train model with Cross-Entropy Loss.

### Examples
``` 
python3 train_classifier.py --seed 21   --model  bert-base-uncased --train_file data/clima/clima_indomain/train.pkl --validation_file data/clima/clima_indomain/validation.pkl --max_length 512 --train_batch_size 8 --output_dir models/clima/clima_indomain/ce_loss/   --with_wandb True --num_train_epochs 8 
```

### Evaluate the trained model (TODO)
``` 
``` 
#  Calculate the model predictions for model saved under `model` path against the dataset `test_file`

## Calculate preditions for in-domain portion of test set
``` 
python3 train_classifier.py --seed 13 --do_predict   --model   models/clima/clima_indomain/  --train_file data/clima/clima_indomain/train.pkl --validation_file data/clima/clima_indomain/validation.pkl --test_file data/clima/clima_indomain/test.pkl --max_length 512 --per_device_eval_batch_size 128 --output_dir results/clima/clima_indomain/  

``` 
## Calculate preditions for OOD portion of test set
``` 
python3 train_classifier.py --seed 13 --do_predict   --model   models/clima/clima_indomain/  --train_file data/clima/clima_indomain/train.pkl --validation_file data/clima/clima_indomain/validation.pkl --test_file data/yelp/yelp_ood/test.pkl --max_length 512 --per_device_eval_batch_size 128 --output_dir results/clima/yelp_ood/  
```
### Calculate metrics against evaluated data (TODO)
``` 
# Calculate performance metrics from the saved logits from path `ind_logit_path.pt` and `ood_logit_path.pt`.
python3 metrics.py --ind_logit_path   results/clima/clima_indomain/logits.pt  --ood_logit_path   results/clima/yelp_ood/logits.pt --ind_label_path results/clima/clima_indomain/labels.pt

``` 
## Results
### Performance measures
``` 
- FPR95
- Detection Error (ERR)
- AUROC
- AUPR

### Results on CIFAR-100

| Model | Indomain-Dataset | OOD-Dataset  | FPR95 | ERR | AUROC | AUPR |
|---------|--------|--------|--------|--------|--------|--------|
| SELFOOD	| Yelp	| NYT	| 63.2  | 19.6 | 79.4 | 82.7 |
| CE Loss	| Yelp	| NYT	| 82.1 | 34.9 | 63.4 | 50.7 |
| SELFOOD	| Yelp	| Clima	| 17.6 | 4.8 | 97.9 | 96.4 |
| CE Loss	| Yelp	| Clima	| 83.2 | 27.8 | 48.6 | 37.0 |
| SELFOOD	| TREC	| NYT	| 0.0 | 0.0 | 100.0 | 100.0 |
| CE Loss	| TREC	| NYT	| 8.4 | 1.3 | 97.6 | 87.5 | 

* More results can be found in the paper.

### Citation
```
@misc{mekala2023selfood,
      title={SELFOOD: Self-Supervised Out-Of-Distribution Detection via Learning to Rank}, 
      author={Dheeraj Mekala and Adithya Samavedhi and Chengyu Dong and Jingbo Shang},
      year={2023},
      eprint={2305.14696},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
```

### Contact for issues
- Dheeraj Mekala, demekala@ucsd.edu
- Adithya Samavedhui, asamavedhi@ucsd.edu
