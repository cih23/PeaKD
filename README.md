# Pea-KD : Parameter-efficient and accurate Knowledge Distillation
This package provides an implementation of Pea-KD, which is to improve KD performance. This package is especially for BERT model.  

## Overview
#### Brief Explanation of the paper. 
Two main ideas proposed in the paper. Shuffled Parameter Sharing (SPS) and Pretraining with Teacher's Predictions (PTP). 

1) SPS 

- step1 : Paired Parameter Sharing. 
We first double the layers of the student model. Then, we share the parameters between the bottom half and the upper half. 
By this way, the model has twice the number of layers and thus can have more 'effective' model complexity while having the same number of actual parameters. 

- step2 : Shuffling. 
In addition to step1, we shuffle the Query and Key parameters between the shared pairs in order to further increase the 'effective' model complexity. 
By this shuffling process, the parameter-shared pairs can have higher model complexity and thus better representation power. 
We will call this architecture the SPS model. 

2) PTP 

- We pretrain the student model with new artificial labels (PTP labels). The labels are assigned as follows.

``` Unicode
PTP labels 
  ├── 'Confidently Correct' = teacher model's prediction is correct & confidence > t 
  ├── 'Unconfidently Correct' = teacher model's prediction is correct & confidence <= t 
  ├── 'Confidently Wrong' = teacher model's prediction is wrong & confidence > t 
  └── 'Unconfidently Wrong' = teacher model's prediction is wrong & confidence <= t
  t = hyperparameter : depends on the downstream task and the teacher model. e.g.) t = 0.95 for MRPC, t = 0.8 for RTE.
```  
#### Baseline Codes
This repository is based on the [GitHub repository](https://github.com/intersun/PKD-for-BERT-Model-Compression) for [Patient Knowledge Distillation for BERT Model Compression](https://arxiv.org/abs/1908.09355). All source files are from the repository if not mentioned otherwise. The main scripts that actually run tasks are the following two files, and they have been modified from the original files in the original repository:
- 'NLI_KD_training.py'
- 'run_glue_benchmark.py'

``` Unicode
PeaKD        
  ├── BERT
  │    └── pytorch_pretrained_bert: BERT sturcture files
  ├── data
  │    ├── data_raw
  │    │     ├── glue_data: task dataset
  │    │     └── download_glue_data.py
  │    ├── models
  │    │     └── bert_base_uncased: ckpt
  │    └── outputs
  │           └── save teacher model prediction & trained student model.
  ├── src : The overall utils. 
  ├── envs.py: save directory paths for several usage.
  ├── save_teacher_outputs.py : save teacher prediction. Used for PTP, KD, PKD e.t.c. 
  ├── PTP.py : pretrain the student model with PTP. 
  └── NLI_KD_training.py: comprehensive training file for teacher and student models.
  
```

#### Data description
- GLUE datasets

* Note that: 
    * GLUE datasets consists of CoLA, diagnostic, MNLI, MRPC, QNLI, QQP, RTE, SNLI, SST-2, STS-B, WNLI
    * You can download GLUE datasets by KDAP/data/data_raw/download_glue_data.py

#### Output
* The trained model will be saved in `data/outputs/PeaKD/{task}` after training.

## Install

#### Environment 
* Ubuntu
* CUDA 10.0
* Pytorch 1.4 
* numpy
* torch
* Tensorly
* tqdm
* pandas
* apex

# Getting Started

## Clone the repository

```
git clone https://monet.snu.ac.kr/gitlab/snudatalab/vet/VTT-project.git
cd PeaKD
```

## Run Training  
* We provide an example how to run the codes. We use task: 'MRPC', teacher layer: 3, and student layer: 3 as an example.
* Before starting, we need to specify a few things.
    * task: one of the GLUE datasets
    * train_type: one of the followings - ft, kd, pkd 
    * model_type: one of the followings - Original, SPS
    * student_hidden_layers: the number of student layers
    * train_seed: the train seed to use. If default -> random 
    * PTP_seed: the seed to use for PTP. If default -> random
    * saving_criterion_acc: if the model's val accuracy is above this value, we save the model.
    * saving_criterion_loss: if the model's val loss is below this value, we save the model.
    * load_model: specify a directory if you want to load a checkpoint.
* First, We begin with finetuning the teacher model
    ```
    run script
    python PeaKD/NLI_KD_training.py \
    --task 'MRPC' \
    --train_type 'ft' \
    --model_type 'Original' \
    --student_hidden_layers 12 \
    --train_seed None \
    --saving_criterion_acc 0.8 \
    --saving_criterion_loss 0 \
    ```
    The trained model will be saved in 'data/outputs/KD/{task}/teacher_12layer'

* To use the teacher model's predictions for example for PTP, KD, PKD do the followings:
    ```
    check lines 56 ~ 60 in 'PeaKD/save_teacher_outputs.py'
    run script:
    python PeaKD/save_teacher_outputs.py
    ```
    The teacher predictions will be saved in 'PeaKD/data/outputs/KD/{task}/{task}_normal_kd_teacher_12layer_result_summary.pkl'
    or 'PeaKD/data/outputs/KD/{task}/{task}_patient_kd_teacher_12layer_result_summary.pkl'

* To apply PTP to the student model, run script:
    ```
    python PeaKD/PTP.py \
    --task 'MRPC' \
    --train_type 'ft' \
    --model_type 'SPS' \
    --student_hidden_layer 3 \
    --PTP_seed None \
    ```
    The pretrained student model will be saved in 'PeaKD/data/outputs/KD/{task}/teacher_12layer/'. 
    you may specify the hyperparameter 't' in PeaKD/src/nli_data_processing.py line 713~.
* When PTP is done, we can finally finetune the student model by doing the followings:
    ```
    run script
    python PeaKD/NLI_KD_training.py \
    --task 'MRPC' \
    --train_type 'pkd' \
    --model_type 'SPS' \
    --student_hidden_layers 3 \
    --train_seed None \
    --saving_criterion_acc 1.0 \
    --saving_criterion_loss 0.0 \
    --load_model '~/PeaKD/data/outputs/KD/MRPC/teacher_12layer/~.pkl' 
    ```

## Contact

- Ikhyun Cho (ikhyuncho@snu.ac.kr)
- U Kang (ukang@snu.ac.kr)
- Data Mining Lab. at Seoul National University.

*This software may be used only for research evaluation purposes.*  
*For other purposes (e.g., commercial), please contact the authors.*
