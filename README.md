# Pea-KD : Parameter-efficient and accurate Knowledge Distillation
This package provides an implementation of Pea-KD, which is to improve KD performance. This package is especially for BERT model.  

## Overview
#### Brief Explanation of the paper. 
Two main ideas proposed in the paper. Shared Parameter Sharing (SPS) and Pretraining with Teacher's Predictions (PTP). 

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
  │           └── KDAP: save teacher model prediction & trained student model.
  ├── src : The overall utils. 
  ├── envs.py: save directory paths for several usage.
  ├── run_glue_benchmark.py : save teacher prediction. Used for PTP, KD, PKD e.t.c. 
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

## How to Run

### Clone the repository

```
git clone https://monet.snu.ac.kr/gitlab/snudatalab/vet/VTT-project.git
cd PeaKD
```

### Training & Testing 
* Before starting, we need to specify a few things.
    * student_layer : The number of student layers, default = 3. Main target is students with 1~3 layers. (can change in 'KDAP/src/argument_parser.py')
    * learning baseline : KD or PKD.
    * model_mode : whether to use the original BERT model or MPS model. default = MPS model. Check lines 177~181 in 'NLI_KD_training.py'.
    * task : need to specify which dataset you want to train & test on. In this code, it is set to 'MRPC' as default. Check lines 35~54 in 'NLI_KD_training.py'
    * early stopping: during the entire training we adopt early stopping based on validation loss or accuracy. Thus, 
    we evaluate the model on the validation set every step in the training process. Thus it could take significant amount of time. 
You can change this in lines 354~401 in 'NLI_KD_training.py'. 
* First, We begin with finetuning the teacher model. do the followings:
    ```
    uncomment one of lines 36~39 in 'KDAP/NLI_KD_training.py' 
    run script
    python KDAP/NLI_KD_training.py 
    ```
    The trained model will be saved in 'data/outputs/KDAP/{task}/teacher_12layer'

* To save the teacher model's predictions, run script:
    ```
    check line 115 and 124 in 'KDAP/run_glue_benchmark.py'
    run script
    python KDAP/run_glue_benchmark.py
    ```
    The teacher predictions will be saved in 'data/outputs/KDAP/{task}/{task}_normal_kd_teacher_12layer_result_summary.pkl'
    or 'data/outputs/KDAP/{task}/{task}_patient_kd_teacher_12layer_result_summary.pkl'

* To TTR-pretrain the student model do the followings:
    ```
    modify lines 655~663 in 'KDAP/src/nli_data_processing.py'
    uncomment one of the lines 44~46 in 'KDAP/TTRP.py'
    modify lines 84~99 in 'KDAP/TTRP.py'
    run script 
    python KDAP/TTRP.py 
    ```
    The pretrained student model will be saved in 'data/outputs/KDAP/{task}/teacher_12layer'. 
* When TTR-pretraining is done, we can finally finetune the student model by doing the followingst:
    ```
    modify line 64 in 'KDAP/NLI_KD_training.py'
    run script
    python KDAP/NLI_KD_training.py 
    ```
    The finetuned student model will be saved in 'data/outputs/KDAP/{task}/teacher_12layer' 

* Example results: 

    (1) task = MRPC, student_layer = 3

    - Original BERT model = 77.69%

    - MPS model = 81.37% 
 
    - MPS + TTRP = 83.57%
    
    (2) task = RTE, student_layer = 3
    
    - Original BERT model = 61.37%
    
    - MPS model = 68.59% 
    
    - MPS + TTRP = 70.03%

## Contact

- Ikhyun Cho (ikhyuncho@snu.ac.kr)
- U Kang (ukang@snu.ac.kr)
- Data Mining Lab. at Seoul National University.

*This software may be used only for research evaluation purposes.*  
*For other purposes (e.g., commercial), please contact the authors.*
