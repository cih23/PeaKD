import logging
import os
import random
import pickle
import glob
import argparse
import numpy as np
import torch
import pandas as pd
from torch.utils.data import RandomSampler, SequentialSampler
from tqdm import tqdm, trange
from envs import PROJECT_FOLDER, HOME_DATA_FOLDER

from BERT.pytorch_pretrained_bert.modeling import BertConfig
from BERT.pytorch_pretrained_bert.optimization import BertAdam, warmup_linear
from BERT.pytorch_pretrained_bert.tokenization import BertTokenizer

from src.argument_parser import default_parser, get_predefine_argv, complete_argument
from src.nli_data_processing import processors, output_modes, init_pretrain_model_PTP, init_pretrain_model_PTP_SPS, init_pretrain_model_PTP_SPS_6layer_student, get_pretrain_dataloader_PTP
from src.data_processing import init_model, get_task_dataloader
from src.modeling import BertForSequenceClassificationEncoder, FCClassifierForSequenceClassification, FullFCClassifierForSequenceClassification
from src.utils import load_model, count_parameters, eval_model_dataloader_nli, eval_model_dataloader, load_model_wonbon
from src.KD_loss import distillation_loss, patience_loss
from envs import HOME_DATA_FOLDER

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)


#########################################################################
# Prepare Parser
##########################################################################
parser = default_parser()
DEBUG = True
logger.info("IN CMD MODE")
args = parser.parse_args()
train_seed_fixed = args.train_seed
PTP_seed_fixed = args.PTP_seed
saving_criterion_acc_fixed = args.saving_criterion_acc
saving_criterion_loss_fixed = args.saving_criterion_loss
train_batch_size_fixed = args.train_batch_size
eval_batch_size_fixed = args.eval_batch_size
model_type_fixed = args.model_type
if DEBUG:
    logger.info("IN DEBUG MODE")
    argv = get_predefine_argv(args, 'glue', args.task, args.train_type, args.student_hidden_layers)
    # run simple fune-tuning *teacher* by uncommenting below cmd
    #argv = get_predefine_argv('glue', 'RTE', 'finetune_teacher')
    #argv = get_predefine_argv('glue', 'MRPC', 'finetune_teacher')
    #argv = get_predefine_argv('glue', 'SST-2', 'finetune_teacher')
    #argv = get_predefine_argv('glue', 'QNLI', 'finetune_teacher')

    # run simple fune-tuning *student* by uncommenting below cmd
    #argv = get_predefine_argv('glue', 'RTE', 'finetune_student')
    #argv = get_predefine_argv('glue', 'SST-2', 'finetune_student')
    #argv = get_predefine_argv('glue', 'MRPC', 'finetune_student')
    #argv = get_predefine_argv('glue', 'QNLI', 'finetune_student')
    
    # run vanilla KD by uncommenting below cmd
    #argv = get_predefine_argv('glue', 'RTE', 'kd')
    #argv = get_predefine_argv('glue', 'MRPC', 'kd')
    #argv = get_predefine_argv('glue', 'SST-2', 'kd')
    #argv = get_predefine_argv('glue', 'QNLI', 'kd')

    # run Patient Teacher by uncommenting below cmd
    #argv = get_predefine_argv('glue', 'RTE', 'kd.cls')
    #argv = get_predefine_argv('glue', 'MRPC', 'kd.cls')
    #argv = get_predefine_argv('glue', 'SST-2', 'kd.cls')
    #argv = get_predefine_argv('glue', 'QNLI', 'kd.cls')
    
    try:
        args = parser.parse_args(argv)
    except NameError:
        raise ValueError('please uncomment one of option above to start training')
else:
    logger.info("IN CMD MODE")
    args = parser.parse_args()
args = complete_argument(args)
if train_seed_fixed is not None:
    args.train_seed = train_seed_fixed
if PTP_seed_fixed is not None:
    args.PTP_seed = PTP_seed_fixed
if saving_criterion_acc_fixed is not None:
    args.saving_criterion_acc = saving_criterion_acc_fixed
if saving_criterion_loss_fixed is not None:
    args.saving_criterion_loss = saving_criterion_loss_fixed
if train_batch_size_fixed is not None:
    args.train_batch_size = train_batch_size_fixed
if eval_batch_size_fixed is not None:
    args.eval_batch_size = eval_batch_size_fixed
args.model_type = model_type_fixed
#########################################################################
# for restoration 
#########################################################################


#args.seed = 80301814



#########################################################################



args.raw_data_dir = os.path.join(HOME_DATA_FOLDER, 'data_raw', args.task_name)
args.feat_data_dir = os.path.join(HOME_DATA_FOLDER, 'data_feat', args.task_name)

args.train_batch_size = args.train_batch_size // args.gradient_accumulation_steps
logger.info('actual batch size on all GPU = %d' % args.train_batch_size)
device, n_gpu = args.device, args.n_gpu

random.seed(args.PTP_seed)
np.random.seed(args.PTP_seed)
torch.manual_seed(args.PTP_seed)
if args.n_gpu > 0:
    torch.cuda.manual_seed_all(args.PTP_seed)

logger.info('Input Argument Information')
args_dict = vars(args)
for a in args_dict:
    logger.info('%-28s  %s' % (a, args_dict[a]))

#########################################################################
# Prepare  Data

train_type = 'finetune'

# Specify where the teacher summary is saved in the below line.
teacher_summary = f'/home/ikhyuncho23/KDAP/data/outputs/KD/MRPC/MRPC_patient_kd_teacher_12layer_result_summary.pkl'

train_dataloader, all_label_ids = get_pretrain_dataloader_PTP(task_name = args.task, types = 'train', train_type = train_type, teacher_summary = teacher_summary)    
eval_dataloader, eval_label_ids = get_pretrain_dataloader_PTP(task_name = args.task, types = 'dev', train_type ='dontmatter', teacher_summary=teacher_summary)
#test_dataloader, test_label_ids = get_pretrain_dataloader_PTP(task_name = 'MRPC', types = 'test', train_type = 'dontmatter')

logger.info("")
logger.info('='*77)
logger.info("PTP_label.eq(0).sum() = "+str(all_label_ids.eq(0).sum()))
logger.info("PTP_label.eq(1).sum() = "+str(all_label_ids.eq(1).sum()))
logger.info("PTP_label.eq(2).sum() = "+str(all_label_ids.eq(2).sum()))
logger.info("PTP_label.eq(3).sum() = "+str(all_label_ids.eq(3).sum()))
logger.info('='*77)

num_train_optimization_steps = int(3668/ args.train_batch_size / args.gradient_accumulation_steps) * args.num_train_epochs
 
#########################################################################
# Prepare model
#########################################################################
student_config = BertConfig(os.path.join(args.bert_model, 'bert_config.json'))
output_all_layers = True

#task_name = 'RTE'
#task_name = 'MRPC'
#task_name = 'SST-2'
#task_name = 'QNLI'
# for original model uncomment below line.
if args.model_type == 'Original':
    student_encoder, student_classifier = init_pretrain_model_PTP(args.task, output_all_layers, args.student_hidden_layers, student_config)

# for SPS model uncomment below line
elif args.student_hidden_layers == 6:
    student_encoder, student_classifier = init_pretrain_model_PTP_SPS_6layer_student(args.task, output_all_layers, args.student_hidden_layers, student_config)
else:
    student_encoder, student_classifier = init_pretrain_model_PTP_SPS(args.task, output_all_layers, args.student_hidden_layers, student_config)


n_student_layer = len(student_encoder.bert.encoder.layer)
student_encoder = load_model_wonbon(student_encoder, args.encoder_checkpoint, args, 'student', verbose=True)
logger.info('*' * 77)
student_classifier = load_model(student_classifier, args.cls_checkpoint, args, 'classifier', verbose=True)


n_param_student = count_parameters(student_encoder) + count_parameters(student_classifier)
logger.info('number of layers in student model = %d' % n_student_layer)
logger.info('num parameters in student model are %d and %d' % (count_parameters(student_encoder),  count_parameters(student_classifier)))

#########################################################################
# Prepare optimizer
#########################################################################
if args.do_train:
    param_optimizer = list(student_encoder.named_parameters()) + list(student_classifier.named_parameters())
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    if args.fp16:
        logger.info('FP16 activate, use apex FusedAdam')
        try:
            from apex.optimizers import FP16_Optimizer
            from apex.optimizers import FusedAdam
        except ImportError:
            raise ImportError(
                "Please install apex from https://www.github.com/nvidia/apex to use distributed and fp16 training.")

        optimizer = FusedAdam(optimizer_grouped_parameters,
                              lr=args.learning_rate,
                              bias_correction=False,
                              max_grad_norm=1.0)
        if args.loss_scale == 0:
            optimizer = FP16_Optimizer(optimizer, dynamic_loss_scale=True)
        else:
            optimizer = FP16_Optimizer(optimizer, static_loss_scale=args.loss_scale)
    else:
        logger.info('FP16 is not activated, use BertAdam')
        optimizer = BertAdam(optimizer_grouped_parameters,
                             lr=args.learning_rate,
                             warmup=args.warmup_proportion,
                             t_total=num_train_optimization_steps)


#########################################################################
# Model Training
#########################################################################
output_model_file = '{}_nlayer.{}_lr.{}_T.{}.alpha.{}_beta.{}_bs.{}'.format(args.task_name, args.student_hidden_layers,
                                                                            args.learning_rate,
                                                                            args.T, args.alpha, args.beta,
                                                                            args.train_batch_size * args.gradient_accumulation_steps)
if args.do_train:
    global_step = 0
    nb_tr_steps = 0
    tr_loss = 0
    student_encoder.train()
    student_classifier.train()
    

    log_train = open(os.path.join(args.output_dir, 'train_log.txt'), 'w', buffering=1)
    log_eval = open(os.path.join(args.output_dir, 'eval_log.txt'), 'w', buffering=1)
    print('epoch,global_steps,step,acc,loss,kd_loss,ce_loss,AT_loss', file=log_train)
    print('epoch,acc,loss', file=log_eval)
    
    eval_loss_min = 100
    eval_best_acc = 0
    for epoch in trange(int(args.num_train_epochs), desc="Epoch"):
        tr_loss, tr_ce_loss, tr_kd_loss, tr_acc_1, tr_acc_2 = 0, 0, 0, 0, 0
        nb_tr_examples, nb_tr_steps = 0, 0
        for step, batch in enumerate(tqdm(train_dataloader, desc="Iteration")):
            student_encoder.train()
            student_classifier.train()
           
            batch = tuple(t.to(device) for t in batch)
            if train_type == 'finetune':
                train_input_ids, label_ids, train_input_mask, train_segment_ids = batch
            else :
                train_input_ids, label_ids, train_input_mask, train_segment_ids, teacher_pred, teacher_patience= batch
            full_output, pooled_output = student_encoder(train_input_ids, train_segment_ids, train_input_mask)
            logits_pred_student = student_classifier(pooled_output)
            if args.kd_model.lower() == 'kd.cls':
                student_patience = torch.stack(full_output[:-1]).transpose(0,1)
            if train_type == 'finetune':
                _,_, ce_loss = distillation_loss(logits_pred_student, label_ids, None, T=args.T, alpha=args.alpha)
            else:
                loss_dl, kd_loss, ce_loss = distillation_loss(logits_pred_student, label_ids, teacher_pred, T=args.T, alpha= args.alpha)
                print("")
                print("kd_loss: ", kd_loss)
            if args.beta > 0:
                pt_loss = args.beta * patience_loss(teacher_patience, student_patience, args.normalize_patience)
                loss = loss_dl + pt_loss
                print("")
                print("pt_loss : ", pt_loss)
            if train_type == 'finetune':
                loss = ce_loss
            elif train_type == 'kd':
                loss = loss_dl
            else:
                loss = loss_dl + pt_loss
                        
            if n_gpu > 1:
                #loss_1 = ce_loss.mean()  # mean() to average on multi-gpu.
                #loss_2 = ce_loss_2.mean()
                loss = loss.mean()
                #loss = loss_2
            if args.fp16:
                optimizer.backward(loss)
            else:
                loss.backward()

            n_sample = train_input_ids.shape[0]
            tr_loss += loss.item() * n_sample
            
            pred_cls_1 = logits_pred_student.data.max(1)[1]
            #pred_cls_2 = logits_pred_student_2.data.max(1)[1]
            tr_acc_1 += pred_cls_1.eq(label_ids).sum().cpu().item()
            #tr_acc_2 += pred_cls_2.eq(train_pred_answers).sum().cpu().item()
            nb_tr_examples += n_sample
            nb_tr_steps += 1

            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps
            
            if (step + 1) % args.gradient_accumulation_steps == 0:
                if args.fp16:
                    lr_this_step = args.learning_rate * warmup_linear(global_step / num_train_optimization_steps,
                                                                      args.warmup_proportion)
                    for param_group in optimizer.param_groups:
                        param_group['lr'] = lr_this_step
                optimizer.step()
                optimizer.zero_grad()
                global_step += 1

            if global_step % args.log_every_step == 0:
                print('{},{},{},{},{}'.format(epoch+1, global_step, step, tr_acc_1 / nb_tr_examples,
                                                       tr_loss / nb_tr_examples),
                      file=log_train)
            if (epoch == 2):
                logger.info("")
                logger.info('='*77)
                logger.info("Validation Loss : "+str(eval_loss_min)+" Validation Accuracy : "+str(eval_best_acc))
                raise ValueError('%s KD not found, please use kd or kd.full' % args.kd)
                
                        
            if (global_step % 2 == 1) & (epoch >0) :
                student_encoder.eval()
                student_classifier.eval()
                
                eval_loss, eval_loss_1, eval_loss_2, eval_acc_1, eval_acc_2 = 0, 0, 0, 0, 0
                nb_eval_examples, nb_eval_steps = 0, 0
                for step, batch in enumerate(eval_dataloader):
                    batch = tuple(t.to(device) for t in batch)
                    train_input_ids, label_ids, train_input_mask, train_segment_ids= batch
                    with torch.no_grad():
                        _, pooled_output = student_encoder(train_input_ids, train_segment_ids, train_input_mask)
                        logits_pred_student = student_classifier(pooled_output)
                        
                        _,_, ce_loss = distillation_loss(logits_pred_student, label_ids, teacher_scores= None, T=args.T, alpha=0)            
                    if n_gpu > 1:
                        loss = ce_loss.mean()
                    
                    n_sample = train_input_ids.shape[0]
                    eval_loss += loss.item() * n_sample

                    pred_cls_1 = logits_pred_student.data.max(1)[1]
                    eval_acc_1 += pred_cls_1.eq(label_ids).sum().cpu().item()
                    nb_eval_examples += n_sample
                    nb_eval_steps += 1

                    if args.gradient_accumulation_steps > 1:
                        loss = loss / args.gradient_accumulation_steps
            
                eval_loss = eval_loss/nb_eval_examples
                eval_acc_1 = eval_acc_1/nb_eval_examples
                print('{},{},{}'.format(epoch+1, eval_acc_1, eval_loss), file=log_eval)
                
                
                if eval_acc_1 > eval_best_acc:
                    logger.info("")
                    logger.info('='*77)
                    logger.info("Validation Accuracy improved! "+str(eval_best_acc)+" -> "+str(eval_acc_1))
                    logger.info('='*77)
                    eval_best_acc = eval_acc_1
                    if eval_best_acc > 1:
                        if args.n_gpu > 1:
                            torch.save(student_encoder.module.state_dict(), os.path.join(args.output_dir, output_model_file + f'_e.{epoch}.encoder_acc.pkl'))
                            torch.save(student_classifier.module.state_dict(), os.path.join(args.output_dir, output_model_file + f'_e.{epoch}.cls_acc.pkl'))
                        else:
                            torch.save(student_encoder.state_dict(), os.path.join(args.output_dir, output_model_file + f'_e.{epoch}.encoder_acc.pkl'))
                            torch.save(student_classifier.state_dict(), os.path.join(args.output_dir, output_model_file + f'_e.{epoch}.cls_acc.pkl'))
                        logger.info("Saving the model...")
                
                if eval_loss < eval_loss_min:
                    logger.info("")
                    logger.info('='*77)
                    logger.info("Validation improved! "+str(eval_loss_min)+" -> "+str(eval_loss))
                    logger.info('='*77)
                    eval_loss_min = eval_loss
                    if eval_loss < 0.63:
                        if args.n_gpu > 1:
                            torch.save(student_encoder.module.state_dict(), \
                                       os.path.join(args.output_dir, output_model_file + f'_e.{epoch}.encoder.pkl'))
                            torch.save(student_classifier.module.state_dict(), \
                                       os.path.join(args.output_dir, output_model_file + f'_e.{epoch}.cls.pkl'))
                        
                        else:
                            torch.save(student_encoder.state_dict(), os.path.join(args.output_dir, output_model_file + f'.e.{epoch}_encoder.pkl'))
                            torch.save(student_classifier.state_dict(), os.path.join(args.output_dir, output_model_file + f'.e.{epoch}_cls.pkl'))
                        logger.info("Saving the model...")        
