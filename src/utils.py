import logging
import torch
import os

import numpy as np
from torch.nn import CrossEntropyLoss, MSELoss
from torch import nn
from tqdm import tqdm

from src.nli_data_processing import compute_metrics


logger = logging.getLogger(__name__)


def fill_tensor(tensor, batch_size):
    """
    for DataDistributed problem in pytorch  ...
    :param tensor:
    :param batch_size:
    :return:
    """
    if len(tensor) % batch_size != 0:
        diff = batch_size - len(tensor) % batch_size
        tensor += tensor[:diff]
    return tensor


def count_parameters(model, trainable_only=True, is_dict=False):
    if is_dict:
        return sum(np.prod(list(model[k].size())) for k in model)
    if trainable_only:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    else:
        return sum(p.numel() for p in model.parameters())

def load_model(model, checkpoint, args, mode='exact', train_mode='finetune', verbose=True, DEBUG=False):
    """

    :param model:
    :param checkpoint:
    :param argstrain:
    :param mode:  this is created because for old training the encoder and classifier are mixed together
                  also adding student mode
    :param train_mode:
    :param verbose:
    :return:
    """

    n_gpu = args.n_gpu
    device = args.device
    local_rank = -1
    if checkpoint in [None, 'None']:
        if verbose:
            logger.info('no checkpoint provided for %s!' % model._get_name())
    else:
        if not os.path.exists(checkpoint):
            raise ValueError('checkpoint %s not exist' % checkpoint)
        if verbose:
            logger.info('loading %s finetuned model from %s' % (model._get_name(), checkpoint))
        model_state_dict = torch.load(checkpoint)
        old_keys = []
        new_keys = []
        pretrained_dict = dict()
        #for values in model_state_dict.values():
        #    print("fuck")
        for key, values in model_state_dict.items():
            new_key = None
            if 'gamma' in key:
                new_key = key.replace('gamma', 'weight')
            if 'beta' in key:
                new_key = key.replace('beta', 'bias')
            if key.startswith('module.'):
                new_key = key.replace('module.', '')
            if new_key:
                old_keys.append(key)
                new_keys.append(new_key)
           
        for old_key, new_key in zip(old_keys, new_keys):
            model_state_dict[new_key] = model_state_dict.pop(old_key)
        pretrained_dict = {k: v for k, v in model_state_dict.items()}
        count = 0
        
        for key, values in model_state_dict.items():
            for count in range(args.student_hidden_layers):
                if key == "bert.encoder.layer."+str(count)+".attention.self.value.weight":
                    new_key = "bert.encoder.layer."+str(count)+".attention.self.v_2.weight"
                    pretrained_dict.update({new_key: model_state_dict[key]})
            
                if key == "bert.encoder.layer."+str(count)+".attention.self.value.bias":
                    new_key = "bert.encoder.layer."+str(count)+".attention.self.v_2.bias"
                    pretrained_dict.update({new_key: model_state_dict[key]})
            
#                 if key == "bert.encoder.layer."+str(count)+".output.dense.weight":
#                     new_key = "bert.encoder.layer."+str(count)+".output_2.dense.weight"
#                     pretrained_dict.update({new_key: model_state_dict[key]})
            
#                 if key == "bert.encoder.layer."+str(count)+".output.dense.bias":
#                     new_key = "bert.encoder.layer."+str(count)+".output_2.dense.bias"
#                     pretrained_dict.update({new_key: model_state_dict[key]})
            
#                 if key == "bert.encoder.layer."+str(count)+".output.LayerNorm.weight":
#                     new_key = "bert.encoder.layer."+str(count)+".output_2.LayerNorm.weight"
#                     pretrained_dict.update({new_key: model_state_dict[key]})
            
#                 if key == "bert.encoder.layer."+str(count)+".output.LayerNorm.bias":
#                     new_key = "bert.encoder.layer."+str(count)+".output_2.LayerNorm.bias"
#                     pretrained_dict.update({new_key: model_state_dict[key]})
                
            #if key == "bert.encoder.layer."+str(count)+".attention.self.value.weight":
            #    neww_key = "bert.encoder.layer."+str(count)+".attention.self.v_2.weight"
            #    neww_values = values
            #    model_state_dict.update({'neww_key' : neww_values})
        
        del_keys = []
        keep_keys = []
        if mode == 'exact':
            pass
        elif mode == 'encoder':
            for t in list(pretrained_dict.keys()):
                if 'classifier' in t or 'cls' in t:
                    del pretrained_dict[t]
                    del_keys.append(t)
                else:
                    keep_keys.append(t)
        elif mode == 'classifier':
            for t in list(pretrained_dict.keys()):
                if 'classifier' not in t:
                    del pretrained_dict[t]
                    del_keys.append(t)
                else:
                    keep_keys.append(t)
        elif mode == 'student':
            model_keys = model.state_dict().keys()
            for t in list(pretrained_dict.keys()):
                if t not in model_keys:
                    del pretrained_dict[t]
                    del_keys.append(t)
                else:
                    keep_keys.append(t)
        else:
            raise ValueError('%s not available for now' % mode)
        #print("Checking")
        #for key, values in pretrained_dict.items():
        #    print(key)
        model.load_state_dict(pretrained_dict)
        if mode != 'exact':
            logger.info('delete %d layers, keep %d layers' % (len(del_keys), len(keep_keys)))
        if DEBUG:
            print('deleted keys =\n {}'.format('\n'.join(del_keys)))
            print('*' * 77)
            print('kept keys =\n {}'.format('\n'.join(keep_keys)))

    if args.fp16:
        logger.info('fp16 activated, now call model.half()')
        model.half()
    model.to(device)

    if train_mode != 'finetune':
        if verbose:
            logger.info('freeze BERT layer in DEBUG mode')
        model.set_mode(train_mode)

    if local_rank != -1:
        raise NotImplementedError('not implemented for local_rank != 1')
    elif n_gpu > 1:
        logger.info('data parallel because more than one gpu')
        model = torch.nn.DataParallel(model)
    return model

def load_model_2(model, checkpoint_1, checkpoint_2, args, mode='exact', train_mode='finetune', verbose=True, DEBUG=False):
    """

    :param model:
    :param checkpoint:
    :param argstrain:
    :param mode:  this is created because for old training the encoder and classifier are mixed together
                  also adding student mode
    :param train_mode:
    :param verbose:
    :return:
    """

    n_gpu = args.n_gpu
    device = args.device
    local_rank = -1
    if checkpoint_1 in [None, 'None']:
        if verbose:
            logger.info('no checkpoint provided for %s!' % model._get_name())
    else:
        if not os.path.exists(checkpoint_1):
            raise ValueError('checkpoint %s not exist' % checkpoint)
        if verbose:
            logger.info('loading %s finetuned model from ckp1: %s and  ckp2 : %s' % (model._get_name(), checkpoint_1, checkpoint_2))
        model_state_dict_1 = torch.load(checkpoint_1)
        model_state_dict_2 = torch.load(checkpoint_2)
        old_keys_1 = []
        new_keys_1 = []
        old_keys_2 = []
        new_keys_2 = []
        pretrained_dict = dict()
        for key, values in model_state_dict_1.items():
            new_key = None
            if 'gamma' in key:
                new_key = key.replace('gamma', 'weight')
            if 'beta' in key:
                new_key = key.replace('beta', 'bias')
            if key.startswith('module.'):
                new_key = key.replace('module.', '')
            if new_key:
                old_key_1s.append(key)
                new_keys_1.append(new_key)
           
        for old_key, new_key in zip(old_keys_1, new_keys_1):
            model_state_dict[new_key] = model_state_dict.pop(old_key)
        pretrained_dict = {k: v for k, v in model_state_dict_1.items()}
        
        
        key = "bert.pooler.dense.weight"
        pretrained_dict.update({key: model_state_dict_2[key]})
        key = "bert.pooler.dense.bias"
        pretrained_dict.update({key: model_state_dict_2[key]})
        #for key, values in model_state_dict_2.items():
        for count in range(3):
            key = "bert.encoder.layer."+str(count)+".attention.self.query.weight"
            new_key = "bert.encoder.layer."+str(count+3)+".attention.self.query.weight"
            pretrained_dict.update({new_key: model_state_dict_2[key]})
            
            key = "bert.encoder.layer."+str(count)+".attention.self.query.bias"
            new_key = "bert.encoder.layer."+str(count+3)+".attention.self.query.bias"
            pretrained_dict.update({new_key: model_state_dict_2[key]})
            
            key = "bert.encoder.layer."+str(count)+".attention.self.key.weight"
            new_key = "bert.encoder.layer."+str(count+3)+".attention.self.key.weight"
            pretrained_dict.update({new_key: model_state_dict_2[key]})
            
            key = "bert.encoder.layer."+str(count)+".attention.self.key.bias"
            new_key = "bert.encoder.layer."+str(count+3)+".attention.self.key.bias"
            pretrained_dict.update({new_key: model_state_dict_2[key]})
            
            key = "bert.encoder.layer."+str(count)+".attention.self.value.weight"
            new_key = "bert.encoder.layer."+str(count+3)+".attention.self.value.weight"
            pretrained_dict.update({new_key: model_state_dict_2[key]})
            
            key = "bert.encoder.layer."+str(count)+".attention.self.value.bias"
            new_key = "bert.encoder.layer."+str(count+3)+".attention.self.value.bias"
            pretrained_dict.update({new_key: model_state_dict_2[key]})
            
            print("kk")
            key = "bert.encoder.layer."+str(count)+".attention.output.dense.weight"
            new_key = "bert.encoder.layer."+str(count+3)+".attention.output.dense.weight"
            pretrained_dict.update({new_key: model_state_dict_2[key]})
            
            key = "bert.encoder.layer."+str(count)+".attention.output.dense.bias"
            new_key = "bert.encoder.layer."+str(count+3)+".attention.output.dense.bias"
            pretrained_dict.update({new_key: model_state_dict_2[key]})
            
            key = "bert.encoder.layer."+str(count)+".attention.output.LayerNorm.weight"
            new_key = "bert.encoder.layer."+str(count+3)+".attention.output.LayerNorm.weight"
            pretrained_dict.update({new_key: model_state_dict_2[key]})
            
            key = "bert.encoder.layer."+str(count)+".attention.output.LayerNorm.bias"
            new_key = "bert.encoder.layer."+str(count+3)+".attention.output.LayerNorm.bias"
            pretrained_dict.update({new_key: model_state_dict_2[key]})
            
            key = "bert.encoder.layer."+str(count)+".intermediate.dense.weight"
            new_key = "bert.encoder.layer."+str(count+3)+".intermediate.dense.weight"
            pretrained_dict.update({new_key: model_state_dict_2[key]})
            
            key = "bert.encoder.layer."+str(count)+".intermediate.dense.bias"
            new_key = "bert.encoder.layer."+str(count+3)+".intermediate.dense.bias"
            pretrained_dict.update({new_key: model_state_dict_2[key]})
            
            key = "bert.encoder.layer."+str(count)+".output.dense.weight"
            new_key = "bert.encoder.layer."+str(count+3)+".output.dense.weight"
            pretrained_dict.update({new_key: model_state_dict_2[key]})
            
            key = "bert.encoder.layer."+str(count)+".output.dense.bias"
            new_key = "bert.encoder.layer."+str(count+3)+".output.dense.bias"
            pretrained_dict.update({new_key: model_state_dict_2[key]})
            
            key = "bert.encoder.layer."+str(count)+".output.LayerNorm.weight"
            new_key = "bert.encoder.layer."+str(count+3)+".output.LayerNorm.weight"
            pretrained_dict.update({new_key: model_state_dict_2[key]})
            
            key = "bert.encoder.layer."+str(count)+".output.LayerNorm.bias"
            new_key = "bert.encoder.layer."+str(count+3)+".output.LayerNorm.bias"
            pretrained_dict.update({new_key: model_state_dict_2[key]})
            
            
            
            #if key not in list(pretrained_dict.keys()):
            #    pretrained_dict.update({key: model_state_dict_2[key]})
                #pretrained_dict.keys().append(key)
                #pretrained_dict[key] = model_state_dict_2[key]
#            pretrained_dict.update({key: model_state_dict_2[key]})
#             for count in range(3):
#                 if key == "bert.encoder.layer."+str(count)+".attention.self.value.weight":
#                     new_key = "bert.encoder.layer."+str(count)+".attention.self.v_2.weight"
#                     pretrained_dict.update({key: model_state_dict_2[key]})
            
#                 if key == "bert.encoder.layer."+str(count)+".attention.self.value.bias":
#                     new_key = "bert.encoder.layer."+str(count)+".attention.self.v_2.bias"
#                     pretrained_dict.update({new_key: model_state_dict[key]})
            
#                 if key == "bert.encoder.layer."+str(count)+".output.dense.weight":
#                     new_key = "bert.encoder.layer."+str(count)+".output_2.dense.weight"
#                     pretrained_dict.update({new_key: model_state_dict[key]})
            
#                 if key == "bert.encoder.layer."+str(count)+".output.dense.bias":
#                     new_key = "bert.encoder.layer."+str(count)+".output_2.dense.bias"
#                     pretrained_dict.update({new_key: model_state_dict[key]})
            
#                 if key == "bert.encoder.layer."+str(count)+".output.LayerNorm.weight":
#                     new_key = "bert.encoder.layer."+str(count)+".output_2.LayerNorm.weight"
#                     pretrained_dict.update({new_key: model_state_dict[key]})
            
#                 if key == "bert.encoder.layer."+str(count)+".output.LayerNorm.bias":
#                     new_key = "bert.encoder.layer."+str(count)+".output_2.LayerNorm.bias"
#                     pretrained_dict.update({new_key: model_state_dict[key]})
                
            #if key == "bert.encoder.layer."+str(count)+".attention.self.value.weight":
            #    neww_key = "bert.encoder.layer."+str(count)+".attention.self.v_2.weight"
            #    neww_values = values
            #    model_state_dict.update({'neww_key' : neww_values})
        
        del_keys = []
        keep_keys = []
        if mode == 'exact':
            pass
        elif mode == 'encoder':
            for t in list(pretrained_dict.keys()):
                if 'classifier' in t or 'cls' in t:
                    del pretrained_dict[t]
                    del_keys.append(t)
                else:
                    keep_keys.append(t)
        elif mode == 'classifier':
            for t in list(pretrained_dict.keys()):
                if 'classifier' not in t:
                    del pretrained_dict[t]
                    del_keys.append(t)
                else:
                    keep_keys.append(t)
        elif mode == 'student':
            model_keys = model.state_dict().keys()
            for t in list(pretrained_dict.keys()):
                if t not in model_keys:
                    del pretrained_dict[t]
                    del_keys.append(t)
                else:
                    keep_keys.append(t)
        else:
            raise ValueError('%s not available for now' % mode)
        model.load_state_dict(pretrained_dict)
        if mode != 'exact':
            logger.info('delete %d layers, keep %d layers' % (len(del_keys), len(keep_keys)))
        if DEBUG:
            print('deleted keys =\n {}'.format('\n'.join(del_keys)))
            print('*' * 77)
            print('kept keys =\n {}'.format('\n'.join(keep_keys)))

    if args.fp16:
        logger.info('fp16 activated, now call model.half()')
        model.half()
    model.to(device)

    if train_mode != 'finetune':
        if verbose:
            logger.info('freeze BERT layer in DEBUG mode')
        model.set_mode(train_mode)

    if local_rank != -1:
        raise NotImplementedError('not implemented for local_rank != 1')
    elif n_gpu > 1:
        logger.info('data parallel because more than one gpu')
        model = torch.nn.DataParallel(model)
    return model

def load_model_wonbon(model, checkpoint, args, mode='exact', train_mode='finetune', verbose=True, DEBUG=False):
    """

    :param model:
    :param checkpoint:
    :param argstrain:
    :param mode:  this is created because for old training the encoder and classifier are mixed together
                  also adding student mode
    :param train_mode:
    :param verbose:
    :return:
    """

    n_gpu = args.n_gpu
    device = args.device
    local_rank = -1
    if checkpoint in [None, 'None']:
        if verbose:
            logger.info('no checkpoint provided for %s!' % model._get_name())
    else:
        if not os.path.exists(checkpoint):
            raise ValueError('checkpoint %s not exist' % checkpoint)
        if verbose:
            logger.info('loading %s finetuned model from %s' % (model._get_name(), checkpoint))
        model_state_dict = torch.load(checkpoint)
        old_keys = []
        new_keys = []
        for key in model_state_dict.keys():
            new_key = None
            if 'gamma' in key:
                new_key = key.replace('gamma', 'weight')
            if 'beta' in key:
                new_key = key.replace('beta', 'bias')
            if key.startswith('module.'):
                new_key = key.replace('module.', '')
            if new_key:
                old_keys.append(key)
                new_keys.append(new_key)
        for old_key, new_key in zip(old_keys, new_keys):
            model_state_dict[new_key] = model_state_dict.pop(old_key)

        del_keys = []
        keep_keys = []
        if mode == 'exact':
            pass
        elif mode == 'encoder':
            for t in list(model_state_dict.keys()):
                if 'classifier' in t or 'cls' in t:
                    del model_state_dict[t]
                    del_keys.append(t)
                else:
                    keep_keys.append(t)
        elif mode == 'classifier':
            for t in list(model_state_dict.keys()):
                if 'classifier' not in t:
                    del model_state_dict[t]
                    del_keys.append(t)
                else:
                    keep_keys.append(t)
        elif mode == 'student':
            model_keys = model.state_dict().keys()
            for t in list(model_state_dict.keys()):
                if t not in model_keys:
                    del model_state_dict[t]
                    del_keys.append(t)
                else:
                    keep_keys.append(t)
        else:
            raise ValueError('%s not available for now' % mode)
        model.load_state_dict(model_state_dict)
        if mode != 'exact':
            logger.info('delete %d layers, keep %d layers' % (len(del_keys), len(keep_keys)))
        if DEBUG:
            print('deleted keys =\n {}'.format('\n'.join(del_keys)))
            print('*' * 77)
            print('kept keys =\n {}'.format('\n'.join(keep_keys)))

    if args.fp16:
        logger.info('fp16 activated, now call model.half()')
        model.half()
    model.to(device)

    if train_mode != 'finetune':
        if verbose:
            logger.info('freeze BERT layer in DEBUG mode')
        model.set_mode(train_mode)

    if local_rank != -1:
        raise NotImplementedError('not implemented for local_rank != 1')
    elif n_gpu > 1:
        logger.info('data parallel because more than one gpu')
        model = torch.nn.DataParallel(model)
    return model


def eval_model_dataloader(encoder_bert, classifier, dataloader, device, detailed=False,
                          criterion=nn.CrossEntropyLoss(reduction='sum'), use_pooled_output=True,
                          verbose = False):
    """
    :param encoder_bert:  either a encoder, or a encoder with classifier
    :param classifier:    if a encoder, classifier needs to be provided
    :param dataloader:
    :param device:
    :param detailed:
    :return:
    """
    if hasattr(encoder_bert, 'module'):
        encoder_bert = encoder_bert.module
    if hasattr(classifier, 'module'):
        classifier = classifier.module

    n_layer = len(encoder_bert.bert.encoder.layer)
    encoder_bert.eval()
    if classifier is not None:
        classifier.eval()

    loss = 0
    acc = 0

    # set loss function
    if detailed:
        feature_maps = [[] for _ in range(n_layer)]   # assume we only deal with bert base here
        predictions = []
        pooled_feat_maps = []

    # evaluate network
    # for idx, batch in enumerate(dataloader):
    for idx, batch in enumerate(dataloader):
        batch = tuple(t.to(device) for t in batch)
        if len(batch) > 4:
            input_ids, input_mask, segment_ids, label_ids, *ignore = batch
        else:
            input_ids, input_mask, segment_ids, label_ids = batch

        with torch.no_grad():
            if classifier is None:
                preds = encoder_bert(input_ids, segment_ids, input_mask)
            else:
                feat = encoder_bert(input_ids, segment_ids, input_mask)
                if isinstance(feat, tuple):
                    feat, pooled_feat = feat
                    if use_pooled_output:
                        preds = classifier(pooled_feat)
                    else:
                        preds = classifier(feat)
                else:
                    feat, pooled_feat = None, feat
                    preds = classifier(pooled_feat)
        loss += criterion(preds, label_ids).sum().item()

        pred_cls = preds.data.max(1)[1]
        acc += pred_cls.eq(label_ids).sum().cpu().item()

        if detailed:
            bs = input_ids.shape[0]
            need_reshape = bs != pooled_feat.shape[0]
            if classifier is None:
                raise ValueError('without classifier, feature cannot be calculated')
            if feat is None:
                pass
            else:
                for fm, f in zip(feature_maps, feat):
                    if need_reshape:
                        fm.append(f.contiguous().view(bs, -1).detach().cpu().numpy())
                    else:
                        fm.append(f.detach().cpu().numpy())
            if need_reshape:
                pooled_feat_maps.append(pooled_feat.contiguous().view(bs, -1).detach().cpu().numpy())
            else:
                pooled_feat_maps.append(pooled_feat.detach().cpu().numpy())

            predictions.append(preds.detach().cpu().numpy())
        if verbose:
            logger.info('input_ids.shape = {}, tot_loss = {}, tot_correct = {}'.format(input_ids.shape, loss, acc))

    loss /= len(dataloader.dataset) * 1.0
    acc /= len(dataloader.dataset) * 1.0
    
    if detailed:
        feat_maps = [np.concatenate(t) for t in feature_maps] if len(feature_maps[0]) > 0 else None
        if n_layer == 24:
            return {'loss': loss,
                    'acc': acc,
                    'pooled_feature_maps': np.concatenate(pooled_feat_maps),
                    'pred_logit': np.concatenate(predictions),
                    'feature_maps': [feat_maps[i] for i in [3, 7, 11, 15, 19]]}
        else:
            return {'loss': loss,
                    'acc': acc,
                    'pooled_feature_maps': np.concatenate(pooled_feat_maps),
                    'pred_logit': np.concatenate(predictions),
                    'feature_maps': feat_maps}

    return {'loss': loss, 'acc': acc}


def run_process(proc):
    os.system(proc)


def eval_model_dataloader_nli(task_name, eval_label_ids, encoder_bert, classifier, dataloader, kd_model, num_labels,
                              device, weights=None, layer_idx=None, output_mode='classification'):
    encoder_bert.eval()
    classifier.eval()

    eval_loss = 0
    nb_eval_steps = 0
    preds = []

    for input_ids, input_mask, segment_ids, label_ids in dataloader:
        input_ids = input_ids.to(device)
        input_mask = input_mask.to(device)
        segment_ids = segment_ids.to(device)
        label_ids = label_ids.to(device)

        with torch.no_grad():
            full_output, pooled_output = encoder_bert(input_ids, segment_ids, input_mask)
            if kd_model.lower() in['kd', 'kd.cls']:
                logits = classifier(pooled_output)
            elif kd_model.lower() == 'kd.full':
                logits = classifier(full_output, weights, layer_idx)
            else:
                raise NotImplementedError(f'{kd_model} not implemented yet')

        # create eval loss and other metric required by the task
        if output_mode == "classification":
            loss_fct = CrossEntropyLoss()
            tmp_eval_loss = loss_fct(logits.view(-1, num_labels), label_ids.view(-1))
        elif output_mode == "regression":
            raise NotImplementedError('regression not implemented yet')

        eval_loss += tmp_eval_loss.mean().item()
        nb_eval_steps += 1
        if len(preds) == 0:
            preds.append(logits.detach().cpu().numpy())
        else:
            preds[0] = np.append(
                preds[0], logits.detach().cpu().numpy(), axis=0)

    eval_loss = eval_loss / nb_eval_steps
    preds = preds[0]
    if output_mode == "classification":
        preds = np.argmax(preds, axis=1).flatten()
    elif output_mode == "regression":
        preds = np.squeeze(preds)
    # if args.task_name in ['CoLA']:
    #result_ = compute_metrics(task_name, preds, eval_label_ids.numpy())
    result_ = compute_metrics(task_name, preds, eval_label_ids)
    #result['eval_loss'] = eval_loss
    
    # for MRPC 
    #return {'eval_loss': eval_loss, 'f1': result_['f1'], 'acc': result_['acc']}
    
    # for CoLA
    return {'mcc' : result_['mcc'], 'eval_loss' : eval_loss}

