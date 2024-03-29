from src.nli_data_processing import init_glue_model, init_glue_model_SPS, init_glue_model_SPS_6layer_student, get_glue_task_dataloader, get_glue_task_dataloader_pretrain5
from src.race_data_processing import init_race_model, get_race_task_dataloader


def init_model(task_name, output_all_layers, num_hidden_layers, config):
    if 'race' in task_name.lower():
        return init_race_model(task_name, output_all_layers, num_hidden_layers, config)
    else:
        return init_glue_model(task_name, output_all_layers, num_hidden_layers, config)

def init_model_SPS(task_name, output_all_layers, num_hidden_layers, config):
    if 'race' in task_name.lower():
        return init_race_model(task_name, output_all_layers, num_hidden_layers, config)
    else:
        return init_glue_model_SPS(task_name, output_all_layers, num_hidden_layers, config)

def init_model_SPS_6layer_student(task_name, output_all_layers, num_hidden_layers, config):
    if 'race' in task_name.lower():
        return init_race_model(task_name, output_all_layers, num_hidden_layers, config)
    else:
        return init_glue_model_SPS_6layer_student(task_name, output_all_layers, num_hidden_layers, config)

def get_task_dataloader(task_name, set_name, tokenizer, args, sampler, batch_size=None, knowledge=None, extra_knowledge=None):
    if 'race' in task_name.lower():
        return get_race_task_dataloader(task_name, set_name, tokenizer, args, sampler, batch_size, knowledge, extra_knowledge)
    else:
        return get_glue_task_dataloader(task_name, set_name, tokenizer, args, sampler, batch_size, knowledge, extra_knowledge)

def get_task_dataloader_pretrain(task_name, set_name, tokenizer, args, sampler, batch_size=None, knowledge=None, extra_knowledge=None, p5_label = None):
    if 'race' in task_name.lower():
        return get_race_task_dataloader(task_name, set_name, tokenizer, args, sampler, batch_size, knowledge, extra_knowledge)
    else:
        return get_glue_task_dataloader_pretrain5(task_name, set_name, tokenizer, args, sampler, batch_size, knowledge, extra_knowledge, p5_label = p5_label)
