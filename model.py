import os

import torch
from transformers import LlamaForCausalLM, LlamaForSequenceClassification, AutoTokenizer,LlamaConfig, AutoModelForSequenceClassification,AutoModelForCausalLM

MODEL_DIR = '/home/wangjinzhi/model'


def get_tokenizer(model: str):
    model_path = os.path.join(MODEL_DIR, model)
    print(model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    tokenizer.pad_token_id = 0
    return tokenizer


def get_score_weight(model: str, num_labels: int = 20):
    model_path = os.path.join(MODEL_DIR, model)
    sc_model = AutoModelForSequenceClassification.from_pretrained(
        model_path,
        torch_dtype=torch.float16,
        device_map='auto',
        num_labels=num_labels,
    )
    model_stat_dict = sc_model.state_dict()
    del sc_model
    return model_stat_dict['score.weight']


def get_causalLM(model: str, seqlen: int = 128):
    model_path = os.path.join(MODEL_DIR, model)
    c_model = AutoModelForCausalLM.from_pretrained(
        model_path,
        # torch_dtype=torch.float16,
    )
    c_model.seqlen = seqlen
    return c_model


def get_sequenceclassification(score_weight, model, causalLM_model, args, num_labels: int = 20):
    model_path = os.path.join(MODEL_DIR, model)
    sc_model = AutoModelForSequenceClassification.from_pretrained(
        model_path,
        torch_dtype=torch.float16,
        device_map='cuda',
        num_labels=num_labels,
    )

    c_sd = causalLM_model.state_dict()
    c_sd.pop('lm_head.weight', None)
    c_sd['score.weight'] = score_weight.cuda()
    sc_model.load_state_dict(c_sd)
    return sc_model
