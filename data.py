import copy
import json
import os
import random

import numpy as np
import pandas as pd
import torch
from datasets import load_dataset, Dataset, DatasetDict, concatenate_datasets
from sklearn.datasets import fetch_20newsgroups
from torch import nn
from torch.utils.data import DataLoader


def get_partition_map(dataset, num_client: int, alpha: float):
    assert alpha >= 0.0
    if alpha == 0:
        return get_iid_partition_map(
            dataset=dataset,
            num_client=num_client,
        )
    else:
        return get_noniid_partition_map(
            dataset=dataset,
            num_client=num_client,
            alpha=alpha,
        )


def get_iid_partition_map(dataset, num_client: int):
    num_samples = len(dataset['train'])
    samples_per_client = (np.repeat(num_samples, num_client) / num_client).astype(int)
    all_train_indices = np.arange(num_samples)
    np.random.shuffle(all_train_indices)
    indices_split = np.split(all_train_indices, np.cumsum(samples_per_client)[:-1])
    partition_map = {i: indices_split[i].tolist() for i in range(num_client)}
    return partition_map


def get_noniid_partition_map(dataset, num_client: int, alpha: float = 1):
    assert alpha > 0.0
    label_list = np.array(dataset['train']['label'])
    unique_labels = np.unique(label_list)
    num_label = len(unique_labels)
    num_samples = len(dataset['train'])
    label_distribution = np.random.dirichlet([alpha] * num_label, num_client)
    proportions = np.array([1 / num_client] * num_client)
    samples_per_client = (proportions * num_samples).astype(int)
    samples_per_client[-1] = num_samples - samples_per_client[:-1].sum()
    copy_samples_per_client = samples_per_client
    indices_split = [[] for _ in range(num_client)]
    label_distribution_cumsum = np.cumsum(label_distribution, axis=1)
    label_data = []
    for label in range(num_label):
        arr = np.array(label_list)
        indices = np.where(arr == label)[0]
        label_data.append(indices)
    cls_amount = [len(label_data[i]) for i in list(range(len(label_data)))]
    copy_cls_amount = copy.deepcopy(cls_amount)
    while np.sum(samples_per_client) != 0:
        curr_clnt = np.argmax(samples_per_client)
        if copy_samples_per_client[curr_clnt] <= 0:
            continue
        copy_samples_per_client[curr_clnt] -= 1
        curr_prior = label_distribution_cumsum[curr_clnt]
        while True:
            cls_label = np.argmax(np.random.uniform() <= curr_prior)
            if cls_amount[cls_label] <= 0:
                if alpha < 1:
                    cls_amount[cls_label] = copy_cls_amount[cls_label]
                continue
            cls_amount[cls_label] -= 1
            indices_split[curr_clnt].append(label_data[cls_label][cls_amount[cls_label]])
            break
    partition_map = {i: list(map(int, indices_split[i])) for i in range(num_client)}
    return partition_map


def prepare_calibration_dataset(
        model,
        tokenizer,
        calib_data_name: str = 'wikitext2',
        train_dataset=None,
        num_samples: int = 128,
        seqlen: int = 128
):
    print(calib_data_name)
    if calib_data_name == "wikitext2":
        train_loader = prepare_calib_wikitext2(
            tokenizer=tokenizer,
            num_samples=num_samples,
            seqlen=seqlen,
        )
    elif calib_data_name == 'c4':
        train_loader = prepare_calib_c4(
            tokenizer=tokenizer,
            num_samples=num_samples,
            seqlen=seqlen,
        )
    elif calib_data_name == '20newsgroup':
        if train_dataset is not None:
            train_loader = prepare_calib_20newsgroup(
                train_dataset=train_dataset, num_samples=num_samples,
                seqlen=seqlen,
            )
        else:
            all_dataset = get_20_newsgroups_tokenizer_dataset(
                tokenizer=tokenizer
            )
            train_dataset = all_dataset['train']
            train_loader = prepare_calib_20newsgroup(
                train_dataset=train_dataset,
                num_samples=num_samples,
                seqlen=seqlen,
            )
    elif calib_data_name == 'patient2trial':
        if train_dataset is not None:
            train_loader = prepare_calib_patient2trial_matching(
                train_dataset=train_dataset, num_samples=num_samples,
                seqlen=seqlen,
            )
        else:
            all_dataset = get_patient2trial_matching(
                tokenizer=tokenizer
            )
            train_dataset = all_dataset['train']
            train_loader = prepare_calib_patient2trial_matching(
                train_dataset=train_dataset,
                num_samples=num_samples,
                seqlen=seqlen,
            )
    elif calib_data_name == 'random':
        train_loader = prepare_calib_random(
            num_samples=num_samples,
            seqlen=seqlen,
        )
    else:
        raise ValueError("undefine calibration dataset")

    use_cache = model.config.use_cache
    model.cuda()
    model.config.use_cache = False
    layers = model.model.layers
    dtype = next(iter(model.parameters())).dtype
    inps = torch.zeros((num_samples, seqlen, model.config.hidden_size), dtype=dtype).cuda()
    inps.requires_grad = False
    cache = {'i': 0, 'attention_mask': None, "position_ids": None}

    class Catcher(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module

        def forward(self, inp, **kwargs):
            inps[cache['i']] = inp
            cache['i'] += 1
            cache['attention_mask'] = kwargs['attention_mask']
            cache['position_ids'] = kwargs['position_ids']
            raise ValueError

    layers[0] = Catcher(layers[0])
    for batch in train_loader:
        try:
            model(batch[0].cuda())
            # model(batch.cuda())
        except ValueError:
            pass
    layers[0] = layers[0].module
    outs = torch.zeros_like(inps)
    attention_mask = cache['attention_mask']
    position_ids = cache['position_ids']
    model.config.use_cache = use_cache
    model.cpu()
    return inps, outs, attention_mask, position_ids


def prepare_calib_wikitext2(tokenizer, num_samples: int, seqlen: int):
    wiki_data = load_dataset('/home/wangjinzhi/aa/FLAP/wikitext/wikitext-2-raw-v1', split='train')
    tokenized_wiki = tokenizer(" ".join(wiki_data['text']), return_tensors='pt')
    train_loader = []
    for _ in range(num_samples):
        idx = random.randint(0, tokenized_wiki.input_ids.shape[1] - seqlen - 1)
        inp = tokenized_wiki.input_ids[:, idx:idx + seqlen]
        tar = inp.clone()
        tar[:, :-1] = -100
        train_loader.append((inp, tar))
    return train_loader


def prepare_calib_c4(tokenizer, num_samples: int, seqlen: int):
    traindata = load_dataset('/home/wangjinzhi/aa/FLAP/c4', split='train')
    train_loader = []
    for _ in range(num_samples):
        while True:
            i = random.randint(0, len(traindata) - 1)
            trainenc = tokenizer(traindata[i]['text'], return_tensors='pt')
            if trainenc.input_ids.shape[1] > seqlen:
                break
        i = random.randint(0, trainenc.input_ids.shape[1] - seqlen - 1)
        j = i + seqlen
        inp = trainenc.input_ids[:, i:j]
        tar = inp.clone()
        tar[:, :-1] = -100
        train_loader.append((inp, tar))
    return train_loader


def prepare_calib_random(num_samples: int, seqlen: int):
    train_loader = []
    for _ in range(num_samples):
        inp = torch.randint(0, 128256, (1, seqlen), dtype=torch.long)
        tar = inp.clone()
        tar[:, :-1] = -100
        train_loader.append((inp, tar))
    return train_loader


def prepare_calib_20newsgroup(train_dataset, num_samples: int, seqlen: int):
    train_loader = []
    shuffled_dataset = train_dataset.shuffle(seed=42)
    sampled_dataset = shuffled_dataset.select(range(num_samples))
    for data in sampled_dataset:
        idx = random.randint(0, data['input_ids'].shape[0] - seqlen - 1)
        inp = data['input_ids'][idx:idx + seqlen].unsqueeze(0)
        tar = inp.clone()
        tar[:, :-1] = -100
        train_loader.append((inp, tar))
    return train_loader


def prepare_calib_patient2trial_matching(train_dataset, num_samples: int, seqlen: int):
    train_loader = []
    shuffled_dataset = train_dataset.shuffle(seed=42)
    sampled_dataset = shuffled_dataset.select(range(num_samples))
    for data in sampled_dataset:
        idx = random.randint(0, data['input_ids'].shape[0] - seqlen - 1)
        inp = data['input_ids'][idx:idx + seqlen].unsqueeze(0)
        tar = inp.clone()
        tar[:, :-1] = -100
        train_loader.append((inp, tar))
    return train_loader


def get_20_newsgroups_tokenizer_dataset(tokenizer, max_length=512):
    newsgroups_train = fetch_20newsgroups(subset='train', remove=('headers', 'footers', 'quotes'))
    newsgroups_test = fetch_20newsgroups(subset='test', remove=('headers', 'footers', 'quotes'))

    train_df = pd.DataFrame({'text': newsgroups_train['data'], 'label': newsgroups_train['target']})
    test_df = pd.DataFrame({'text': newsgroups_test['data'], 'label': newsgroups_test['target']})

    train_dataset = Dataset.from_pandas(train_df)
    test_dataset = Dataset.from_pandas(test_df)

    train_dataset = train_dataset.train_test_split(test_size=0.15, seed=42)

    full_dataset = DatasetDict(
        {'train': train_dataset['train'], 'validation': train_dataset['test'], 'test': test_dataset})

    def tokenize_function(examples):
        return tokenizer(examples['text'], padding='max_length', truncation=True, max_length=max_length)

    tokenized_dataset = full_dataset.map(tokenize_function, batched=True)
    tokenized_dataset = tokenized_dataset.remove_columns(['text'])
    tokenized_dataset.set_format("torch")
    return tokenized_dataset


def load_fed_dataset(tokenizer, data_name: str):
    if data_name == '20newsgroup':
        return get_20_newsgroups_tokenizer_dataset(tokenizer)
    if data_name == 'patient2trial':
        return get_patient2trial_matching(tokenizer)
    if data_name == 'trial_design':
        return get_trial_design(tokenizer)


def get_patient2trial_matching(tokenizer, max_length=8192):
    data_path = '/home/wangjinzhi/data/trialPanorama/matching/patient2trial/TREC2021/train.json'
    test_path = '/home/wangjinzhi/data/trialPanorama/matching/patient2trial/TREC2021/test.json'
    with open(data_path, 'r') as f:
        data = json.load(f)
    with open(test_path, 'r') as f:
        test_data = json.load(f)

    output_list = [inner_dict['input'][443:-183] for inner_dict in data.values()]
    label_list = [inner_dict['label'] for inner_dict in data.values()]

    test_output_list = [inner_dict['input'] for inner_dict in test_data.values()]
    test_label_list = [inner_dict['label'] for inner_dict in test_data.values()]

    np_array = np.array(label_list, dtype=np.int32)
    train_df = pd.DataFrame({'text': output_list, 'label': np_array})

    np_array_test = np.array(test_label_list, dtype=np.int32)
    test_df = pd.DataFrame({'text': test_output_list, 'label': np_array_test})

    train_dataset = Dataset.from_pandas(train_df)
    test_dataset = Dataset.from_pandas(test_df)
    combined_train = concatenate_datasets([train_dataset, test_dataset])
    combined_train = combined_train.train_test_split(test_size=0.35, seed=42)
    train_dataset = combined_train['train']
    train_dataset = train_dataset.train_test_split(test_size=0.15, seed=42)

    full_dataset = DatasetDict(
        {'train': train_dataset['train'], 'validation': train_dataset['test'], 'test': combined_train['test']})

    def tokenize_function(examples):
        return tokenizer(examples['text'], padding='max_length', truncation=True, max_length=max_length)

    tokenized_dataset = full_dataset.map(tokenize_function, batched=True)
    tokenized_dataset = tokenized_dataset.remove_columns(['text'])
    tokenized_dataset.set_format("torch")
    return tokenized_dataset


def get_trial_design(tokenizer):
    # task_names = ['criteria', 'study_arms', 'outcome_measures']
    task_name = 'study_arms'
    data_path = '/home/wangjinzhi/data/trialPanorama/design'
    train_path = os.path.join(data_path, task_name, 'train.json')
    test_path = os.path.join(data_path, task_name, 'test.json')

    with open(train_path, 'r') as f:
        train_data = json.load(f)
    with open(test_path, 'r') as f:
        test_data = json.load(f)

    train_data_list = []
    for key, value in train_data.items():
        train_data_list.append(value)
    train_data_dict = [
        {"messages": conversation}
        for conversation in train_data_list
    ]

    train_dataset = Dataset.from_list(train_data_dict)
    train_dataset = train_dataset.train_test_split(test_size=0.99, seed=42)

    default_chat_template = "{% for message in messages %}\n" \
                            "{% if message['role'] == 'user' %}\n" \
                            "{{ '<|user|>\n' + message['content'] + eos_token }}\n" \
                            "{% elif message['role'] == 'system' %}\n" \
                            "{{ '<|system|>\n' + message['content'] + eos_token }}\n" \
                            "{% elif message['role'] == 'assistant' %}\n" \
                            "{{ '<|assistant|>\n'  + message['content'] + eos_token }}\n" \
                            "{% endif %}\n{% if loop.last and add_generation_prompt %}\n" \
                            "{{ '<|assistant|>' }}\n{% endif %}\n{% endfor %}"
    if tokenizer.chat_template is None:
        tokenizer.chat_template = default_chat_template
    full_dataset = DatasetDict(
        {'train': train_dataset['train']})

    def tokenize_function(example):
        messages = example["messages"]
        maybe_insert_system_message(messages, tokenizer)
        tokenized = tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            padding="max_length",  # or padding=True + max_length=1024
            return_dict=True,
            max_length=8192,
            truncation=True,
            add_generation_prompt=False
        )
        tokenized["label"] = tokenized["input_ids"]
        return tokenized

    tokenized_dataset = full_dataset.map(tokenize_function)
    tokenized_dataset = tokenized_dataset.remove_columns(['messages'])
    tokenized_dataset.set_format("torch")
    tokenized_dataset['validation'] = train_dataset['train']
    tokenized_dataset['test'] = test_data
    return tokenized_dataset


def maybe_insert_system_message(messages, tokenizer):
    if messages[0]["role"] == "system":
        return

    chat_template = tokenizer.chat_template
    if chat_template is None:
        chat_template = tokenizer.default_chat_template

    if "system" in chat_template:
        messages.insert(0, {"role": "system", "content": ""})
