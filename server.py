import copy
import os
import random

import numpy as np
import torch
from peft import LoraConfig, get_peft_model, get_peft_model_state_dict, set_peft_model_state_dict
from sklearn.metrics import accuracy_score, classification_report, balanced_accuracy_score
from tqdm import tqdm
from transformers import TrainingArguments, Trainer

from client import Client
from data import load_fed_dataset, get_partition_map
from model import get_tokenizer, get_score_weight, get_causalLM, get_sequenceclassification
from feddse import FedDSE_pruner
from fedspu import FedSPU_pruner
from heterofl import HeteroFL_pruner
from fedrolex import FedRolex_pruner
from fedmp import fedmp_pruner
from scalefl import ScaleFL_pruner
from oprune import Our_pruner
from pruner import Pruner


class Server:
    def __init__(self, args):
        self.rnd = 0
        self.args = args
        print('load model...')
        self.score_weight = get_score_weight(args.model)
        self.model = get_causalLM(args.model).cpu()
        self.tokenizer = get_tokenizer(args.model)
        print('load dataset...')
        self.dataset = load_fed_dataset(
            tokenizer=self.tokenizer,
            data_name=args.dataset,
        )
        print('init config...')
        self.client_selection = []
        client_sample_num = int(args.sample_fraction * args.client_num)
        for rnd in range(args.comm_round):
            self.client_selection.append(random.sample(range(args.client_num), client_sample_num))

        self.prune_ratio = {}
        ratio = 0.20
        print(ratio)
        for i in range(args.client_num):
            self.prune_ratio[i] = ratio
        # self.prune_ratio = {0: 0.1, 1: 0.1, 2: 0.2, 3: 0.2, 4: 0.3, 5: 0.3, 6: 0.4, 7: 0.4, 8: 0.5, 9: 0.5}

        self.pruner = get_pruner(args.prune_method, args, self.tokenizer)

        self.data_indices = get_partition_map(
            dataset=self.dataset,
            num_client=args.client_num,
            alpha=args.alpha
        )

        self.lora_config = LoraConfig(
            r=args.lora_rank,
            lora_alpha=args.lora_alpha,
            target_modules=["o_proj", "down_proj"],
            bias="none",
            task_type="CAUSAL_LM",
        )
        if args.lora:
            lora_model = get_peft_model(copy.deepcopy(self.model), self.lora_config)
            self.agg_update = get_peft_model_state_dict(lora_model.cuda())
            del lora_model
        else:
            self.agg_update = self.model.state_dict()
            if not self.args.gen:
                self.agg_update.pop('lm_head.weight', None)
                self.agg_update['score.weight'] = self.score_weight
        self.delta_update = copy.deepcopy(self.agg_update)
        # self.delta_update = self.agg_update

        self.attn_mask = []
        self.mlp_mask = []

        self.client = Client(
            args=args,
            tokenizer=self.tokenizer,
            lora_config=self.lora_config,
            downstream_dataset=self.dataset,
            data_indices=self.data_indices,
            pruner=self.pruner
        )
        self.delta_client_update = []
        self.result = []

    def run(self):
        print('--------- run federated learning ---------')
        self.pruner.server_on_init_end(model=self.model)
        while self.rnd < self.args.comm_round:
            self.rnd += 1
            print('--------- rnd {} ---------'.format(self.rnd))
            self.pruner.server_on_rnd_begin(model=self.model)
            self.client_train()
            self.fed_avg()
            self.pruner.server_on_rnd_end(model=self.model, delta_update=self.delta_update)
            print('validation task...')
            self.valid_downstream()
        print('--------- finish federated learning ---------')
        print('eval task...')
        self.eval_downstream()
        self.save_metric()

    def fed_avg(self):
        agg_attn_mask = []
        agg_mlp_mask = []
        group = self.model.model.layers[0].self_attn.num_key_value_groups
        head_dim = self.model.model.layers[0].self_attn.head_dim
        scale_factor = len(self.delta_client_update)

        if len(self.attn_mask) > 0:
            for a_m in self.attn_mask[0]:
                agg_attn_mask.append(torch.zeros_like(a_m.int()))
            for m_m in self.mlp_mask[0]:
                agg_mlp_mask.append(torch.zeros_like(m_m.int()))

            for a_m in self.attn_mask:
                for i in range(len(a_m)):
                    agg_attn_mask[i] += a_m[i].int()

            for m_m in self.mlp_mask:
                for i in range(len(m_m)):
                    agg_mlp_mask[i] += m_m[i].int()

            for a_m in agg_attn_mask:
                a_m[a_m == 0] = 1
            for m_m in agg_mlp_mask:
                m_m[m_m == 0] = 1

        for key in self.agg_update.keys():
            delta_v = torch.zeros_like(self.agg_update[key]).cuda()
            other_keys = ['model.embed_tokens.weight', 'score.weight', 'model.norm.weight', 'lm_head.weight']
            if key in other_keys or 'norm' in key:
                for delta_update in self.delta_client_update:
                    if delta_update.get(key) is not None:
                        delta_v += delta_update[key].cuda()
                delta_v /= scale_factor
            else:
                if self.args.lora:
                    layer = int(key.split('.')[4])
                else:
                    layer = int(key.split('.')[2])

                for delta_update in self.delta_client_update:
                    if delta_update.get(key) is not None:
                        delta_v += delta_update[key].cuda()

                if self.args.lora:
                    if len(agg_mlp_mask) > 0 and 'lora_A' in key:
                        if 'attn' in key:
                            delta_v /= agg_attn_mask[layer].repeat_interleave(head_dim * group).cuda()
                        else:
                            delta_v /= agg_mlp_mask[layer]
                    else:
                        delta_v /= scale_factor
                else:
                    if 'attn' in key:
                        delta_v /= agg_attn_mask[layer].repeat_interleave(head_dim * group).cuda()
                    else:
                        if 'down' in key:
                            delta_v /= agg_mlp_mask[layer]
                        else:
                            delta_v /= agg_mlp_mask[layer].unsqueeze(1)

            restored = delta_v.clone().cuda() + self.agg_update[key].clone().cuda()
            self.delta_update[key] = delta_v.clone().cpu()
            self.agg_update[key] = restored.clone().cpu()
        self.delta_client_update.clear()
        if not self.args.lora and not self.args.gen:
            self.score_weight = self.agg_update['score.weight']

    def client_train(self):
        for client in self.client_selection[self.rnd - 1]:
            print('client {} train...'.format(client))
            delta_state = self.client.train(
                idx=client,
                init_state=copy.deepcopy(self.agg_update),
                model=self.get_model(),
                prune_ratio=self.prune_ratio[client]
            )
            self.pruner.server_on_train_end(
                model=self.model,
                delta_update=delta_state,
                prune_ratio=self.prune_ratio[client]
            )
            self.delta_client_update.append(delta_state)
            if len(self.pruner.attn_mask) > 0:
                self.attn_mask.append(self.pruner.attn_mask)
                self.mlp_mask.append(self.pruner.mlp_mask)

    def get_model(self):
        if self.args.gen:
            return copy.deepcopy(self.model).cuda()
        else:
            return get_sequenceclassification(
                score_weight=self.score_weight,
                args=self.args,
                model=self.args.model,
                causalLM_model=self.model,
            )

    def valid_downstream(self):
        if self.args.gen:
            self.eval_generate(self.dataset['test'])
            return
        model = get_sequenceclassification(
            score_weight=self.score_weight,
            model=self.args.model,
            args=self.args,
            causalLM_model=self.model,
        ).cuda()
        if self.args.lora:
            model = get_peft_model(model, self.lora_config)

            set_peft_model_state_dict(model, self.agg_update, "default")
        else:
            model.load_state_dict(self.agg_update)
        model.config.pad_token_id = self.tokenizer.pad_token_id = 0
        train_args = TrainingArguments(output_dir='./op',
                                       save_strategy='no',
                                       ddp_find_unused_parameters=False,
                                       remove_unused_columns=True,
                                       per_device_eval_batch_size=2,
                                       do_eval=True,
                                       fp16=True)

        trainer = Trainer(
            model=model,
            args=train_args,
            processing_class=self.tokenizer,
            compute_metrics=compute_metrics_all,
            eval_dataset=self.dataset['validation']
        )
        result = trainer.evaluate(metric_key_prefix="eval")
        print(result)
        self.result.append(result['eval_acc'])
        self.attn_mask.clear()
        self.mlp_mask.clear()

    def eval_downstream(self):
        if self.args.gen:
            self.eval_generate(self.dataset['validation'])
            return
        model = get_sequenceclassification(
            score_weight=self.score_weight,
            model=self.args.model,
            args=self.args,
            causalLM_model=self.model,
        ).cuda()
        if self.args.lora:
            model = get_peft_model(model, self.lora_config)

            set_peft_model_state_dict(model, self.agg_update, "default")
        else:
            model.load_state_dict(self.agg_update)
        model.config.pad_token_id = self.tokenizer.pad_token_id = 0
        train_args = TrainingArguments(output_dir='./op',
                                       save_strategy='no',
                                       ddp_find_unused_parameters=False,
                                       remove_unused_columns=True,
                                       per_device_eval_batch_size=2,
                                       do_eval=True,
                                       fp16=True)

        trainer = Trainer(
            model=model,
            args=train_args,
            processing_class=self.tokenizer,
            compute_metrics=compute_metrics_all,
            eval_dataset=self.dataset['test']
        )
        result = trainer.evaluate(metric_key_prefix="eval")
        print(result)
        self.result.append(result['eval_acc'])

    def eval_generate(self, dataset):
        model = copy.deepcopy(self.model).cuda()
        if self.args.lora:
            model = get_peft_model(model, self.lora_config).cuda()

            set_peft_model_state_dict(model, self.agg_update, "default")
        else:
            model.load_state_dict(self.agg_update)
        model.config.pad_token_id = self.tokenizer.pad_token_id = 0
        eval_dataset = dataset
        total_rouge = {"rouge1": 0, "rouge2": 0, "rougeL": 0}
        count = 0
        bleu_score = 0.0
        eval_dataset = {k: eval_dataset[k] for k in random.sample(list(eval_dataset.keys()), 10)}  # test
        for key, value in tqdm(eval_dataset.items()):
            if len(value) % 2 != 0:
                value = value[:-1]
            for i in range(3, len(value) // 2):
                # try:
                count += 1
                inp = value[:i * 2 + 1]
                encodeds = self.tokenizer.apply_chat_template(inp, return_tensors="pt",
                                                              add_generation_prompt=True).cuda()
                generated_ids = model.generate(encodeds, max_new_tokens=512, do_sample=True,
                                               eos_token_id=self.tokenizer.eos_token_id,
                                               pad_token_id=self.tokenizer.eos_token_id)

                response = self.tokenizer.batch_decode(generated_ids[:, encodeds.shape[1]:],
                                                       skip_special_tokens=True)[0]
                from rouge_score import rouge_scorer
                rouge_types = ["rouge1", "rouge2", "rougeL"]
                scorer = rouge_scorer.RougeScorer(rouge_types)

                scores = scorer.score(value[i * 2 + 1]['content'], response)
                from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
                smoothing_function = SmoothingFunction().method1
                bleu_score += sentence_bleu(
                    references=[value[i * 2 + 1]['content'].split()],
                    hypothesis=response.split(),
                    weights=(0.25, 0.25, 0.25, 0.25),  # 4-gram均等权重
                    smoothing_function=smoothing_function,
                )
                for k in total_rouge:
                    total_rouge[k] += scores[k].fmeasure
        if count != 0:
            for k in total_rouge:
                total_rouge[k] /= count
            bleu_score /= count
        print('rouge score:', total_rouge)
        print('bleu score:', bleu_score)
        del model

    def save_metric(self):
        name = "acc_{}_{}_{}_{}.txt".format(
            self.args.prune_method,
            self.args.comm_round,
            self.args.model,
            self.args.alpha,
        )
        path = os.path.join('/home/wangjinzhi/fed/result', name)
        with open(path, 'w') as file:
            for item in self.result:
                file.write(str(item) + '\n')


def compute_metrics_all(eval_pred):
    predictions, references = eval_pred
    preds = np.argmax(predictions, axis=-1)
    accuracy = accuracy_score(references, preds)
    print(f"Accuracy: {accuracy:.4f}")
    report = classification_report(references, preds, output_dict=True, zero_division=0)
    print(f"Precision: {report['weighted avg']['precision']:.4f}")
    print(f"Recall: {report['weighted avg']['recall']:.4f}")
    print(f"F1-score: {report['weighted avg']['f1-score']:.4f}")
    balanced_accuracy = balanced_accuracy_score(references, preds)
    print(f"Balanced accuracy: {balanced_accuracy:.4f}")
    from sklearn.metrics import cohen_kappa_score
    kappa = cohen_kappa_score(references, preds)
    print(f"Kappa: {kappa:.4f}")
    return {"acc": accuracy}


def get_pruner(prune_method, args, tokenizer):
    if prune_method == 'o':
        print("Our_pruner")
        return Our_pruner(args, tokenizer)
    elif prune_method == 'FedDSE':
        print('FedDSE_pruner')
        return FedDSE_pruner(args, tokenizer)
    # elif prune_method == 'FedSPU':
    #     print('FedSPU_pruner')
    #     return FedSPU_pruner(args, tokenizer)
    elif prune_method == 'HeteroFL':
        print("HeteroFL_pruner")
        return HeteroFL_pruner(args, tokenizer)
    elif prune_method == 'FedRolex':
        print("FedRolex_pruner")
        return FedRolex_pruner(args, tokenizer)
    elif prune_method == 'FedMP':
        print("FedMP_pruner")
        return fedmp_pruner(args, tokenizer)
    elif prune_method == 'ScaleFL':
        print("ScaleFL_pruner")
        return ScaleFL_pruner(args, tokenizer)
    else:
        print('Pruner')
        return Pruner(args, tokenizer)
