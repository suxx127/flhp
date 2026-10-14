import copy
import time

import numpy as np
import torch
from peft import get_peft_model, set_peft_model_state_dict, get_peft_model_state_dict
from torch.utils.data import Subset
from transformers import TrainingArguments, Trainer, LlamaConfig


def compute_metrics(eval_pred):
    predictions, references = eval_pred
    preds = np.argmax(predictions, axis=-1)
    acc = float((preds == references).mean())
    return {"acc": acc}


class Client:
    def __init__(self, args, tokenizer, lora_config, downstream_dataset, data_indices, pruner):
        self.tokenizer = tokenizer
        self.args = args
        self.lora_config = lora_config
        self.dataset = downstream_dataset
        self.data_indices = data_indices
        self.pruner = pruner

    def train_full(self, idx, model, prune_ratio):
        print(sum(p.numel() for p in model.parameters()))
        self.pruner.client_on_train_begin(
            model=model,
            prune_ratio=prune_ratio,
            raw_lora_state=None,
            lora_state=None,
            train_data=self.dataset['train'],
        )
        init_state = {
            k: v.detach().cpu().clone()
            for k, v in model.state_dict().items()
        }

        model.config.pad_token_id = self.tokenizer.pad_token_id = 0
        for n, p in model.named_parameters():
            if 'embed_tokens' in n:
                p.requires_grad = False
                continue
            if not p.requires_grad:
                continue  # frozen weights
            p.data = p.data.to(torch.float32)

        train_args = TrainingArguments(output_dir='./op',
                                       save_strategy='no',
                                       num_train_epochs=2,
                                       per_device_train_batch_size=self.args.bs, do_train=True,
                                       learning_rate=self.args.lr,
                                       seed=self.args.seed,
                                       eval_strategy="no",
                                       ddp_find_unused_parameters=False,
                                       remove_unused_columns=True,
                                       fp16=True,
                                       )
        trainer = Trainer(
            model=model,
            args=train_args,
            processing_class=self.tokenizer,
            train_dataset=Subset(self.dataset["train"], self.data_indices[idx]),
            compute_metrics=compute_metrics,
            eval_dataset=self.dataset['validation'],

        )
        trainer.train()
        # metrics = trainer.state.log_history
        # print(metrics)
        # flops = next((m['total_flos'] for m in metrics if 'total_flos' in m), None)
        # step = next((m['step'] for m in metrics if 'step' in m), None)
        # if flops:
        #     print(f"FLOPs: {flops/step / 1e9:.2f} GFLOPs")
        # else:
        #     print('failed')
        model_state = model.state_dict()
        torch.cuda.empty_cache()
        delta_update = get_delta_state(model_state, init_state)
        self.pruner.client_on_train_end(model=model, delta_update=delta_update)
        for k in delta_update.keys():
            delta_update[k] = delta_update[k].cpu()
        del model
        torch.cuda.empty_cache()
        return delta_update

    def train_lora(self, idx, model, init_lora_state, prune_ratio):
        raw_lora_state = copy.deepcopy(init_lora_state)
        self.pruner.client_on_train_begin(
            model=model,
            prune_ratio=prune_ratio,
            raw_lora_state=raw_lora_state,
            lora_state=init_lora_state,
            train_data=self.dataset['train'],
        )
        model = get_peft_model(model, self.lora_config)
        model.config.pad_token_id = self.tokenizer.pad_token_id = 0
        set_peft_model_state_dict(model, init_lora_state, "default")
        train_args = TrainingArguments(output_dir='./op',
                                       save_strategy='no',
                                       num_train_epochs=2,
                                       per_device_train_batch_size=self.args.bs, do_train=True,
                                       learning_rate=self.args.lr,
                                       seed=self.args.seed,
                                       eval_strategy="no",
                                       ddp_find_unused_parameters=False,
                                       remove_unused_columns=True,
                                       fp16=True,
                                       )

        trainer = Trainer(
            model=model,
            args=train_args,
            processing_class=self.tokenizer,
            train_dataset=Subset(self.dataset["train"], self.data_indices[idx]),
            compute_metrics=compute_metrics,
            eval_dataset=self.dataset['validation'],
        )
        trainer.train()
        lora_state = get_peft_model_state_dict(model)
        delta_lora_state = get_delta_state(lora_state, raw_lora_state)
        self.pruner.client_on_train_end(model=model, delta_update=delta_lora_state)
        del model
        torch.cuda.empty_cache()
        return delta_lora_state

    def train(self, idx, model, init_state, prune_ratio):
        if self.args.lora:
            delta_state = self.train_lora(
                idx=idx,
                init_lora_state=init_state,
                model=model,
                prune_ratio=prune_ratio,
            )
        else:
            model.load_state_dict(init_state)
            delta_state = self.train_full(
                idx=idx,
                model=model,
                prune_ratio=prune_ratio,
            )
        return delta_state


def get_delta_state(train_state, init_state):
    delta_state = copy.deepcopy(train_state)
    for key in delta_state.keys():
        delta_state[key] -= init_state[key].cuda()
    return delta_state
