import math

import torch

from pruner import Pruner, prune_lora, prune_client_model, un_prune_lora, un_prune_full


def prune_lora_width_depth(model, lora_state, attn_mask, mlp_mask, depth: int):
    keys_to_delete = []
    for key in lora_state.keys():
        layer_idx = int(key.split('.')[4])
        if layer_idx >= depth:
            keys_to_delete.append(key)

    for key in keys_to_delete:
        del lora_state[key]

    prune_lora(
        model=model,
        lora_state=lora_state,
        attn_mask=attn_mask,
        mlp_mask=mlp_mask
    )


class ScaleFL_pruner(Pruner):
    def __init__(self, args, tokenizer):
        super().__init__(args, tokenizer)
        self.depth_prserve = 100

    def client_on_train_begin(self, model, prune_ratio: float, lora_state, raw_lora_state, train_data):
        width_ratio = depth_ratio = math.sqrt(1 - prune_ratio)
        width_ratio = 1 - width_ratio
        layer_depth = len(model.model.layers)
        depth_prserve = int(layer_depth * depth_ratio)
        self.depth_prserve = depth_prserve
        model.model.layers = model.model.layers[:depth_prserve]
        attn_mask = []
        mlp_mask = []
        intermediate_size = model.config.intermediate_size
        head_dim = model.model.layers[0].self_attn.head_dim
        group = model.model.layers[0].self_attn.num_key_value_groups
        layers = model.model.layers
        num_key_value_heads = model.model.layers[0].self_attn.num_key_value_heads
        prune_attn_num = int(num_key_value_heads * width_ratio)
        all_params = 3 * intermediate_size + 2 * head_dim * num_key_value_heads * (group + 1)
        all_prune_num = int(all_params * width_ratio)
        prune_mlp_num = all_prune_num - 2 * head_dim * (group + 1) * prune_attn_num
        prune_mlp_num = int(prune_mlp_num / 3)

        for _ in range(len(layers)):
            a_m = torch.arange(num_key_value_heads) < (num_key_value_heads - prune_attn_num)
            m_m = torch.arange(intermediate_size) < (intermediate_size - prune_mlp_num)
            attn_mask.append(a_m.cuda())
            mlp_mask.append(m_m.cuda())

        self.attn_mask = attn_mask
        self.mlp_mask = mlp_mask
        if self.args.lora:
            for k in lora_state.keys():
                lora_state[k] = lora_state[k].cuda()
                raw_lora_state[k] = raw_lora_state[k].cuda()
            prune_lora_width_depth(
                model=model,
                lora_state=lora_state,
                attn_mask=attn_mask,
                mlp_mask=mlp_mask,
                depth=depth_prserve,
            )
            prune_lora_width_depth(
                model=model,
                lora_state=raw_lora_state,
                attn_mask=attn_mask,
                mlp_mask=mlp_mask,
                depth=depth_prserve,
            )
        prune_client_model(
            model=model,
            mlp_mask=mlp_mask,
            attn_mask=attn_mask,
        )
        pad_tensor_list(self.attn_mask, layer_depth)
        pad_tensor_list(self.mlp_mask, layer_depth)

    def server_on_train_end(self, model, delta_update, prune_ratio: float):
        if self.args.lora:
            un_prune_lora(
                model=model,
                lora_state=delta_update,
                attn_mask=self.attn_mask,
                mlp_mask=self.mlp_mask
            )
        else:
            un_prune_full(
                model=model,
                model_state=delta_update,
                attn_mask=self.attn_mask,
                mlp_mask=self.mlp_mask
            )


def pad_tensor_list(tensor_list: list[torch.Tensor], k: int) -> list[torch.Tensor]:
    current_len = len(tensor_list)

    if current_len == k:
        return tensor_list

    sample_tensor = tensor_list[0]
    tensor_shape = sample_tensor.shape
    device = sample_tensor.device

    false_tensor = torch.zeros(tensor_shape, dtype=torch.bool, device=device)

    num_pad = k - current_len
    tensor_list.extend([false_tensor] * num_pad)

    return tensor_list
