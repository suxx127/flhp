import math

import torch
from torch import nn


class Pruner:
    def __init__(self, args, tokenizer):
        self.args = args
        self.tokenizer = tokenizer
        self.attn_salience = []
        self.mlp_salience = []
        self.attn_mask = []
        self.mlp_mask = []

    def server_on_init_end(self, model):
        pass

    def client_on_train_begin(self, model, prune_ratio: float, lora_state, raw_lora_state, train_data):
        pass

    def client_on_train_end(self, model, delta_update):
        add_dp_noise_to_updates(delta_update)
        pass

    def server_on_rnd_begin(self, model):
        pass

    def server_on_train_end(self, model, delta_update, prune_ratio: float):
        pass

    def server_on_rnd_end(self, model, delta_update):
        pass

    def get_cur_mask(self):
        return self.attn_mask, self.mlp_mask

    def get_mask(self, model, prune_ratio: float):
        attn_mask = []
        mlp_mask = []
        intermediate_size = model.config.intermediate_size
        head_dim = model.model.layers[0].self_attn.head_dim
        group = model.model.layers[0].self_attn.num_key_value_groups
        num_key_value_heads = model.model.layers[0].self_attn.num_key_value_heads
        prune_attn_num = int(num_key_value_heads * prune_ratio)
        all_params = 3 * intermediate_size + 2 * head_dim * num_key_value_heads * (group + 1)
        all_prune_num = int(all_params * prune_ratio)
        prune_mlp_num = all_prune_num - 2 * head_dim * (group + 1) * prune_attn_num
        prune_mlp_num = int(prune_mlp_num / 3)
        for att_s in self.attn_salience:
            _, indices = torch.topk(att_s, prune_attn_num, largest=False)
            W_mask = torch.ones_like(att_s, dtype=torch.bool)
            W_mask[indices] = False
            attn_mask.append(W_mask)
        for mlp_s in self.mlp_salience:
            _, indices = torch.topk(mlp_s, prune_mlp_num, largest=False)
            W_mask = torch.ones_like(mlp_s, dtype=torch.bool)
            W_mask[indices] = False
            mlp_mask.append(W_mask)
        self.attn_mask = attn_mask
        self.mlp_mask = mlp_mask
        return attn_mask, mlp_mask


def prune_client_model(model, attn_mask, mlp_mask):
    head_dim = model.model.layers[0].self_attn.head_dim
    group = model.model.layers[0].self_attn.num_key_value_groups
    layers = model.model.layers
    for i in range(len(layers)):
        layer = layers[i]
        prune(layer, attn_mask[i], mlp_mask[i], head_dim, group)
    # model.config.hidden_size = model.model.embed_tokens.weight.shape[1]
    # model.config.intermediate_size = model.model.layers[0].mlp.gate_proj.weight.data.shape[0]
    # model.config.num_attention_heads = model.model.layers[0].self_attn.num_heads
    # model.config.num_key_value_heads = model.model.layers[0].self_attn.num_key_value_heads
    # model.config.num_hidden_layers = len(model.model.layers)
    #
    # config = model.config
    # state = model.cuda().state_dict()
    # model.__init__(config)
    # model.load_state_dict(state)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")
    return


def prune(layer, attn_mask, mlp_mask, head_dim, group=1):
    # Attention Weight Pruning
    retain_heads = torch.count_nonzero(attn_mask)
    kv_attn_mask = attn_mask.repeat_interleave(head_dim)
    attn_mask = attn_mask.repeat_interleave(group * head_dim)

    layer.self_attn.q_proj.weight.data = layer.self_attn.q_proj.weight.data[torch.where(attn_mask)[0]]
    layer.self_attn.k_proj.weight.data = layer.self_attn.k_proj.weight.data[torch.where(kv_attn_mask)[0]]
    layer.self_attn.v_proj.weight.data = layer.self_attn.v_proj.weight.data[torch.where(kv_attn_mask)[0]]

    layer.self_attn.q_proj.out_features = attn_mask.sum().item()
    layer.self_attn.k_proj.out_features = kv_attn_mask.sum().item()
    layer.self_attn.v_proj.out_features = kv_attn_mask.sum().item()

    layer.self_attn.o_proj.weight.data = layer.self_attn.o_proj.weight.data[:, torch.where(attn_mask)[0]]

    layer.self_attn.o_proj.in_features = attn_mask.sum().item()

    layer.self_attn.num_heads = retain_heads * group
    layer.self_attn.num_key_value_heads = retain_heads
    layer.self_attn.hidden_size = retain_heads * head_dim

    # MLP Weight Pruning

    layer.mlp.up_proj.weight.data = layer.mlp.up_proj.weight.data[torch.where(mlp_mask)[0]]
    layer.mlp.gate_proj.weight.data = layer.mlp.gate_proj.weight.data[torch.where(mlp_mask)[0]]

    layer.mlp.down_proj.weight.data = layer.mlp.down_proj.weight.data[:, torch.where(mlp_mask)[0]]

    layer.mlp.up_proj.out_features = mlp_mask.sum().item()
    layer.mlp.gate_proj.out_features = mlp_mask.sum().item()
    layer.mlp.down_proj.in_features = mlp_mask.sum().item()

    layer.mlp.intermediate_size = mlp_mask.sum().item()

    torch.cuda.empty_cache()
    return layer


# def prune(layer, attn_mask, mlp_mask, head_dim, group=1):
#     # 创建与权重形状相同的掩码，初始化为1（保留）
#     device = layer.self_attn.q_proj.weight.data.device
#
#     # Attention Weight Pruning - 创建掩码并置零
#     retain_heads = torch.count_nonzero(attn_mask)
#     kv_attn_mask = attn_mask.repeat_interleave(head_dim)
#     attn_mask = attn_mask.repeat_interleave(group * head_dim)
#
#     # 为Q、K、V、O投影创建掩码
#     # Q_proj掩码：attn_mask形状扩展为输出特征维度
#     q_mask = attn_mask.unsqueeze(1).float().to(device)  # 扩展为2D掩码
#     # K、V_proj掩码：kv_attn_mask形状扩展
#     kv_mask = kv_attn_mask.unsqueeze(1).float().to(device)
#     # O_proj掩码：attn_mask形状扩展为输入特征维度
#     o_mask = attn_mask.unsqueeze(0).float().to(device)
#
#     # 应用掩码：将不重要的权重置零
#     layer.self_attn.q_proj.weight.data *= q_mask
#     layer.self_attn.k_proj.weight.data *= kv_mask
#     layer.self_attn.v_proj.weight.data *= kv_mask
#     layer.self_attn.o_proj.weight.data *= o_mask
#
#     # MLP Weight Pruning - 创建掩码并置零
#     # up_proj和gate_proj掩码：mlp_mask形状扩展为输出特征维度
#     mlp_output_mask = mlp_mask.unsqueeze(1).float().to(device)
#     # down_proj掩码：mlp_mask形状扩展为输入特征维度
#     mlp_input_mask = mlp_mask.unsqueeze(0).float().to(device)
#
#     layer.mlp.up_proj.weight.data *= mlp_output_mask
#     layer.mlp.gate_proj.weight.data *= mlp_output_mask
#     layer.mlp.down_proj.weight.data *= mlp_input_mask
#     torch.cuda.empty_cache()
#     return layer

def prune_lora(model, lora_state, attn_mask, mlp_mask):
    head_dim = model.model.layers[0].self_attn.head_dim
    group = model.model.layers[0].self_attn.num_key_value_groups
    for key, value in lora_state.items():
        if 'lora_A' in key:
            layer_idx = int(key.split('.')[4])
            if 'attn' in key:
                mask = attn_mask[layer_idx]
                mask = mask.repeat_interleave(group * head_dim)
            else:
                mask = mlp_mask[layer_idx]
            lora_state[key] = lora_state[key][:, torch.where(mask)[0]]


def un_prune_lora(model, lora_state, attn_mask, mlp_mask):
    head_dim = model.model.layers[0].self_attn.head_dim
    group = model.model.layers[0].self_attn.num_key_value_groups
    for key, value in lora_state.items():
        if 'lora_A' in key:
            layer_idx = int(key.split('.')[4])
            if 'attn' in key:
                mask = attn_mask[layer_idx]
                mask = mask.repeat_interleave(group * head_dim)
            else:
                mask = mlp_mask[layer_idx]
            unmask_size = mask.size(0)
            rank = lora_state[key].size(0)
            weights_unmasked = torch.zeros([rank, unmask_size]).cuda()
            valid_indices = torch.where(mask)[0]
            weights_unmasked[:, valid_indices] = lora_state[key]
            lora_state[key] = weights_unmasked


def un_prune_full(model, model_state, attn_mask, mlp_mask):
    # return
    head_dim = model.model.layers[0].self_attn.head_dim
    group = model.model.layers[0].self_attn.num_key_value_groups
    for key, value in model_state.items():
        other_keys = ['model.embed_tokens.weight', 'score.weight', 'model.norm.weight', 'lm_head.weight']
        if key in other_keys:
            continue
        if 'layernorm' in key:
            continue
        layer_idx = int(key.split('.')[2])
        if 'q_proj' in key or 'o_proj' in key:
            mask = attn_mask[layer_idx]
            mask = mask.repeat_interleave(group * head_dim)
        elif 'k_proj' in key or 'v_proj' in key:
            mask = attn_mask[layer_idx]
            mask = mask.repeat_interleave(head_dim)
        else:
            mask = mlp_mask[layer_idx]
        unmask_size = mask.size(0)

        if 'o_proj' in key or 'down_proj' in key:
            hidden_size = model_state[key].size(0)
            weights_unmasked = torch.zeros([hidden_size, unmask_size]).cpu()
            valid_indices = torch.where(mask)[0]
            weights_unmasked[:, valid_indices] = model_state[key].cpu()
            model_state[key] = weights_unmasked
        else:
            hidden_size = model_state[key].size(1)
            weights_unmasked = torch.zeros([unmask_size, hidden_size]).cpu()
            valid_indices = torch.where(mask)[0]
            weights_unmasked[valid_indices] = model_state[key].cpu()
            model_state[key] = weights_unmasked


def find_layers(module, layers=None, name=''):
    if layers is None:
        layers = [nn.Conv2d, nn.Linear]
    if type(module) in layers:
        return {name: module}
    res = {}
    for name1, child in module.named_children():
        res.update(find_layers(
            child, layers=layers, name=name + '.' + name1 if name != '' else name1
        ))
    return res


def add_dp_noise_to_updates(delta_update, epsilon=10.0, delta=1e-3, clip_norm=1.0):
    print("dp")
    sensitivity = clip_norm
    noise_stddev = sensitivity * math.sqrt(2 * math.log(1.25 / delta)) / epsilon
    for key, value in delta_update.items():

        if not isinstance(value, torch.Tensor):
            value = torch.tensor(value)

        current_norm = torch.norm(value)

        noise_stddev = current_norm * math.sqrt(2 * math.log(1.25 / delta)) / epsilon
        noise = torch.normal(mean=0.0, std=noise_stddev, size=value.shape).cuda()
        noisy_value = value + noise /10

        delta_update[key] = noisy_value
