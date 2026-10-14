import torch

from pruner import Pruner

# 这个严格来说不能进行比较
# 没有进行剪枝
class FedSPU_pruner(Pruner):
    def __init__(self, args, tokenizer):
        super().__init__(args, tokenizer)

    def client_on_train_begin(self, model, prune_ratio: float, lora_state, raw_lora_state, train_data):
        layers = model.model.layers
        head_num = model.config.num_key_value_heads
        intermediate_size = model.config.intermediate_size
        for i in range(len(layers)):
            attn_salience = torch.rand(head_num, device=model.device)
            self.attn_salience.append(attn_salience)
            mlp_salience = torch.rand(intermediate_size, device=model.device)
            self.mlp_salience.append(mlp_salience)
        self.get_mask(
            model=model,
            prune_ratio=prune_ratio,
        )

    def client_on_train_end(self, model, delta_update):
        head_dim = model.config.head_dim
        group = int(model.config.num_attention_heads / model.config.num_key_value_heads)
        for key in delta_update.keys():
            if 'lora_A' in key:
                layer_idx = int(key.split('.')[4])
                if 'attn' in key:
                    mask = self.attn_mask[layer_idx]
                    mask = mask.repeat_interleave(group * head_dim)
                else:
                    mask = self.mlp_mask[layer_idx]
                delta_update[key] = delta_update[key] * mask
        self.mlp_mask.clear()
        self.attn_mask.clear()