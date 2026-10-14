import torch

from pruner import Pruner, un_prune_lora, prune_lora, prune_client_model, un_prune_full


class FedRolex_pruner(Pruner):
    def __init__(self, args, tokenizer):
        super().__init__(args, tokenizer)
        self.rnd = 0

    def server_on_rnd_end(self, model, delta_update):
        self.rnd += 1

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

    def client_on_train_begin(self, model, prune_ratio: float, lora_state, raw_lora_state, train_data):
        attn_mask = []
        mlp_mask = []
        intermediate_size = model.config.intermediate_size
        head_dim = model.model.layers[0].self_attn.head_dim
        group = model.model.layers[0].self_attn.num_key_value_groups
        layers = model.model.layers
        num_key_value_heads = model.model.layers[0].self_attn.num_key_value_heads
        prune_attn_num = int(num_key_value_heads * prune_ratio)
        all_params = 3 * intermediate_size + 2 * head_dim * num_key_value_heads * (group + 1)
        all_prune_num = int(all_params * prune_ratio)
        prune_mlp_num = all_prune_num - 2 * head_dim * (group + 1) * prune_attn_num
        prune_mlp_num = int(prune_mlp_num / 3)
        for _ in range(len(layers)):
            relative_pos = (torch.arange(num_key_value_heads) -
                            self.rnd % num_key_value_heads) % num_key_value_heads
            a_m = relative_pos < (num_key_value_heads - prune_attn_num)
            relative_pos = (torch.arange(intermediate_size) -
                            self.rnd % intermediate_size) % intermediate_size
            m_m = relative_pos < (intermediate_size - prune_mlp_num)
            attn_mask.append(a_m.cuda())
            mlp_mask.append(m_m.cuda())
        self.attn_mask = attn_mask
        self.mlp_mask = mlp_mask
        if self.args.lora:
            for k in lora_state.keys():
                lora_state[k] = lora_state[k].cuda()
                raw_lora_state[k] = raw_lora_state[k].cuda()
            prune_lora(
                model=model,
                lora_state=lora_state,
                attn_mask=attn_mask,
                mlp_mask=mlp_mask
            )
            prune_lora(
                model=model,
                lora_state=raw_lora_state,
                attn_mask=attn_mask,
                mlp_mask=mlp_mask
            )
        prune_client_model(
            model=model,
            mlp_mask=mlp_mask,
            attn_mask=attn_mask,
        )
