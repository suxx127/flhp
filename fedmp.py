import torch

from pruner import Pruner, find_layers, prune_lora, prune_client_model, un_prune_lora, un_prune_full


class fedmp_pruner(Pruner):
    def __init__(self, args, tokenizer):
        super().__init__(args, tokenizer)

    def server_on_rnd_begin(self, model):
        head_dim = model.model.layers[0].self_attn.head_dim
        group = model.model.layers[0].self_attn.num_key_value_groups
        hidden_size = model.config.hidden_size
        layers = model.model.layers
        for i in range(len(layers)):
            layer = layers[i]
            attn_layer = [layer.self_attn.o_proj, layer.self_attn.q_proj, layer.self_attn.k_proj,
                          layer.self_attn.v_proj]
            # attn_layer = [layer.self_attn.o_proj]
            all_salience_attn = {}
            for sub_layer in attn_layer:
                salience = sub_layer.weight.data.abs()
                if sub_layer is layer.self_attn.o_proj:
                    salience = salience.view(hidden_size, group, -1).mean(dim=1)
                    salience = salience.t()
                elif sub_layer is layer.self_attn.q_proj:
                    salience = salience.view(group, -1, hidden_size).mean(dim=0)
                all_salience_attn[sub_layer] = salience.abs()
            salience_attn = sum(all_salience_attn.values()).t()

            all_salience_mlp = {}
            mlp_layer = [layer.mlp.up_proj, layer.mlp.down_proj, layer.mlp.gate_proj]
            # mlp_layer = [layer.mlp.down_proj]
            for sub_layer in mlp_layer:
                salience = sub_layer.weight.data
                if sub_layer is layer.mlp.down_proj:
                    salience = salience.t()
                all_salience_mlp[sub_layer] = salience.abs()
            salience_mlp = sum(all_salience_mlp.values()).t()

            salience_attn = salience_attn.mean(axis=0).reshape(-1, head_dim).sum(dim=1)
            salience_mlp = salience_mlp.mean(axis=0)
            self.attn_salience.append(salience_attn.cuda())
            self.mlp_salience.append(salience_mlp.cuda())

    def client_on_train_begin(self, model, prune_ratio: float, lora_state, raw_lora_state, train_data):
        attn_mask, mlp_mask = self.get_mask(
            model=model,
            prune_ratio=prune_ratio
        )
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
