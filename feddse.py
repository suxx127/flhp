import torch
from torch import nn

from data import prepare_calibration_dataset
from pruner import Pruner, find_layers, prune_lora, prune_client_model, un_prune_lora, un_prune_full


class FedDSE_pruner(Pruner):
    def __init__(self, args, tokenizer):
        super().__init__(args, tokenizer)

    def client_on_train_begin(self, model, prune_ratio: float, lora_state, raw_lora_state, train_data):
        self.attn_salience.clear()
        self.mlp_salience.clear()
        use_cache = model.config.use_cache
        model.config.use_cache = False
        layers = model.model.layers
        head_dim = model.model.layers[0].self_attn.head_dim
        group = model.model.layers[0].self_attn.num_key_value_groups

        with torch.no_grad():
            inps, outs, attention_mask, position_ids = prepare_calibration_dataset(
                model=model,
                tokenizer=self.tokenizer,
                calib_data_name='20newsgroup',
                train_dataset=train_data,
            )
        for i in range(len(layers)):
            layer = layers[i].cuda()
            subset = {}
            subset.update({'self_attn.o_proj': find_layers(layer)['self_attn.o_proj']})
            subset.update({'mlp.down_proj': find_layers(layer)['mlp.down_proj']})
            pruner_dict = {}
            for module in subset:
                pruner_dict[module] = feddse(
                    layer=subset[module],
                    attn=True if 'attn' in module else False,
                )

            def add_batch(name):
                def func(_, inp, out):
                    pruner_dict[name].add_batch(inp[0].data, out.data)

                return func

            handles = []
            for module in subset:
                handles.append(subset[module].register_forward_hook(add_batch(module)))
            for j in range(self.args.nsamples):
                with torch.no_grad():
                    outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask, position_ids=position_ids)[0]
            for h in handles:
                h.remove()

            for module in subset:
                salience = pruner_dict[module].get_salience(
                    head_dim=head_dim,
                    group=group
                )
                if 'attn' in module:
                    self.attn_salience.append(salience)
                else:
                    self.mlp_salience.append(salience)
            torch.cuda.empty_cache()
            inps, outs = outs, inps
        model.config.use_cache = use_cache

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
        attn_mask, mlp_mask = self.get_mask(
            model=model,
            prune_ratio=prune_ratio
        )
        if self.args.lora:
            un_prune_lora(
                model=model,
                lora_state=delta_update,
                attn_mask=attn_mask,
                mlp_mask=mlp_mask
            )
        else:
            un_prune_full(
                model=model,
                model_state=delta_update,
                attn_mask=attn_mask,
                mlp_mask=mlp_mask
            )


class feddse:
    def __init__(self, layer, attn: bool):
        self.layer = layer
        W = layer.weight.data.clone()
        self.dev = W.device
        self.rows = W.shape[0]
        self.columns = W.shape[1]

        self.nsamples = 0
        self.salience = torch.zeros(self.columns, device=self.dev)
        self.attn = attn

    def add_batch(self, inp, out):
        if len(inp.shape) == 2:
            inp = inp.unsqueeze(0)
        tmp = inp.shape[0]
        if isinstance(self.layer, nn.Linear):
            if len(inp.shape) == 3:
                inp = inp.reshape((-1, inp.shape[-1]))
            inp = inp.t()
        inp = torch.relu(inp)
        self.salience *= self.nsamples / (self.nsamples + tmp)
        self.nsamples += tmp
        self.salience += torch.norm(inp, p=1, dim=1) / self.nsamples

    def get_salience(self, head_dim: int, group: int = 1):
        if self.attn:
            salience = self.salience
            salience = salience.view(group, -1).sum(dim=0)
            salience = salience.view(-1, head_dim).sum(dim=1)
            salience = salience / salience.max()
        else:
            salience = torch.sqrt(self.salience)
            salience = salience / salience.max()
        return salience
