import torch
from torch import nn

from data import prepare_calibration_dataset
from pruner import Pruner, prune_client_model, prune_lora, un_prune_lora, find_layers, un_prune_full


class Our_pruner(Pruner):
    def __init__(self, args, tokenizer):
        self.layer_prune_ratios = None
        self.layer_salience = []
        super().__init__(args, tokenizer)

    def server_on_init_end(self, model):
        use_cache = model.config.use_cache
        model.config.use_cache = False
        layers = model.model.layers
        head_dim = model.model.layers[0].self_attn.head_dim
        group = model.model.layers[0].self_attn.num_key_value_groups

        with torch.no_grad():
            inps, outs, attention_mask, position_ids = prepare_calibration_dataset(
                model=model,
                tokenizer=self.tokenizer,
                # calib_data_name='20newsgroup',
            )

        for i in range(len(layers)):
            layer = layers[i].cuda()
            subset = {}
            subset.update({'self_attn.o_proj': find_layers(layer)['self_attn.o_proj']})
            subset.update({'mlp.down_proj': find_layers(layer)['mlp.down_proj']})
            pruner_dict = {}
            for module in subset:
                pruner_dict[module] = Our(
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

            ls = 0.0
            for module in subset:
                salience, layer_salience = pruner_dict[module].get_salience(
                    head_dim=head_dim,
                    group=group
                )
                ls += layer_salience
                if 'attn' in module:
                    self.attn_salience.append(salience)
                else:
                    self.mlp_salience.append(salience)
            self.layer_salience.append(ls)
            # print(ls)
            layer.cpu()
            inps, outs = outs, inps
            torch.cuda.empty_cache()
        model.config.use_cache = use_cache

    def client_on_train_begin(self, model, prune_ratio: float, lora_state, raw_lora_state, train_data):
        # attn_mask, mlp_mask = self.get_mask(
        # attn_mask, mlp_mask = self.get_mask_global(
        attn_mask, mlp_mask = self.get_mask_layer(
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
        attn_mask, mlp_mask = self.get_cur_mask()
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

    def server_on_rnd_end(self, model, delta_update):
        beta = 0.9
        head_dim = model.model.layers[0].self_attn.head_dim
        group = model.model.layers[0].self_attn.num_key_value_groups
        # self.layer_salience = []
        # self.layer_prune_ratios = None
        if self.args.lora:
            for key in delta_update:
                layer = int(key.split('.')[4])
                if "lora_A" in key:
                    if 'attn' in key:
                        salience = delta_update[key].abs().sum(dim=0)
                        salience = salience.reshape(-1, group).sum(dim=1)
                        salience = salience.reshape(-1, head_dim).sum(dim=1)
                        salience /= salience.max()
                        salience = get_overall_gradient_importance(salience)
                        self.attn_salience[layer] = self.attn_salience[layer] * beta + (1 - beta) * salience.cuda()
                        self.attn_salience[layer].to('cpu')
                    else:
                        salience = delta_update[key].abs().sum(dim=0)
                        salience /= salience.max()
                        salience = get_overall_gradient_importance(salience)
                        self.mlp_salience[layer] = self.mlp_salience[layer] * beta + (1 - beta) * salience.cuda()
                        self.mlp_salience[layer].to('cpu')
        else:
            layer_salience = 0.0
            ratio = 50
            for key in delta_update:
                if 'o_proj' in key:
                    # layer_salience = 0.0
                    layer = int(key.split('.')[2])
                    salience = delta_update[key].abs().sum(dim=0)
                    salience = salience.reshape(-1, group).sum(dim=1)
                    salience = salience.reshape(-1, head_dim).sum(dim=1)
                    salience /= salience.max()
                    mask = salience == 0
                    # salience1 = get_overall_gradient_importance(salience)
                    w = model.state_dict()[key].clone()
                    k = salience.size()[0]
                    w = w.abs().reshape(-1, k, group * head_dim).reshape(k, -1)
                    # if layer == 3:
                    #     for i in range(1, 6):
                    #         r = i * 20
                    #         print(estimate_masked_grad_norms(w, salience.clone(), mask, r))
                    # layer_salience += salience.std()
                    # salience = estimate_masked_grad_norms(w, salience, mask, ratio)
                    # ranks = torch.argsort(torch.argsort(salience))
                    # print(ranks)
                    # print(salience1)
                    self.attn_salience[layer] = self.attn_salience[layer] * beta + (1 - beta) * salience.cuda()
                elif 'down_proj' in key:
                    salience = delta_update[key].abs().sum(dim=0)
                    salience /= salience.max()
                    # salience = get_overall_gradient_importance(salience)
                    # layer_salience += salience.std()
                    layer = int(key.split('.')[2])
                    mask = salience != 0
                    w = model.state_dict()[key]
                    # layer_salience += salience.std()
                    salience = estimate_masked_grad_norms(w.T, salience, mask, ratio)
                    self.mlp_salience[layer] = self.mlp_salience[layer] * beta + (1 - beta) * salience.cuda()
                    # self.layer_salience.append(layer_salience)
        pass

    def get_mask_global(self, model, prune_ratio: float):
        print("global")
        intermediate_size = model.config.intermediate_size
        head_dim = model.model.layers[0].self_attn.head_dim
        group = model.model.layers[0].self_attn.num_key_value_groups
        num_key_value_heads = model.model.layers[0].self_attn.num_key_value_heads
        num_layers = len(model.model.layers)

        # 计算每层需要剪枝的注意力头数量
        prune_attn_num_per_layer = int(num_key_value_heads * prune_ratio)

        # 计算总参数量和总剪枝量
        all_params_per_layer = 3 * intermediate_size + 2 * head_dim * num_key_value_heads * (group + 1)
        all_prune_num_per_layer = int(all_params_per_layer * prune_ratio)

        # 计算每层需要剪枝的MLP神经元数量
        prune_mlp_num_per_layer = all_prune_num_per_layer - 2 * head_dim * (group + 1) * prune_attn_num_per_layer
        prune_mlp_num_per_layer = int(prune_mlp_num_per_layer / 3)

        # 计算全局需要剪枝的总量
        total_prune_attn = prune_attn_num_per_layer * num_layers
        total_prune_mlp = prune_mlp_num_per_layer * num_layers

        # 合并所有层的注意力显著性分数
        all_attn_salience = torch.cat(self.attn_salience, dim=0)

        # 全局排序，找出最不重要的注意力头
        _, global_attn_indices = torch.topk(all_attn_salience, total_prune_attn, largest=False)

        # 合并所有层的MLP显著性分数
        all_mlp_salience = torch.cat(self.mlp_salience, dim=0)

        # 全局排序，找出最不重要的MLP神经元
        _, global_mlp_indices = torch.topk(all_mlp_salience, total_prune_mlp, largest=False)

        # 初始化掩码列表
        attn_mask = []
        mlp_mask = []

        # 为每层生成注意力掩码
        start_idx = 0
        for i in range(num_layers):
            end_idx = start_idx + self.attn_salience[i].numel()
            layer_mask = torch.ones_like(self.attn_salience[i], dtype=torch.bool)

            # 找出当前层中被选中的索引
            layer_indices = global_attn_indices[(global_attn_indices >= start_idx) &
                                                (global_attn_indices < end_idx)] - start_idx

            layer_mask[layer_indices] = False
            attn_mask.append(layer_mask)
            start_idx = end_idx

        # 为每层生成MLP掩码
        start_idx = 0
        for i in range(num_layers):
            end_idx = start_idx + self.mlp_salience[i].numel()
            layer_mask = torch.ones_like(self.mlp_salience[i], dtype=torch.bool)

            # 找出当前层中被选中的索引
            layer_indices = global_mlp_indices[(global_mlp_indices >= start_idx) &
                                               (global_mlp_indices < end_idx)] - start_idx

            layer_mask[layer_indices] = False
            mlp_mask.append(layer_mask)
            start_idx = end_idx

        self.attn_mask = attn_mask
        self.mlp_mask = mlp_mask
        return attn_mask, mlp_mask

    def compute_layer_prune_ratios(self, model, global_prune_ratio: float):
        num_layers = len(model.model.layers)

        layer_salience_scalars = []
        for salience in self.layer_salience:
            layer_salience_scalars.append(salience)
        weights = torch.tensor(layer_salience_scalars, dtype=torch.float32)
        # weights = (weights - weights.mean()) / weights.std()
        # weights *= global_prune_ratio
        # weights = torch.sigmoid(weights)
        weights = weights / weights.sum()

        layer_prune_budgets = num_layers * global_prune_ratio * weights
        layer_prune_ratios = torch.clamp(layer_prune_budgets, min=0, max=0.9)
        # print(layer_prune_ratios)
        # print(layer_prune_ratios.std())
        return layer_prune_ratios

    def get_mask_layer(self, model, prune_ratio: float):
        print("layer")
        head_dim = model.model.layers[0].self_attn.head_dim
        num_key_value_heads = model.model.layers[0].self_attn.num_key_value_heads
        intermediate_size = model.config.intermediate_size
        group = model.model.layers[0].self_attn.num_key_value_groups
        num_layers = len(model.model.layers)
        layer_prune_ratios = self.compute_layer_prune_ratios(model, prune_ratio)
        attn_mask = []
        mlp_mask = []

        for i in range(num_layers):
            prune_ratio = layer_prune_ratios[i].item()

            prune_attn_num = int(num_key_value_heads * prune_ratio)
            prune_attn_num = min(prune_attn_num, num_key_value_heads - 1)

            all_params = 3 * intermediate_size + 2 * head_dim * num_key_value_heads * (group + 1)
            all_prune_num = int(all_params * prune_ratio)
            prune_mlp_num = all_prune_num - 2 * head_dim * (group + 1) * prune_attn_num
            prune_mlp_num = int(prune_mlp_num / 3)

            # 生成注意力掩码
            attn_salience = self.attn_salience[i]
            _, indices = torch.topk(attn_salience, prune_attn_num, largest=False)
            attn_layer_mask = torch.ones_like(attn_salience, dtype=torch.bool)
            attn_layer_mask[indices] = False
            attn_mask.append(attn_layer_mask)

            # 生成MLP掩码
            mlp_salience = self.mlp_salience[i]
            _, indices = torch.topk(mlp_salience, prune_mlp_num, largest=False)
            mlp_layer_mask = torch.ones_like(mlp_salience, dtype=torch.bool)
            mlp_layer_mask[indices] = False
            mlp_mask.append(mlp_layer_mask)

        self.attn_mask = attn_mask
        self.mlp_mask = mlp_mask
        return attn_mask, mlp_mask


class Our:
    def __init__(self, layer, attn: bool):
        self.layer = layer
        W = layer.weight.data.detach().clone()
        self.weight = W
        self.dev = W.device
        self.rows = W.shape[0]
        self.columns = W.shape[1]
        self.xi1 = 0.7
        self.xi2 = 1.6
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

        inp = inp.abs()
        self.salience *= self.nsamples / (self.nsamples + tmp)
        self.nsamples += tmp
        w = self.weight.abs()
        inp = inp / inp.max()
        s = torch.norm(inp ** self.xi2, p=1, dim=1) * \
            torch.norm(w, p=1, dim=0) ** self.xi1
        # print(s)

        self.salience += s / self.nsamples
        # self.salience += torch.rand_like(s) / self.nsamples
        # self.salience += torch.norm(inp, p=2, dim=1) ** 2 / self.nsamples

    def get_salience(self, head_dim: int, group: int = 1):
        salience = self.salience
        layer_salience = 0.0
        # salience = torch.sqrt(salience)
        # salience *= torch.norm(self.weight, p=1, dim=0)
        if self.attn:
            # s = salience.size()[0]
            salience = salience.view(group, -1).sum(dim=0)
            salience = salience.view(-1, head_dim).sum(dim=1)
            # layer_salience += salience.mean(dim=0)
            layer_salience += salience.std()
            salience = salience ** (1 / 4)
            norm_data = (salience - salience.mean()) / (salience.std() + 1e-8)
            salience = torch.sigmoid(norm_data)
        else:
            # s = salience.size()[0]
            # layer_salience += salience.mean(dim=0)
            layer_salience += salience.std()
            salience = salience ** (1 / 4)
            norm_data = (salience - salience.mean()) / (salience.std() + 1e-8)
            salience = torch.sigmoid(norm_data)
        return salience, layer_salience

    def free(self):
        self.salience = None
        torch.cuda.empty_cache()


def get_overall_gradient_importance(tensor):
    zero_mask = (tensor == 0)
    non_zero_values = tensor[~zero_mask]
    if len(non_zero_values) == 0:
        return tensor
    mean_value = non_zero_values.mean()
    tensor[zero_mask] = mean_value
    # norm_data = (tensor - tensor.mean()) / (tensor.std() + 1e-8)
    # tensor = torch.sigmoid(norm_data)
    return tensor


def estimate_masked_grad_norms(weight, grad, mask, k):
    C = weight.size(0)  # 通道总数
    weight_var = torch.var(weight, dim=1, unbiased=False)  # 形状 [C]
    true_grad_norms = grad

    # 2. 获取保留通道的索引
    retained_indices = torch.where(~mask)[0]  # 未被掩码的通道索引

    # 3. 初始化结果向量
    result = torch.zeros_like(true_grad_norms)

    # 4. 对于掩码通道，使用估计值
    for i in torch.where(mask)[0]:
        # 获取当前掩码通道的权重方差
        ref_var = weight_var[i]

        # 计算保留通道与当前通道的方差差异
        retained_vars = weight_var[retained_indices]
        diff = torch.abs(retained_vars - ref_var)

        # 选择最相似的k%通道
        num_select = max(1, int(round(k / 100 * len(retained_indices))))
        sorted_indices = torch.argsort(diff)
        selected_indices = retained_indices[sorted_indices[:num_select]]

        # 计算选中通道的梯度范数平均值
        selected_grad_norms = true_grad_norms[selected_indices]
        estimated_norm = torch.mean(selected_grad_norms)

        result[i] = estimated_norm

    # 5. 对于保留通道，使用真实值
    result[~mask] = true_grad_norms[~mask]

    return result


def generate_linear_sequence(n, k):
    d = torch.sqrt(torch.tensor(12 * k / (n ** 2 - 1)))
    indices = torch.arange(n, dtype=torch.float32)
    a = -d * (n - 1) / 2
    sequence = a + indices * d
    # sequence = torch.flip(sequence, dims=[0])
    return sequence
