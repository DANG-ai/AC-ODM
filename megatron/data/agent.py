import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
import deepspeed
import random
from typing import List, Optional, Dict, Iterator
from collections import OrderedDict
import os
from megatron.utils import reduce_losses
import copy
from megatron.learning_rates import AnnealingLR



def mean_std_norm(tensor):
    mean = tensor.mean()
    std = tensor.std()
    tensor = (tensor - mean) / (std + 1e-8)
    return torch.clamp(tensor, -3, 3)


def get_model_weights(model, module_list):
    weights_list = []
    for name, param in model.named_parameters():
        if any(module==name for module in module_list):
            weights_list.append(param.data.clone().detach().cpu())
    return weights_list



def get_model_grad_flat(model, module_list):
    """Retrieve and flatten gradients of specified modules in the model"""
    grad_list = []
    for name, param in model.named_parameters():
        if any(module==name for module in module_list) and param.grad is not None:
            grad_list.append(param.grad.clone().detach().flatten().cpu())
    return torch.cat(grad_list) if grad_list else torch.tensor([])


def compute_domain_gradients(neox_args, timers, model, loss, domain_idx, dnum_dict=None, grad_matrix=None, prev_grad=None, return_loss=False):
    if neox_args.deepspeed:
        timers("backward-backward").start()
        # domain_loss.backward(retain_graph=True)
        model.backward(loss)
        timers("backward-backward").stop()
        timers("backward-allreduce").reset()
    else:
        raise ValueError("Must be using deepspeed to run neox")


    # Retrieve current gradients
    current_grad = get_model_grad_flat(model, neox_args.acodm["selected_weights_name"])

    # Compute the gradient difference if the previous gradient exists
    if prev_grad is not None:
        domain_grad = current_grad - prev_grad
    else:
        domain_grad = current_grad.clone()

    # Update the previous gradient
    prev_grad = current_grad.clone()

    # Store the gradient in the matrix
    if "load_path" not in neox_args.acodm.keys():
        grad_matrix[domain_idx, :] += domain_grad

        if dnum_dict is not None:
            for domain_idx, domain in enumerate(neox_args.acodm["datasets_names"]):
                if dnum_dict[domain].item() != 0:
                    grad_matrix[domain_idx, :] /= dnum_dict[domain].to("cpu")

    # return grad_matrix, prev_grad
    return grad_matrix, prev_grad



def compute_rewards(neox_args, timers, model, loss, domain_idx, prev_grad=None, dnum_dict=None, old_mat=None):
    grad_matrix, prev_grad = compute_domain_gradients(neox_args, timers, model, loss, domain_idx, prev_grad=prev_grad, dnum_dict=dnum_dict, grad_matrix=old_mat)
    if dnum_dict is None:
        return grad_matrix, prev_grad

    if "load_path" not in neox_args.acodm.keys():
        grad_matrix = grad_matrix.to(torch.cuda.current_device())
        if torch.distributed.is_initialized():
            torch.distributed.barrier()
            torch.distributed.all_reduce(grad_matrix)
        grad_matrix = grad_matrix.to("cpu")
        scores_mat_all = grad_matrix @ grad_matrix.T

        diag_elements = torch.diag(scores_mat_all)
        diag_matrix = torch.diag(diag_elements)
        scores_mat = scores_mat_all - diag_matrix

        scores = scores_mat.sum(dim=-1)
        avg_norm = grad_matrix.norm(dim=-1).mean()
        scores = scores / (avg_norm + 1e-6)
        scores = torch.clip(scores, min=neox_args.acodm["dw_min"], max=neox_args.acodm["dw_max"])
        scores = mean_std_norm(scores)
    else:
        scores = None
    return scores, None



class PolicyNet(nn.Module):
    def __init__(self, state_dim, hidden_dim, action_dim):
        super(PolicyNet, self).__init__()
        self.fc_block = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
            nn.Softmax(dim=-1)
        )

    def forward(self, x):
        xin = x[0]
        for xi in range(1, len(x)):
            xin = torch.cat((xin, x[xi]), dim=1)
        xin = xin.to(dtype=self.fc_block[0].weight.dtype)
        outs = self.fc_block(xin)
        return outs


class QValueNet(nn.Module):
    def __init__(self, state_dim, hidden_dim, action_dim, reward_dim):
        super(QValueNet, self).__init__()
        self.fc_block = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
        )

    def forward(self, x, a):
        xin = x[0]
        for xi in range(1, len(x)):
            xin = torch.cat((xin, x[xi]), dim=1)
        xin = xin.to(dtype=self.fc_block[0].weight.dtype)
        a = a.to(dtype=self.fc_block[0].weight.dtype)
        cat = torch.cat([xin, a], dim=1)
        outs = self.fc_block(cat)
        return outs



class ACODM:
    def __init__(
            self,
            neox_args
    ):
        dataset_names = neox_args.acodm["datasets_names"]
        self.update_n = neox_args.acodm["update_n"]
        self.gamma = neox_args.acodm["gamma"]
        self.tau = neox_args.acodm["tau"]
        self.selected_params_num = neox_args.acodm["selected_params_num"]
        self.dataset_names = dataset_names
        self.domain_2_idx = {s: i for i, s in enumerate(self.dataset_names)}
        self.idx_2_domain = {i: s for i, s in enumerate(self.dataset_names)}
        self.state_dim = len(dataset_names) * 3 + len(neox_args.acodm["weights_layers"]) * 2 + 1
        self.action_dim = len(dataset_names)
        self.rewards_dim = len(dataset_names)
        self.online_Q_model = QValueNet(self.state_dim, neox_args.acodm["Q_hidden_dim"], self.action_dim, self.rewards_dim)
        self.online_P_model = PolicyNet(self.state_dim, neox_args.acodm["P_hidden_dim"], self.action_dim)
        self.target_Q_model = QValueNet(self.state_dim, neox_args.acodm["Q_hidden_dim"], self.action_dim, self.rewards_dim)
        self.target_P_model = PolicyNet(self.state_dim, neox_args.acodm["P_hidden_dim"], self.action_dim)
        self.target_Q_model.load_state_dict(self.online_Q_model.state_dict())
        self.target_P_model.load_state_dict(self.online_P_model.state_dict())


        ds_config = {
            "train_batch_size": neox_args.acodm["update_n"],
            "train_micro_batch_size_per_gpu": neox_args.acodm["train_micro_batch_size_per_gpu"],
            "gradient_accumulation_steps": 1,
            "fp16": {
                "enabled": False
            },
            "optimizer": {
                "type": "Adam",
                "params": {
                  "lr": 1e-3
                }
              },
        }

        try:
            # default to apex as it's slightly faster
            from apex.optimizers import FusedAdam as Adam
        except ImportError:
            # if apex isn't installed, use deepspeed's FusedAdam
            print(
                "WARNING: APEX not installed - defaulting to deepspeed's fused adam"
            )
            from deepspeed.ops.adam import FusedAdam as Adam

        self.optimizer_Q = Adam(self.online_Q_model.parameters(), lr=1e-4, weight_decay=neox_args.weight_decay)
        self.optimizer_P = Adam(self.online_P_model.parameters(), lr=1e-4, weight_decay=neox_args.weight_decay)
        self.optimizer_target_Q = Adam(self.target_Q_model.parameters(), lr=1e-4, weight_decay=neox_args.weight_decay)
        self.optimizer_target_P = Adam(self.target_P_model.parameters(), lr=1e-4, weight_decay=neox_args.weight_decay)


        num_iters = max(1, neox_args.train_iters)
        init_step = neox_args.acodm["init_step"]
        warmup_iter = neox_args.acodm["warmup_iter"]
        lr_scheduler_Q = AnnealingLR(
            self.optimizer_Q,
            start_lr=neox_args.acodm["start_lr"],
            warmup_iter=warmup_iter,
            total_iters=num_iters,
            decay_style=neox_args.acodm["lr_decay_style"],
            last_iter=init_step,
            min_lr=neox_args.acodm["min_lr"],
        )

        lr_scheduler_P = AnnealingLR(
            self.optimizer_P,
            start_lr=neox_args.acodm["start_lr"],
            warmup_iter=warmup_iter,
            total_iters=num_iters,
            decay_style=neox_args.acodm["lr_decay_style"],
            last_iter=init_step,
            min_lr=neox_args.acodm["min_lr"],
        )

        lr_scheduler_target_Q = AnnealingLR(
            self.optimizer_target_Q,
            start_lr=neox_args.acodm["start_lr"],
            warmup_iter=warmup_iter,
            total_iters=num_iters,
            decay_style=neox_args.acodm["lr_decay_style"],
            last_iter=init_step,
            min_lr=neox_args.acodm["min_lr"],
        )




        lr_scheduler_target_P = AnnealingLR(
            self.optimizer_target_P,
            start_lr=neox_args.acodm["start_lr"],
            warmup_iter=warmup_iter,
            total_iters=num_iters,
            decay_style=neox_args.acodm["lr_decay_style"],
            last_iter=init_step,
            min_lr=neox_args.acodm["min_lr"],
        )



        self.model_engine_online_Q, self.optimizer_online_Q, _, self.lr_scheduler_Q = deepspeed.initialize(
            model=self.online_Q_model,
            model_parameters=self.online_Q_model.parameters(),
            config_params=ds_config,
            optimizer=self.optimizer_Q,
            lr_scheduler=lr_scheduler_Q,
            )
        self.model_engine_online_P, self.optimizer_online_P, _, self.lr_scheduler_P = deepspeed.initialize(
            model=self.online_P_model,
            model_parameters=self.online_P_model.parameters(),
            config_params=ds_config,
            optimizer=self.optimizer_P,
            lr_scheduler=lr_scheduler_P,
            )

        self.model_engine_target_Q, self.optimizer_target_Q, _, self.lr_scheduler_target_Q = deepspeed.initialize(
            model=self.target_Q_model,
            model_parameters=self.target_Q_model.parameters(),
            config_params=ds_config,
            optimizer=self.optimizer_target_Q,
            lr_scheduler=lr_scheduler_target_Q,
            )
        self.model_engine_target_P, self.optimizer_target_P, _, self.lr_scheduler_target_P = deepspeed.initialize(
            model=self.target_P_model,
            model_parameters=self.target_P_model.parameters(),
            config_params=ds_config,
            optimizer=self.optimizer_target_P,
            lr_scheduler=lr_scheduler_target_P,
            )

        self.cur_state = None
        self.pool = []
        self.data_weights = neox_args.train_data_weights # list
        self.dataset_map = {name: i for i, name in enumerate(dataset_names)}
        total_weights = np.sum(neox_args.train_data_weights)
        self._probabilities = {name: weight / total_weights for name, weight in zip(dataset_names, neox_args.train_data_weights)}
        self.warmup_flag = False
        self.critic_loss = None
        self.actor_loss = None
        self.micro_steps = 0
        self.micro_num_dict = {name: torch.tensor(0.0).to(torch.cuda.current_device()) for name in dataset_names}
        self.old_mat = torch.zeros((len(neox_args.acodm["datasets_names"]), neox_args.acodm["selected_params_num"]), dtype=torch.float32)
        self.domain_loss_dict = {name: torch.tensor([0.0, 0.0]).to(torch.cuda.current_device()) for name in dataset_names}
        self.prev_grad = None
        self.score_w = [0.0] * len(dataset_names)
        self.warmup_weighter = None
        self.warmup_P_loss_func = torch.nn.MSELoss()
        self.warmup_P_losses = None
        self.warmup_Q_loss_func = torch.nn.MSELoss()
        self.warmup_Q_losses = None

        self.last_domain_loss = None
        self.last_weights_list = None

        self.batchsize = neox_args.acodm["train_micro_batch_size_per_gpu"]


    def update_domain_loss(self, domain_loss, dname):
        self.domain_loss_dict[dname][0] += domain_loss
        self.domain_loss_dict[dname][1] += 1
        return

    def update_pools(self, neox_args, timers, model, loss, domain_name, weights=None):
        self.micro_num_dict[domain_name] += 1
        self.update_domain_loss(loss, domain_name)
        if (self.micro_steps + 1) % neox_args.gradient_accumulation_steps == 0:
            if torch.distributed.is_initialized():
                torch.distributed.barrier()
                for name in self.micro_num_dict.keys():
                    torch.distributed.all_reduce(self.micro_num_dict[name])
                    torch.distributed.all_reduce(self.domain_loss_dict[name])

            dnum_dict = copy.deepcopy(self.micro_num_dict)

            score_w, self.prev_grad = compute_rewards(neox_args, timers, model, loss, self.domain_2_idx[domain_name], prev_grad=self.prev_grad, dnum_dict=dnum_dict, old_mat=self.old_mat)


            domain_loss = torch.tensor([self.domain_loss_dict[dictname][0] / self.domain_loss_dict[dictname][1] for dictname in self.dataset_names]).to("cpu")
            domain_loss_norm = mean_std_norm(domain_loss)

            model_weights = get_model_weights(model, neox_args.acodm["weights_layers"])
            model_weights_l2_norm = mean_std_norm(torch.tensor([torch.norm(mv, p="fro").item() for mv in model_weights]))

            # calculate state
            domain_nums = mean_std_norm(torch.tensor([self.domain_loss_dict[dictname][1] * neox_args.train_micro_batch_size_per_gpu for dictname in self.dataset_names]).to("cpu"))
            if self.last_domain_loss is not None:
                sub_domain_loss_l2_norm = mean_std_norm(domain_loss - self.last_domain_loss)
                sub_weights_l2_norm = mean_std_norm(torch.tensor([torch.norm(mw-self.last_weights_list[mi], p="fro").item() for mi, mw in enumerate(model_weights)]))
            else:
                sub_domain_loss_l2_norm = torch.zeros_like(domain_loss_norm)
                sub_weights_l2_norm = torch.zeros_like(model_weights_l2_norm)
                self.last_domain_loss = domain_loss.clone()
                self.last_weights_list = [mv.clone() for mv in model_weights]

            state = [
                domain_loss_norm,
                sub_domain_loss_l2_norm,
                domain_nums,
                model_weights_l2_norm,
                sub_weights_l2_norm,
                torch.tensor([int((self.micro_steps + 1) // neox_args.gradient_accumulation_steps) / neox_args.train_iters * neox_args.acodm["iter_coe"]])
            ]


            if self.warmup_flag:
                if self.cur_state is not None:
                    pool_item = (
                        self.cur_state,
                        weights,
                        score_w,
                        state
                    )
                else:
                    pool_item = (
                        state,
                        weights,
                        score_w,
                        state
                    )
            else:
                if self.cur_state is not None:
                    pool_item = (
                        self.cur_state,
                        self.data_weights,
                        score_w,
                        state
                    )
                else:
                    pool_item = (
                        state,
                        self.data_weights,
                        score_w,
                        state
                    )
            if "load_path" not in neox_args.acodm.keys():
                self.score_w = score_w.tolist()



            self.pool.append(pool_item)
            if len(self.pool) > neox_args.acodm["pool_size"]:
                self.pool.pop(0)
            self.cur_state = [si.clone() for si in state]


            if self.warmup_flag:
                self.warmup_weighter.group_update(
                    int((self.micro_steps + 1) // neox_args.gradient_accumulation_steps),
                    **{"dataset_names": self.dataset_names,
                       "rewards": domain_loss.tolist()}
                )


            self.old_mat = torch.zeros((len(neox_args.acodm["datasets_names"]), neox_args.acodm["selected_params_num"]),
                                       dtype=torch.float32)
            self.domain_loss_dict = {name: torch.tensor([0.0, 0.0]).to(torch.cuda.current_device()) for name in
                                     self.dataset_names}
            self.micro_num_dict = {name: torch.tensor(0.0).to(torch.cuda.current_device()) for name in self.dataset_names}

            if self.warmup_flag and "load_path" not in neox_args.acodm.keys():
                self.warmup()

        else:
            dnum_dict = None
            self.old_mat, self.prev_grad = compute_rewards(neox_args, timers, model, loss, self.domain_2_idx[domain_name], prev_grad=self.prev_grad, dnum_dict=dnum_dict, old_mat=self.old_mat)
        self.micro_steps += 1
        return


    def warmup(self):
        self.model_engine_online_Q.train()
        self.model_engine_online_P.train()
        self.device = self.model_engine_target_Q.device


        if len(self.pool) >= self.batchsize:
            selected_items = random.sample(self.pool, self.batchsize)
        else:
            selected_items = self.pool + random.choices(self.pool, k=self.batchsize - len(self.pool))
        states = [torch.tensor([], dtype=torch.float)] * 6
        actions = torch.tensor([], dtype=torch.float)
        rewards = torch.tensor([], dtype=torch.float)
        self.device = self.model_engine_target_Q.device
        for si, selected_item in enumerate(selected_items):
            tensor_item = [torch.as_tensor(si, dtype=torch.float).clone().detach().view(1, -1) for si in
                           selected_item[0]]
            states = [torch.cat((states[si], tensor_item[si]), dim=0) for si in range(len(tensor_item))]

            tensor_item = torch.as_tensor(selected_item[1], dtype=torch.float).clone().detach()
            actions = torch.cat((actions, tensor_item.view(1, -1)), dim=0)

            tensor_item = torch.as_tensor(selected_item[2], dtype=torch.float).clone().detach()
            rewards = torch.cat((rewards, tensor_item.view(1, -1)), dim=0)


        states = [si.to(self.device) for si in states]
        domain_weights = actions.to(self.device)
        rewards = rewards.to(self.device)

        predict_weights = self.model_engine_online_P(states)
        loss_a = self.warmup_P_loss_func(domain_weights, predict_weights)
        self.model_engine_online_P.backward(loss_a)
        self.model_engine_online_Q.step()
        self.warmup_P_losses = reduce_losses([loss_a]).mean().item()


        predict_rewards = self.model_engine_online_Q(states, domain_weights)
        q_targets = rewards + self.gamma * rewards
        loss_c = self.warmup_Q_loss_func(predict_rewards, q_targets)
        self.model_engine_online_Q.backward(loss_c)
        self.model_engine_online_Q.step()
        self.warmup_Q_losses = reduce_losses([loss_c]).mean().item()
        return


    def init_target_network(self):
        self.model_engine_target_Q.load_state_dict(self.model_engine_online_Q.state_dict())
        self.model_engine_target_P.load_state_dict(self.model_engine_online_P.state_dict())
        return


    def soft_update(self, net, target_net):
        for param_target, param in zip(target_net.parameters(), net.parameters()):
            param_target.data.copy_(param_target.data * (1.0 - self.tau) + param.data * self.tau)

    def update(self):
        self.model_engine_online_Q.train()
        self.model_engine_online_P.train()
        self.model_engine_target_Q.train()
        self.model_engine_target_P.train()
        selected_items = random.sample(self.pool, self.batchsize)
        states = [torch.tensor([], dtype=torch.float)] * 6
        actions = torch.tensor([], dtype=torch.float)
        rewards = torch.tensor([], dtype=torch.float)
        next_states = [torch.tensor([], dtype=torch.float)] * 6
        self.device = self.model_engine_target_Q.device
        for si, selected_item in enumerate(selected_items):

            tensor_item = [torch.as_tensor(si, dtype=torch.float).clone().detach().view(1, -1) for si in selected_item[0]]
            states = [torch.cat((states[si], tensor_item[si]), dim=0) for si in range(len(tensor_item))]

            tensor_item = torch.as_tensor(selected_item[1], dtype=torch.float).clone().detach()
            actions = torch.cat((actions, tensor_item.view(1, -1)), dim=0)

            tensor_item = torch.as_tensor(selected_item[2], dtype=torch.float).clone().detach()
            rewards = torch.cat((rewards, tensor_item.view(1, -1)), dim=0)

            tensor_item = [torch.as_tensor(si, dtype=torch.float).clone().detach().view(1, -1) for si in
                           selected_item[3]]
            next_states = [torch.cat((next_states[si], tensor_item[si]), dim=0) for si in range(len(tensor_item))]

        states = [si.to(self.device) for si in states]
        actions = actions.to(self.device)
        rewards = rewards.to(self.device)
        next_states = [si.to(self.device) for si in next_states]
        next_q_values = self.model_engine_target_Q(next_states, self.model_engine_target_P(next_states))
        q_targets = rewards + self.gamma * next_q_values
        critic_loss = torch.mean(F.mse_loss(self.model_engine_online_Q(states, actions), q_targets))

        self.model_engine_online_Q.backward(critic_loss)
        self.model_engine_online_Q.step()
        self.critic_loss = reduce_losses([critic_loss]).mean().item()

        actor_loss = -torch.mean(self.model_engine_online_Q(states, self.model_engine_online_P(states)))
        self.model_engine_online_P.backward(actor_loss)
        self.model_engine_online_Q.step()
        self.actor_loss = reduce_losses([actor_loss]).mean().item()

        self.soft_update(self.model_engine_online_Q, self.model_engine_target_Q)
        self.soft_update(self.model_engine_online_P, self.model_engine_target_P)

    def gen_action(self, state, training=True):
        self.model_engine_online_P.eval()
        device = self.model_engine_online_P.device
        if training:
            action_ = self.model_engine_online_P([si.clone().detach().view(1, -1).to(device) for si in self.cur_state])
        else:
            action_ = self.model_engine_online_P([si.clone().detach().view(1, -1).to(device) for si in state])
        action = action_.cpu().detach().clone().numpy()[0]
        total_weights = sum(action)
        self.data_weights = list(action)
        for name in self.dataset_names:
            self._probabilities[name] = self.data_weights[self.dataset_map[name]] / total_weights
        final_a = np.array(list(self._probabilities.values())) - np.random.rand(22) / 1000
        final_a = final_a / final_a.sum()
        self.data_weights = final_a.tolist()
        return self.data_weights

    def get_weights(self):
        data_weights = torch.tensor(self.data_weights).to(torch.cuda.current_device())
        if torch.distributed.is_initialized():
            torch.distributed.barrier()
            torch.distributed.all_reduce(data_weights)
            data_weights /= torch.distributed.get_world_size()
        self.data_weights = data_weights.cpu().detach().tolist()
        return self.data_weights

    def save_ckpts(self, neox_args, iteration):
        save_dir = os.path.join(neox_args.acodm["save_path"], f"model_{iteration}")
        os.makedirs(save_dir, exist_ok=True)
        self.model_engine_online_Q.save_checkpoint(save_dir, tag="online_Q")
        self.model_engine_online_P.save_checkpoint(save_dir, tag="online_P")
        self.model_engine_target_Q.save_checkpoint(save_dir, tag="target_Q")
        self.model_engine_target_P.save_checkpoint(save_dir, tag="target_P")
        print("All models and optimizers have been saved successfully.")

    def load_ckpts(self, neox_args):
        load_dir = neox_args.acodm["load_path"]

        # Load online Q网络
        self.model_engine_online_Q.load_checkpoint(
            load_dir,
            tag="online_Q",
            load_optimizer_states=False,
            load_lr_scheduler_states=False
        )

        # Load online P网络
        self.model_engine_online_P.load_checkpoint(
            load_dir,
            tag="online_P",
            load_optimizer_states=False,
            load_lr_scheduler_states=False
        )

        # Load target Q网络
        self.model_engine_target_Q.load_checkpoint(
            load_dir,
            tag="target_Q",
            load_optimizer_states=False,
            load_lr_scheduler_states=False
        )

        # Load target P网络
        self.model_engine_target_P.load_checkpoint(
            load_dir,
            tag="target_P",
            load_optimizer_states=False,
            load_lr_scheduler_states=False
        )

        print("All models and optimizers have been loaded successfully.")



def load_ACODM(neox_args, model):
    selected_params_num = 0
    for name, param in model.named_parameters():
        if param.requires_grad and name in neox_args.acodm["selected_weights_name"]:
            selected_params_num += param.numel()
    if "selected_params_num" not in neox_args.acodm.keys():
        neox_args.acodm["selected_params_num"] = selected_params_num
    if neox_args.acodm["selected_params_num"] == 0 or neox_args.acodm["selected_params_num"] is None:
        raise NotImplementedError("Must have selected weights name for ACODM model!")
    if "datasets_names" not in neox_args.acodm.keys():
        neox_args.acodm["datasets_names"] = []
        for train_path in neox_args.train_data_paths:
            name = train_path.split("/")[-2]
            neox_args.acodm["datasets_names"].append(name)
        neox_args.acodm["datasets_names"] = list(OrderedDict.fromkeys(neox_args.acodm["datasets_names"]))
    ac_agent = ACODM(neox_args)
    if "load_path" in neox_args.acodm.keys():
        ac_agent.load_ckpts(neox_args=neox_args)
    return ac_agent, neox_args



class DDPGWeightUpdater:
    def __init__(
            self,
            dataset_names: List[str],
            weights: List[float],
    ):
        self.dataset_names = dataset_names
        self.dataset_map = {name: i for i, name in enumerate(dataset_names)}
        self.num_datasets = len(dataset_names)
        self.weights = weights
        total_weights = np.sum(weights)
        self._probabilities = {name: weight / total_weights for name, weight in zip(dataset_names, weights)}
        self.eps = 1 / self.num_datasets
        self.prev_eps = None


    def group_update(self, iteration, state, agent, training):
        return agent.gen_action(state, training)






