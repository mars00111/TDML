import random
import pickle as pk

import torch
import torch.distributed.autograd as dist_autograd
from torch import nn
from torch.distributed import rpc
from torch.distributed.rpc import RRef
from chain.chain_utils import blockchain_request

from chain.rest_objects import NodeType
from ciphers import *
from file_access import *


def create_distributed_layers_trainers(model, num_trainers):
    overall_layers = list(model.children())

    num_layers = len(overall_layers)
    if num_trainers > num_layers:
        raise ValueError(
            "Number of trainers exceeds the number of layers in the model."
        )

    layers_per_trainer = num_layers // num_trainers
    remainder = num_layers % num_trainers

    layers_trainers = []
    start = 0

    for i in range(num_trainers):
        stop = start + layers_per_trainer + (1 if i < remainder else 0)
        layers_trainers.append(overall_layers[start:stop])
        start = stop

    assert len(layers_trainers) == num_trainers
    assert sum(len(layers) for layers in layers_trainers) == num_layers

    return layers_trainers


class Ref_Model(nn.Module):
    def __init__(self, name, model, device):
        super(Ref_Model, self).__init__()

        self.name = name
        self.device = device
        self.model = model.to(device)
        self.num_classes = model.fc.out_features
        self.model.fc = self.model.fc.to(device)

    def forward(self, x_rref):
        x = x_rref.to_here().to(self.device)
        out = self.model(x)
        return out.cpu()

    def parameter_rrefs(self):
        return [RRef(p) for p in self.model.parameters()]

    def set_state_dicts(self, state_dict_rref):
        state_dict = state_dict_rref.to_here()
        self.model.load_state_dict(state_dict)

    def get_state_dicts(self):
        state_dict = {k: v.cpu() for k, v in self.model.state_dict().items()}
        return RRef(state_dict)

    def get_dist_gradients(self, context_id):
        grads = dist_autograd.get_gradients(context_id)
        cpu_grads = {k.to("cpu"): v.to("cpu") for k, v in grads.items()}
        return RRef(cpu_grads)


class DistModel(nn.Module):
    def __init__(self, layers_trainers, ngpus, num_classes, **kwargs):
        super(DistModel, self).__init__()

        self.trainer_names = kwargs.get(
            "trainer_names", [f"trainer_{i+1}" for i in range(len(layers_trainers))]
        )
        self.trainer_gpus = kwargs.get(
            "trainer_gpus",
            [random.randint(0, ngpus - 1) for _ in range(len(layers_trainers))],
        )
        self.use_ddp = kwargs.get("use_ddp", False)

        self.trainer_rrefs = []
        for i, layers in enumerate(layers_trainers):
            trainer_name = self.trainer_names[i]
            device = f"cuda:{self.trainer_gpus[i]}"
            output_classes = num_classes if i == (len(layers_trainers) - 1) else 0

            trainer_rref = rpc.remote(
                trainer_name,
                Ref_Model,
                args=(trainer_name, layers, device, output_classes, self.use_ddp),
            )
            self.trainer_rrefs.append(trainer_rref)

    def forward(self, x):
        x_rref = RRef(x)

        for trainer_rref in self.trainer_rrefs:
            x_rref = trainer_rref.remote().forward(x_rref)

        return torch.cat([x_rref.to_here()])

    def parameter_rrefs(self):
        remote_params = []
        for trainer_rref in self.trainer_rrefs:
            remote_params.extend(trainer_rref.remote().parameter_rrefs().to_here())
        return remote_params

    def get_state_dict_rrefs(self):
        return [
            trainer_rref.remote().get_state_dicts().to_here()
            for trainer_rref in self.trainer_rrefs
        ]

    def set_state_dict_rrefs(self, parameters):
        assert len(parameters) == len(
            self.trainer_rrefs
        ), "Parameter count mismatch with trainers"

        for trainer_rref, param in zip(self.trainer_rrefs, parameters):
            trainer_rref.remote().set_state_dicts(RRef(param)).to_here()


class ChainDistModel(DistModel):
    def __init__(self, layers_trainers, num_classes, **kwargs):
        assert "trainer_names" in kwargs
        assert "trainer_gpus" in kwargs

        super(ChainDistModel, self).__init__(layers_trainers, 0, num_classes, **kwargs)
        self.chain_url = None
        self.cipher = None
        self.ipfs_client = get_ipfs_instance()

    def get_trainer_gradients(self, context_id):
        trainer_grads = {}
        for i, w_rref in enumerate(self.trainer_rref):
            cpu_grads = w_rref.remote().get_dist_gradients(context_id).to_here()
            w_name = self.trainer_names[i]
            trainer_grads[w_name] = cpu_grads

        return trainer_grads

    def log_epoch_to_blockchain(self, epoch, trainer_name, trainer_params):
        if not self.chain_url:
            return

        trainer_params_enc = self.cipher.encrypt_data(trainer_params)
        f_name = f"enc_params_{trainer_name}_epoch_{epoch}.pt"
        with open(f_name, "wb") as f:
            pk.dump(trainer_params_enc, f)

        c_id = self.ipfs_client.upload(f_name)

        epoch_info = {
            "epoch_n": epoch,
            "client_id": trainer_name,
            "client_type": NodeType.Trainer,
            "parameter_url": c_id,
        }

        blockchain_request(self.chain_url, "validation", epoch_info)

    def epoch_done(self, epoch):
        trainers_states_dict = self.get_state_dict_rrefs().to_here(timeout=0)
        for trainer_name, trainer_params in zip(
            self.trainer_names, trainers_states_dict
        ):
            self.log_epoch_to_blockchain(epoch, trainer_name, trainer_params)
