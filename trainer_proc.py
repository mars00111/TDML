import pickle as pk
import threading

import torch
from torch import optim
import torch.nn as nn
from torch.distributed import rpc
from torch.distributed.optim import DistributedOptimizer
from torch.distributed.autograd import context as dist_autograd
from torch.utils.data import Dataset, DataLoader

from model_proc import ChainDistModel, create_distributed_layers_trainers
from ciphers import *
from file_access import *
from configs import BATCH_SIZE
from chain.rest_objects import NodeType
from chain.chain_utils import blockchain_request


def create_distributed_model(
    ps_name,
    num_classes,
    registered_trainers,
    ml_model,
    cipher_obj,
    private_chain_root_url_encried,
    lr=0.1,
):

    trainers_name = [trainer["name"] for trainer in registered_trainers]
    trainers_gpu = [trainer["gpu_info"]["gpu_id"] for trainer in registered_trainers]
    trainers_name = sorted(trainers_name)

    loss_fn = nn.CrossEntropyLoss()
    layers_trainers = create_distributed_layers_trainers(ml_model, len(trainers_name))

    chain_model = ChainDistModel(
        layers_trainers,
        num_classes=num_classes,
        trainer_names=trainers_name,
        trainer_gpus=trainers_gpu,
    )

    opt = DistributedOptimizer(
        optim.SGD,
        chain_model.parameter_rrefs(),
        lr=lr,
    )

    ps_rref = rpc.remote(
        ps_name,
        DistParamServer,
        args=(
            chain_model,
            opt,
            loss_fn,
            ps_name,
            cipher_obj,
            private_chain_root_url_encried,
        ),
    )

    return ps_rref, trainers_name


class DistParamServer(object):
    def __init__(
        self,
        dist_model,
        optimizer,
        loss_fn,
        name,
        cipher_obj,
        private_chain_root_url_encried,
    ):
        self.dist_model = dist_model
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.name = name
        self.cipher_obj = cipher_obj

        self.private_chain_root_url = cipher_obj.decrypt_data(
            private_chain_root_url_encried
        )
        self.dist_model.chain_url = self.private_chain_root_url
        self.dist_model.cipher = cipher_obj
        self.ipfs_client = get_ipfs_instance()

        self.idle = threading.Lock()

    def set_state_dict_rrefs(self, trainer_state_dicts):
        self.dist_model.set_state_dict_rrefs(trainer_state_dicts)

    def get_state_dict_rrefs(self):
        return self.dist_model.get_state_dict_rrefs()

    def get_dist_gradients(self, context_id):
        return self.dist_model.get_trainer_gradients(context_id)

    def get_trainer_names(self):
        return self.dist_model.trainer_names

    def load_state_dict(self, paths):
        trainers_state_dicts = []
        for path in paths:
            with open(path, "rb") as file:
                trainers_state_dicts.append(pk.load(file))
        assert len(trainers_state_dicts) == len(
            self.get_trainer_names()
        ), "Mismatch in model states and trainers."
        self.set_state_dict_rrefs(trainers_state_dicts)

    def run(self, train_cid, epoch):
        self.idle.acquire(timeout=2)

        self.dist_model.train()
        losses = []
        running_loss = 0.0

        f_trainloader = f"./{self.name}_trainloader.pk"
        self.ipfs_client.download(train_cid, f_trainloader)
        with open(f_trainloader, "wb") as f:
            enc_data = pk.load(f_trainloader)
            dataset = self.cipher_obj.decrypt_dataset(enc_data)

        trainloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

        for i, (inputs, labels) in enumerate(trainloader):
            with dist_autograd.context() as context_id:
                outputs = self.dist_model(inputs)
                loss = self.loss_fn(outputs, labels)
                dist_autograd.backward(context_id, [loss])
                self.optimizer.step(context_id)

                losses.append(loss.item())
                running_loss += loss.item()

                if i % 10 == 0 and i > 0:
                    print(
                        f"Epoch [{epoch}, Batch {i}, {self.name}] Avg Loss: {running_loss / 10:.4f}"
                    )
                    running_loss = 0.0

        avg_loss = sum(losses) / len(losses)
        print(f"Epoch [{epoch}, {self.name}] completed. Avg Loss: {avg_loss:.4f}")

        self.dist_model.epoch_done(epoch)

        params_enc = self.cipher.encrypt_data(self.get_state_dict_rrefs())
        f_name = f"enc_params_{self.name}_epoch_{epoch}.pk"
        with open(f_name, "wb") as f:
            pk.dump(params_enc, f)
        c_id = self.ipfs_client.upload(f_name)

        self.idle.release()
        return avg_loss

    def eval(self, test_cid):
        self.idle.acquire(timeout=2)

        f_testloader = f"./{self.name}_testloader.pk"
        self.ipfs_client.download(test_cid, f_testloader)
        with open(f_testloader, "wb") as f:
            enc_data = pk.load(f_testloader)
            dataset = self.cipher_obj.decrypt_dataset(enc_data)

        testloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

        correct, total, loss = 0, 0, 0.0
        self.dist_model.eval()

        with torch.no_grad():
            for inputs, labels in testloader:
                outputs = self.dist_model(inputs)
                loss += self.loss_fn(outputs, labels).item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        avg_loss = loss / len(testloader)
        accuracy = correct / total
        print(
            f"Evaluation on {self.name} - Loss: {avg_loss:.4f}, Accuracy: {accuracy * 100:.2f}%"
        )

        self.idle.release()
        return avg_loss, accuracy

    def idle_status(self):
        return not self.idle.locked()
