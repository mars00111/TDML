import os
import json
import numpy as np
import matplotlib.pyplot as plt
import pickle as pk
import requests
import time
import random

import torch
from torch.utils.data import random_split
import torch.distributed.rpc as rpc
import torchvision.models as torch_models

from chain.chain_utils import blockchain_request
from dataset import *
from configs import *
from utils import *
from ciphers import get_ciphers_instance
from trainer_proc import create_distributed_model
from file_access import get_ipfs_instance
from chain.rest_objects import NodeType


def gen_color():
    r = np.round(np.random.rand(), 1)
    g = np.round(np.random.rand(), 1)
    b = np.round(np.random.rand(), 1)
    return [r, g, b]


def plot_results(
    train_losses, test_accs, num_dataparal, num_clients, dataset, f_output
):
    plt_title = f"Distributed (Data-Model) parallelism training loss. {num_dataparal}x{num_clients} clients {dataset}"
    fig, ax = plt.subplots()
    for trainer_name, losses in train_losses.items():
        ax.plot(
            np.arange(1, EPOCHS + 1),
            np.array(losses[:EPOCHS]),
            color=gen_color(),
            label=f"Data pipeline {trainer_name} loss",
        )

    ax.set(xlabel="Epoch", ylabel="Train Loss", title=plt_title)
    ax.grid()
    fig.savefig(
        f"{f_output}/DM_paral_D{num_dataparal}_C{num_clients}_loss.jpg", format="jpg"
    )

    plt_title = f"Distributed (Data-Model) parallelism test accuracy. {num_dataparal}x{num_clients} clients {dataset}"
    fig, ax = plt.subplots()
    ax.plot(
        np.arange(1, EPOCHS + 1),
        np.array(test_accs[:EPOCHS]),
        "tab:blue",
        label="test_accuracy",
    )
    ax.set(xlabel="Epoch", ylabel="Test Accuracy", title=plt_title)
    ax.grid()
    fig.savefig(
        f"{f_output}/DM_paral_D{num_dataparal}_C{num_clients}_acc.jpg", format="jpg"
    )


def resume_from_previous_epoch(PSs_names, dist_PS_refs, trainers_names, f_output):
    trainer_states = blockchain_request(
        private_chain_root_url, "parameters", trainers_names
    )
    resume_epoch = trainer_states["epoch_n"]

    start_epoch = 0
    test_losses, test_accuracies = [], []
    train_losses = {name: [] for name in PSs_names}

    if resume_epoch != 0:
        state_dicts = [trainer_states[c_id] for c_id in trainers_names]

        for rref in dist_PS_refs:
            rref.rpc_sync().load_state_dict(state_dicts)

        start_epoch = resume_epoch + 1
        if resume_epoch < EPOCHS:
            with open(f"{f_output}/test_loss_acc.pk", "rb") as f:
                test_losses, test_accuracies = pk.load(f)

            with open(f"{f_output}/train_loss.pk", "rb") as f:
                train_losses = pk.load(f)

    return start_epoch, test_losses, test_accuracies, train_losses


def fetch_registered_trainers(num_trainers, n_trials, blacklist=[]):
    while n_trials > 0:
        response = requests.get(f"{public_chain_root_url}/trainers")
        ret = []

        registered_trainers = response.json()["trainers"]
        if len(registered_trainers) >= num_trainers * 2:
            registered_trainers = random.choices(registered_trainers, num_trainers)

            for trainer in registered_trainers:
                if trainer["name"] not in blacklist:
                    ret.append(trainer)

            return ret
        n_trials -= 1
        time.sleep(3)
    raise ValueError("Unable to obtain enough trainers.")


def fetch_registered_PSs(num_dataparal, n_trials, blacklist=[]):
    while n_trials > 0:
        response = requests.get(f"{public_chain_root_url}/PSs")
        ret = []

        registered_PSs = response.json()["PSs"]
        if len(registered_PSs) >= num_dataparal * 2:
            registered_PSs = random.choices(registered_PSs, num_dataparal)

            for ps in registered_PSs:
                if ps["name"] not in blacklist:
                    ret.append(ps)

            return ret

        n_trials -= 1
        time.sleep(3)
    raise ValueError("Unable to obtain enough PSs.")


def prepare_datasets(dataset, num_dataparal, batch_size=BATCH_SIZE):
    trainset, trainloader, testset, testloader, grayscale = dataset_from_name(
        dataset, batch_size
    )
    lengths = [1 / num_dataparal] * num_dataparal
    train_subsets = random_split(trainset, lengths)
    train_subloaders = [
        torch.utils.data.DataLoader(ds, batch_size=batch_size, shuffle=True)
        for ds in train_subsets
    ]
    gth_classes = np.array(list(trainset.class_to_idx.values()))

    return train_subloaders, testset, testloader, gth_classes, grayscale


def setup_ps(
    num_trainers,
    registered_PSs,
    registered_trainers,
    num_classes,
    cipher_obj,
    private_chain_root_url_encried,
):
    rref_dist_PSs, PSs_name, trainers_names = [], [], {}
    c_start = 0
    for ps in range(registered_PSs):
        c_stop = c_start + num_trainers
        name = ps["name"]
        ml_model = torch_models.resnet50()
        ml_model.fc = torch.nn.Linear(ml_model.fc.in_features, num_classes)

        sel_trainers = registered_trainers[c_start:c_stop]
        sel_trainers_names = [t["name"] for t in sel_trainers]

        rref_dist_ps, trainers_name_part = create_distributed_model(
            name,
            num_classes,
            sel_trainers,
            ml_model,
            cipher_obj,
            private_chain_root_url_encried,
        )

        rref_dist_PSs.append(rref_dist_ps)
        trainers_names[name] = trainers_name_part
        PSs_name.append(name)
        c_start = c_stop

        ps_trainers = {"name": name, "trainers": sel_trainers_names}

        blockchain_request(private_chain_root_url, "/ps/trainers", ps_trainers)
    return rref_dist_PSs, trainers_names, PSs_name


# def fetch_master_connection_info(node_name):
#     response = blockchain_request(
#         "trainer/transactions", {"trainer_id": node_name}, param=True
#     )
#     transactions = response.get("transactions", [])

#     for transaction in transactions:
#         if transaction.get("recipient") == "init_connect":
#             master_url = transaction["data"]["url"]
#             master_port = transaction["data"]["port"]
#             return master_url, master_port

#     raise ValueError("Master connection info not found.")