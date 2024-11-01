import os
import time
import requests
import random
import socket
import numpy as np
import pickle as pk
import uuid

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
from malicious import *


def enable_remoters(rank, num_trainers, num_dataparal, world_size, master_info, rpc_module=rpc):
    options = rpc_module.TensorPipeRpcBackendOptions(
        rpc_timeout=5000, _transports=["uv"]
    )

    if rank == 0:
        initialize_master_node(
            rank, num_trainers, num_dataparal, world_size, options, rpc_module
        )
    elif rank <= num_dataparal:
        initialize_parameter_server(rank, world_size, options, rpc_module)
    else:
        initialize_trainer_node(
            rank, num_trainers, num_dataparal, world_size, options, rpc_module
        )


def initialize_master_node(
    rank, num_trainers, num_dataparal, world_size, options, rpc_module
):
    rpc_module.init_rpc(
        "master", rank=rank, world_size=world_size, rpc_backend_options=options
    )
    run_master(num_trainers, num_dataparal)


def initialize_parameter_server(rank, world_size, options, rpc_module):
    current_job = fetch_current_job()
    if current_job:
        ps_name = f"dist_ps_{rank}"
        ps_info = {"name": ps_name, "root": "master", "job_id": current_job["id"]}
        blockchain_request(public_chain_root_url, "ps/registration", ps_info)
        rpc_module.init_rpc(
            ps_name, rank=rank, world_size=world_size, rpc_backend_options=options
        )


def initialize_trainer_node(
    rank, num_trainers, num_dataparal, world_size, options, rpc_module
):
    ngpus = torch.cuda.device_count()
    l_gpus = [random.randint(0, ngpus - 1) for _ in range(num_trainers * num_dataparal)]
    current_job = fetch_current_job()

    if current_job:
        gpu_id = l_gpus[rank - 1 - num_dataparal]
        device_properties = torch.cuda.get_device_properties(gpu_id)
        compute_capability = f"{device_properties.major}.{device_properties.minor}"

        node_name = f"{socket.gethostname()}_{rank}"
        trainer_info = {
            "name": node_name,
            "root": "master",
            "gpu_info": {"gpu_id": gpu_id, "score": float(compute_capability)},
        }

        blockchain_request(public_chain_root_url, "trainer/registration", trainer_info)
        rpc_module.init_rpc(
            node_name, rank=rank, world_size=world_size, rpc_backend_options=options
        )


def fetch_current_job(max_trials=10, interval=3):
    for _ in range(max_trials):
        response = requests.get(f"{public_chain_root_url}/curr_job")
        job_data = response.json().get("job")
        if job_data and "rewards" in job_data:
            return job_data
        time.sleep(interval)

    print("Failed to retrieve a valid job after multiple attempts.")
    return None


def run_master(num_trainers, num_dataparal, dataset="cifar10"):
    cipher_obj = get_ciphers_instance()
    private_chain_root_url_encried = cipher_obj.encrypt_data(private_chain_root_url)
    ipfs_client = get_ipfs_instance()

    job_info = {
        "publisher": "master",
        "id": uuid.uuid1(),
        "name": "job_training_1",
        "description": "job_training_1 xxx",
        "rewards": "rewards xxx",
    }
    blockchain_request(public_chain_root_url, "publish_job", job_info)

    f_output, f_checkpoints, num_classes = initialize_output_paths(
        dataset, num_dataparal, num_trainers
    )
    train_subloaders, _, testloader, _, _ = prepare_datasets(dataset, num_dataparal)
    train_cids = []
    for i, t_loader in enumerate(train_subloaders):
        enc_data = cipher_obj.encrypt_dataset(t_loader.dataset)
        f_ = f"train_data_paral_{i}.pk"
        with open(f_, "wb") as f:
            pk.dump(enc_data, f)
        train_cids.append(ipfs_client.upload(f_))

    enc_data = cipher_obj.encrypt_dataset(testloader.dataset)
    f_ = f"test_data.pk"
    with open(f_, "wb") as f:
        pk.dump(enc_data, f)
    test_cid = ipfs_client.upload(f_)

    master_info = {"root": "master", "url": MASTER_ADDR, "port": MASTER_PORT}
    blockchain_request(private_chain_root_url, "init_connect", master_info)

    registered_PSs = fetch_registered_PSs(num_dataparal, n_trials=10)

    n_req_trainers = num_trainers * num_dataparal
    registered_trainers = fetch_registered_trainers(n_req_trainers, n_trials=10)

    blockchain_request(private_chain_root_url, "init_training", registered_trainers)

    ref_dist_PSs, trainers_names, PSs_name = setup_ps(
        num_trainers,
        registered_PSs,
        registered_trainers,
        num_classes,
        cipher_obj,
        private_chain_root_url_encried,
    )

    start = 0
    test_losses, test_accs, train_losses = [], [], []
    # start, test_losses, test_accs, train_losses = resume_from_previous_epoch(
    #     PSs_name, ref_dist_PSs, trainers_names, f_output
    # )

    for epoch in range(start, EPOCHS):
        train_losses = execute_training(
            epoch, ref_dist_PSs, train_cids, train_losses, PSs_name, test_cid
        )

        ps_accuracies = eval_ps(epoch, ref_dist_PSs, test_cid)
        ps_outliers = find_outliers_score(ps_accuracies)

        malicious_ps, malicious_trainers = eval_weights(epoch, ps_outliers)
        black_list = malicious_ps + malicious_trainers

        if len(black_list) > 0:
            ref_dist_PSs, trainers_names, PSs_name = blacklist_process(
                black_list,
                trainers_names,
                malicious_ps,
                malicious_trainers,
                registered_trainers,
                registered_PSs,
                num_trainers,
                num_classes,
                cipher_obj,
                private_chain_root_url_encried,
            )

        updated_state_dict = aggregate_weights(ref_dist_PSs, ps_outliers)

        for ref_dist_ps in ref_dist_PSs:
            ref_dist_ps.rpc_sync().set_state_dict_rrefs(updated_state_dict)

        test_loss, accuracy = ref_dist_PSs[0].rpc_sync(timeout=0).eval(test_cid)
        test_losses.append(test_loss)
        test_accs.append(accuracy)

        save_epoch_results(
            f_output,
            epoch,
            test_losses,
            test_accs,
            train_losses,
            updated_state_dict,
            f_checkpoints,
        )


def initialize_output_paths(
    dataset, num_dataparal, num_clients, gth_classes, output_root="."
):
    f_output = f"{output_root}/{dataset}_outputs/dist_ps_{num_dataparal}/{num_clients}"
    f_checkpoints = f"{f_output}/checkpoints"
    os.makedirs(f_checkpoints, exist_ok=True)
    num_classes = len(gth_classes)
    return f_output, f_checkpoints, num_classes


def execute_training(
    epoch, rref_dist_PSs, train_cids, train_losses, ps_names, test_cid
):
    futs = {}
    for i, rref_dist_ps in enumerate(rref_dist_PSs):
        c_id = train_cids[i]
        h = rpc.rpc_async(
            rref_dist_ps.owner(),
            rref_dist_ps.rpc_sync(timeout=0).run,
            args=(c_id, epoch),
            timeout=0,
        )
        futs[ps_names[i]] = h

    for ps_name, fut in futs.items():
        fut.wait()
        train_losses[ps_name].append(fut.value())

        target_ps, target_name = None, None
        for rref_ps in rref_dist_PSs:
            _name = rref_ps.remote().name.to_here(timeout=0)
            if _name == ps_name:
                target_ps = rref_ps
                target_name = _name
                break

        target_state_dict = target_ps.remote().get_state_dict_rrefs().to_here(timeout=0)
        params_enc = get_ciphers_instance().encrypt_data(target_state_dict)
        f_name = f"enc_params_{target_name}_epoch_{epoch}.pk"
        with open(f_name, "wb") as f:
            pk.dump(params_enc, f)

        c_id = get_ipfs_instance().upload(f_name)

        epoch_val_info = {
            "epoch_n": epoch,
            "client_id": target_name,
            "client_type": NodeType.PS,
            "parameter_url": c_id,
        }

        blockchain_request(private_chain_root_url, "validation", epoch_val_info)

    return train_losses


def aggregate_weights(rref_dist_PSs, ps_outliers):
    if len(ps_outliers) > 0:
        trainers_states_dict = [
            rref_dist_ps.remote().get_state_dict_rrefs().to_here(timeout=0)
            for rref_dist_ps in rref_dist_PSs
            if rref_dist_ps.name not in ps_outliers
        ]
    else:
        trainers_states_dict = [
            rref_dist_ps.remote().get_state_dict_rrefs().to_here(timeout=0)
            for rref_dist_ps in rref_dist_PSs
        ]

    updated_trainers_state_dict = {}

    for trainer_params in trainers_states_dict:
        for trainer_name, trainer_state_dict in enumerate(trainer_params):
            updated_trainers_state_dict.setdefault(trainer_name, []).append(
                trainer_state_dict
            )

    updated_state_dict = []
    for _, trainers_state_dict in updated_trainers_state_dict.items():
        trainers_state_dict = [w.to_here() for w in trainers_state_dict]
        keys = {k for w in trainers_state_dict for k in w.keys()}
        update_trainer_params = {
            k: torch.stack([w[k] for w in trainers_state_dict]).mean(dim=0)
            for k in keys
        }
        updated_state_dict.append(update_trainer_params)

    return updated_state_dict


def save_epoch_results(
    f_output,
    epoch,
    test_losses,
    test_accs,
    train_losses,
    updated_state_dict,
    f_checkpoints,
):
    with open(f"{f_output}/test_loss_acc.pk", "wb") as f:
        pk.dump([test_losses, test_accs], f)
    with open(f"{f_output}/train_loss.pk", "wb") as f:
        pk.dump(train_losses, f)
    log_epoch_to_blockchain(epoch, updated_state_dict, f_checkpoints)


def log_epoch_to_blockchain(epoch, updated_state_dict, f_checkpoints):
    f_param = f"{f_checkpoints}/global_parameters_{epoch}.pk"
    with open(f_param, "wb") as f:
        pk.dump(get_ciphers_instance().encrypt_data(updated_state_dict), f)
    c_id = get_ipfs_instance().upload(f_param)

    epoch_info = {
        "epoch_n": epoch,
        "parameter_url": c_id,
    }
    blockchain_request(private_chain_root_url, "epoch", epoch_info)
