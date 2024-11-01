import random
import time
import pickle as pk
import numpy as np
import torch
import requests

from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

from chain.rest_objects import NodeType
from chain.chain_utils import blockchain_request
from configs import *
from file_access import *
from ciphers import *
from utils import *

def normalize(v):
    norm=np.linalg.norm(v)
    if norm==0:
        norm=np.finfo(v.dtype).eps
    return v/norm

def find_outliers_score(ps_accs):
    accs = list(ps_accs.values())
    ps_names = list(ps_accs.keys())

    mean = np.mean(accs)
    std_dev = np.std(accs)
    z_scores = [(x - mean) / std_dev for x in accs]
    outliers = [i for i in range(len(accs)) if abs(z_scores[i]) > 0.3]
    outliers = [ps_names[i] for i in outliers]

    return outliers


def eval_ps(epoch, rref_dist_PSs, test_cid):
    val_ps = random.choice(rref_dist_PSs)
    while not val_ps.idle_status():
        val_ps = random.choice(rref_dist_PSs)
        time.sleep(3)

    saved_state_dict = val_ps.remote().get_state_dict_rrefs().to_here(timeout=0)

    accuracy_list = blockchain_request(private_chain_root_url, "accuracy_list", epoch)
    done_val_ids = [acc["client_id"] for acc in accuracy_list]

    accuracies = {}
    validation_list = blockchain_request(
        private_chain_root_url, "validation_list", epoch
    )
    for val_obj in validation_list:
        if val_obj["client_type"] != NodeType.PS:
            continue

        ps_name = val_obj["client_id"]
        param_cid = val_obj["parameter_url"]

        if ps_name in done_val_ids:
            continue

        f_name = f"enc_params_{ps_name}_epoch_{epoch}.pk"
        get_ipfs_instance().download(param_cid, f_name)
        with open(f_name, "rb") as f:
            params_enc = pk.load(f)
        param_dict = get_ciphers_instance().decrypt_data(params_enc)

        val_ps.rpc_sync().set_state_dict_rrefs(param_dict)
        test_loss, accuracy = val_ps.rpc_sync(timeout=0).eval(test_cid)

        val_accuracy = {"epoch_n": epoch, "client_id": ps_name, "accuracy": accuracy}
        blockchain_request(private_chain_root_url, "accuracy", val_accuracy)
        accuracies[ps_name] = accuracy

    val_ps.rpc_sync().set_state_dict_rrefs(saved_state_dict)
    return accuracies


def eval_weights(epoch, ps_outliers):
    validation_list = blockchain_request(
        private_chain_root_url, "validation_list", epoch
    )
    PSs_trainers, ps_states_dicts, trainers_states_dicts = {}, {}, {}

    for validator in validation_list:
        ps_name = validator["client_id"]

        if ps_name in ps_outliers:
            param_cid = validator["parameter_url"]
            file_name = f"enc_params_{ps_name}_epoch_{epoch}.pk"

            get_ipfs_instance().download(param_cid, file_name)
            with open(file_name, "rb") as f:
                encrypted_params = pk.load(f)

            ps_states_dicts[ps_name] = get_ciphers_instance().decrypt_data(
                encrypted_params
            )

    for ps_name in ps_outliers:
        response = blockchain_request(
            private_chain_root_url, "/ps/trainers_list", ps_name
        )
        PSs_trainers[ps_name] = response["trainers"]

    for ps_name, trainers in PSs_trainers.items():
        for trainer_id in trainers:
            for validator in validation_list:

                if validator["client_type"] != NodeType.Trainer:
                    continue

                if trainer_id == validator["client_id"]:
                    t_param_cid = validator["parameter_url"]
                    file_name = f"enc_params_{trainer_id}_epoch_{epoch}.pk"

                    get_ipfs_instance().download(t_param_cid, file_name)
                    with open(file_name, "rb") as f:
                        encrypted_params = pk.load(f)

                    trainers_states_dicts[trainer_id] = (
                        get_ciphers_instance().decrypt_data(encrypted_params)
                    )

    malicious_ps = []
    benign_ps = []
    for ps_name, trainers in PSs_trainers.items():
        ps_states = ps_states_dicts[ps_name]
        trainers_states = [trainers_states_dicts[t] for t in trainers]

        assert len(ps_states) == len(trainers_states)

        if not compare_state_dicts(ps_states, trainers_states):
            malicious_ps.append(ps_name)
        else:
            benign_ps.append(ps_name)

    suspicious_ps_trainers = {
        ps_name: trainers for ps_name, trainers in PSs_trainers if ps_name in benign_ps
    }
    malicious_trainers = detect_trainers(
        validation_list, suspicious_ps_trainers, ps_outliers
    )

    return malicious_ps, malicious_trainers


def compare_state_dicts(list1, list2):
    if len(list1) != len(list2):
        return False

    for state_dict1, state_dict2 in zip(list1, list2):
        if state_dict1.keys() != state_dict2.keys():
            return False

        for key in state_dict1:
            if not torch.equal(state_dict1[key], state_dict2[key]):
                return False

    return True


def detect_trainers(validation_list, suspicious_ps_trainers, ps_outliers):
    pca_projector = PCA(n_components=2)
    kmeans = KMeans(n_clusters=2, n_init="auto")

    benign_trainers_weights = []
    for validator in validation_list:
        ps_name = validator["client_id"]
        if ps_name in ps_outliers:
            continue

        param_cid = validator["parameter_url"]
        file_name = f"enc_params_{ps_name}_eval_trainers.pk"

        get_ipfs_instance().download(param_cid, file_name)
        with open(file_name, "rb") as f:
            encrypted_params = pk.load(f)

        trainers_weights = get_ciphers_instance().decrypt_data(
            encrypted_params
        )

        if len(benign_trainers_weights) == 0:
            benign_trainers_weights = [[] for _ in range(trainers_weights)]

        for i, state_dict in enumerate(trainers_weights):
            tmp_weights = []
            for k, v in state_dict.items():
                if "conv" in k:
                    v = torch.flatten(v)
                    v = normalize(v) / v[np.argmax(v)]  
                    tmp_weights.append(v)
            benign_trainers_weights[i].append(torch.hstack(tmp_weights))

    benign_trainers_weights = [torch.vstack(v) for v in benign_trainers_weights]

    malicious_trainers = []
    for ps_name, trainers in suspicious_ps_trainers.items():
        for validator in validation_list:
            if ps_name != validator["client_id"]:
                continue

            param_cid = validator["parameter_url"]
            file_name = f"enc_params_{ps_name}_eval_trainers.pk"
            get_ipfs_instance().download(param_cid, file_name)
            with open(file_name, "rb") as f:
                encrypted_params = pk.load(f)
            trainers_weights = get_ciphers_instance().decrypt_data(
                encrypted_params
            )

            for i, (trainer_name, state_dict) in enumerate(zip(trainers, trainers_weights)):
                tmp_weights = []
                for k, v in state_dict.items():
                    if "conv" in k:
                        v = torch.flatten(v)
                        v = normalize(v) / v[np.argmax(v)]  
                        tmp_weights.append(v)
                
                tmp_weights = torch.hstack(tmp_weights)
                mix_weights = torch.vstack([benign_trainers_weights[i], tmp_weights]).numpy()
                pca_projector.fit(mix_weights)
                mix_weights = pca_projector.transform(mix_weights)
                kmean_labels = kmeans.fit(mix_weights).predict(mix_weights)

                unique, counts = np.unique(kmean_labels, return_counts=True)
                max_count_index = np.argmax(counts)
                max_label = unique[max_count_index]
                if kmean_labels[-1] != max_label:
                    malicious_trainers.append(trainer_name)

            break

    return malicious_trainers


def update_blacklist(black_list):
    if black_list:
        return blockchain_request(private_chain_root_url, "blacklist", black_list)
    return black_list


def gather_all_trainers(trainers_names):
    all_trainers_names = []
    for _, t_names in trainers_names.items():
        all_trainers_names.extend(t_names)
    return all_trainers_names


def replace_registered_trainers(num_train, previsou_trainerss, blacklist=[]):
    response = requests.get(f"{public_chain_root_url}/trainers")
    ret = []

    registered_trainers = response.json()["trainers"]
    for trainer in registered_trainers:
        trainer_name = trainer["name"]
        if trainer_name not in previsou_trainerss and trainer_name not in blacklist:
            ret.append(trainer)

    if len(ret) >= num_train:
        return random.choices(ret, num_train)
    else:
        raise ValueError("Unable to obtain enough trainers.")


def replace_registered_PSs(num_ps, previsou_PSs, blacklist=[]):
    response = requests.get(f"{public_chain_root_url}/PSs")
    ret = []

    registered_PSs = response.json()["PSs"]
    for ps in registered_PSs:
        ps_name = ps["name"]
        if ps_name not in previsou_PSs and ps_name not in blacklist:
            ret.append(ps)

    if len(ret) >= num_ps:
        return random.choices(ret, num_ps)
    else:
        raise ValueError("Unable to obtain enough PSs.")


def handle_malicious_ps(
    malicious_ps,
    black_list,
    trainers_names,
    registered_trainers,
    all_trainers_names,
    malicious_trainers,
):
    replaced_PSs = replace_registered_PSs(len(malicious_ps), black_list)
    new_trainers = []

    for m_ps_name in malicious_ps:
        name_trainers = trainers_names[m_ps_name]
        trainers = [rt for rt in registered_trainers if rt["name"] in name_trainers]

        for t in trainers:
            if t["name"] in malicious_trainers:
                replaced_t = replace_registered_trainers(
                    1, all_trainers_names, blacklist=[]
                )
                new_trainers.extend(replaced_t)
                all_trainers_names.append(replaced_t[0]["name"])
            else:
                new_trainers.append(t)

    return replaced_PSs, new_trainers, all_trainers_names


def setup_updated_ps(
    num_trainers,
    replaced_PSs,
    new_trainers,
    trainers_names,
    ref_dist_PSs,
    PSs_name,
    num_classes,
    cipher_obj,
    private_chain_root_url_encried,
    malicious_ps,
):
    _ref_dist_PSs, _trainers_names, _PSs_name = setup_ps(
        num_trainers,
        replaced_PSs,
        new_trainers,
        num_classes,
        cipher_obj,
        private_chain_root_url_encried,
    )

    ref_dist_PSs_new, trainers_names_new, PSs_name_new = [], {}, []
    for rref, ps_name in zip(ref_dist_PSs, PSs_name):
        if ps_name not in malicious_ps:
            ref_dist_PSs_new.append(rref)
            trainers_names_new[ps_name] = trainers_names[ps_name]
            PSs_name_new.append(ps_name)

    ref_dist_PSs_new.extend(_ref_dist_PSs)
    PSs_name_new.extend(_PSs_name)
    trainers_names_new.update(_trainers_names)

    return ref_dist_PSs_new, trainers_names_new, PSs_name_new


def replace_malicious_trainers(
    malicious_trainers,
    trainers_names,
    registered_trainers,
    all_trainers_names,
    registered_PSs,
):
    ps_with_mt = {
        ps_name: [mt for mt in malicious_trainers if mt in trainers]
        for ps_name, trainers in trainers_names.items()
    }

    new_trainers, affect_ps_names = [], []
    for ps_name, mts in ps_with_mt.items():
        if mts:
            affect_ps_names.append(ps_name)
            name_trainers = trainers_names[ps_name]
            trainers = [rt for rt in registered_trainers if rt["name"] in name_trainers]

            for t in trainers:
                if t["name"] in mts:
                    replaced_t = replace_registered_trainers(
                        1, all_trainers_names, blacklist=[]
                    )
                    new_trainers.extend(replaced_t)
                    all_trainers_names.append(replaced_t[0]["name"])
                else:
                    new_trainers.append(t)

    affected_PSs = [ps for ps in registered_PSs if ps["name"] in affect_ps_names]
    return affected_PSs, new_trainers, all_trainers_names


def blacklist_process(
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
):
    black_list = update_blacklist(black_list)
    all_trainers_names = gather_all_trainers(trainers_names)

    if malicious_ps:
        replaced_PSs, new_trainers, all_trainers_names = handle_malicious_ps(
            malicious_ps,
            black_list,
            trainers_names,
            registered_trainers,
            all_trainers_names,
            malicious_trainers,
        )

        ref_dist_PSs, trainers_names, PSs_name = setup_updated_ps(
            num_trainers,
            replaced_PSs,
            new_trainers,
            trainers_names,
            ref_dist_PSs,
            PSs_name,
            num_classes,
            cipher_obj,
            private_chain_root_url_encried,
            malicious_ps,
        )

    if malicious_trainers:
        affected_PSs, new_trainers, all_trainers_names = replace_malicious_trainers(
            malicious_trainers,
            trainers_names,
            registered_trainers,
            all_trainers_names,
            registered_PSs,
        )
        ref_dist_PSs, trainers_names, PSs_name = setup_updated_ps(
            num_trainers,
            affected_PSs,
            new_trainers,
            trainers_names,
            ref_dist_PSs,
            PSs_name,
        )

    return ref_dist_PSs, trainers_names, PSs_name
