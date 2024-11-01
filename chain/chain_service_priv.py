from typing import List, Dict, Tuple

from fastapi import FastAPI
from fastapi import APIRouter
from fastapi.middleware.cors import CORSMiddleware
from fastapi.exceptions import HTTPException
from fastapi.responses import JSONResponse
from fastapi.encoders import jsonable_encoder

from blockchain import BlockChain, BlockType, TransactionEvents
from rest_objects import *

import os
import uvicorn
from chain.chain_config import Settings


router = APIRouter(prefix="/v1")
blockchain = BlockChain()


@router.post("/init_connect")
def init_conn(master: MasterInfo, status_code=200):
    last_block = blockchain.last_block
    if last_block is not None:
        last_proof = last_block["proof"]
        proof = blockchain.proof_of_work(last_proof)
        previous_hash = blockchain.hash(last_block)
        block = blockchain.new_block(
            BlockType.INIT_CONN, proof=proof, previous_hash=previous_hash
        )
    else:
        proof = 100
        block = blockchain.new_block(BlockType.INIT_CONN, previous_hash=1, proof=proof)

    blockchain.new_transaction(
        sender=master.root,
        recipient="init_connect",
        trans_type=TransactionEvents.MASTER_URL,
        data={"url": master.url, "port": master.port},
    )

    response = {
        "index": block["index"],
        "transactions": block["transactions"],
        "proof": block["proof"],
        "previous_hash": block["previous_hash"],
    }

    return JSONResponse(response)


@router.post("/init_training")
def init_training(clients: List[TrainerInfo], status_code=200):
    last_block = blockchain.last_block
    if last_block is None:
        raise HTTPException(status_code=400, detail="Chain is not created.")

    last_proof = last_block["proof"]
    proof = blockchain.proof_of_work(last_proof)
    previous_hash = blockchain.hash(last_block)
    block = blockchain.new_block(BlockType.EPOCH, proof, previous_hash)

    master = clients[0].root
    clients_ = [c.model_dump() for c in clients]
    blockchain.new_transaction(
        sender=master,
        recipient="",
        trans_type=TransactionEvents.TRAINERS,
        data=clients_,
    )

    response = {
        "index": block["index"],
        "transactions": jsonable_encoder(block["transactions"]),
        "proof": block["proof"],
        "previous_hash": block["previous_hash"],
    }

    return JSONResponse(response)


@router.post("/epoch", status_code=200)
def epoch(epoch_info: EpochInfo):
    last_block = blockchain.last_block
    if last_block is None:
        raise HTTPException(status_code=400, detail="Chain is not created.")

    if last_block["block_type"] is not BlockType.EPOCH:
        raise HTTPException(
            status_code=400, detail="Training is not initialised in the chain."
        )

    last_epoch_n = -1
    transaction = last_block["transactions"][-1]
    if transaction["type"] == TransactionEvents.PARAMETERS:
        last_epoch_n = transaction["data"]["epoch_n"]
        assert last_epoch_n <= epoch_info.epoch_n

    if last_epoch_n < epoch_info.epoch_n:
        last_proof = last_block["proof"]
        proof = blockchain.proof_of_work(last_proof)
        previous_hash = blockchain.hash(last_block)
        block = blockchain.new_block(BlockType.EPOCH, proof, previous_hash)
    else:
        block = last_block

    blockchain.new_transaction(
        sender="global",
        recipient="",
        trans_type=TransactionEvents.PARAMETERS,
        data=epoch_info.model_dump(),
    )

    response = {
        "message": f"Global {epoch_info.epoch_n} parameters saved.",
    }

    return JSONResponse(response)


@router.post("/validation", status_code=200)
def validation(epoch_info: EpochInfoNode):
    last_block = blockchain.last_block
    if last_block is None:
        raise HTTPException(status_code=400, detail="Chain is not created.")

    if last_block["block_type"] is not BlockType.EPOCH:
        raise HTTPException(
            status_code=400, detail="Training is not initialised in the chain."
        )

    last_epoch_n = -1
    transaction = last_block["transactions"][-1]
    if transaction["type"] == TransactionEvents.VALIDATE:
        last_epoch_n = transaction["data"]["epoch_n"]
        assert last_epoch_n <= epoch_info.epoch_n

    if last_epoch_n < epoch_info.epoch_n:
        last_proof = last_block["proof"]
        proof = blockchain.proof_of_work(last_proof)
        previous_hash = blockchain.hash(last_block)
        block = blockchain.new_block(BlockType.EPOCH, proof, previous_hash)
    else:
        block = last_block

    blockchain.new_transaction(
        sender=epoch_info.client_id,
        recipient="",
        trans_type=TransactionEvents.VALIDATE,
        data=epoch_info.model_dump(),
    )

    response = {
        "message": f"Client {epoch_info.client_id} validation saved.",
    }

    return JSONResponse(response)


@router.post("/validation_list", status_code=200)
def validation_list(epoch_n: int):
    last_block = blockchain.last_block
    if last_block is None:
        raise HTTPException(status_code=400, detail="Chain is not created.")

    if last_block["block_type"] is not BlockType.EPOCH:
        raise HTTPException(
            status_code=400, detail="Training is not initialised in the chain."
        )

    val_transactions = []
    chain = blockchain.chain
    for block in reversed(chain):
        if block["block_type"] == BlockType.EPOCH:
            for t in reversed(block["transactions"]):
                if (
                    t["type"] == TransactionEvents.VALIDATE
                    and t["epoch_n"] == epoch_n
                    and t["type"] == NodeType.PS
                ):
                    val_transactions.append(t)

    return JSONResponse(val_transactions)


@router.post("/parameters", status_code=200)
def parameters(query_ids: List[str]):
    last_block = blockchain.last_block
    if last_block is None:
        raise HTTPException(status_code=400, detail="Chain is not created.")

    if last_block["block_type"] != BlockType.EPOCH:
        raise HTTPException(status_code=400, detail="Training is not initialised.")

    parameter_transactions = []
    chain = blockchain.chain
    for block in reversed(chain):
        if block["block_type"] == BlockType.EPOCH:
            for t in reversed(block["transactions"]):
                if t["type"] == TransactionEvents.PARAMETERS:
                    parameter_transactions.append(t)

            if len(parameter_transactions) == len(query_ids):
                break
            else:
                parameter_transactions = []

    epoch_ns = []
    client_param_url = {}
    client_param_hash = {}

    for c_id in query_ids:
        for t in parameter_transactions:
            if t["data"]["client_id"] == c_id:
                epoch_ns.append(t["data"]["epoch_n"])
                client_param_url[c_id] = t["data"]["parameter_url"]
                client_param_hash[c_id] = t["data"]["parameter_hash"]

    epoch_ns = list(set(epoch_ns))
    if len(epoch_ns) != 1:
        epoch_ns = [0]
        client_param_url = {}
        client_param_hash = {}

    client_param_url["epoch_n"] = epoch_ns[0]
    return JSONResponse(client_param_url)


@router.post("/accuracy", status_code=200)
def accuracy(epoch_val: EpochVal):
    last_block = blockchain.last_block
    if last_block is None:
        raise HTTPException(status_code=400, detail="Chain is not created.")

    if last_block["block_type"] is not BlockType.EPOCH:
        raise HTTPException(
            status_code=400, detail="Training is not initialised in the chain."
        )

    last_epoch_n = -1
    transaction = last_block["transactions"][-1]
    if transaction["type"] == TransactionEvents.PARAMETERS:
        last_epoch_n = transaction["data"]["epoch_n"]
        assert last_epoch_n <= epoch_val.epoch_n

    if last_epoch_n < epoch_val.epoch_n:
        last_proof = last_block["proof"]
        proof = blockchain.proof_of_work(last_proof)
        previous_hash = blockchain.hash(last_block)
        block = blockchain.new_block(BlockType.EPOCH, proof, previous_hash)
    else:
        block = last_block

    blockchain.new_transaction(
        sender=epoch_val.client_id,
        recipient="",
        trans_type=TransactionEvents.ACCURACY,
        data=epoch_val.model_dump(),
    )

    response = {
        "message": f"Client {epoch_val.client_id} accuracy saved.",
    }

    return JSONResponse(response)


@router.post("/accuracy_list", status_code=200)
def accuracy_list(epoch_n: int):
    last_block = blockchain.last_block
    if last_block is None:
        raise HTTPException(status_code=400, detail="Chain is not created.")

    if last_block["block_type"] is not BlockType.EPOCH:
        raise HTTPException(
            status_code=400, detail="Training is not initialised in the chain."
        )

    acc_transactions = []
    chain = blockchain.chain
    for block in reversed(chain):
        if block["block_type"] == BlockType.EPOCH:
            for t in reversed(block["transactions"]):
                if t["type"] == TransactionEvents.ACCURACY and t["epoch_n"] == epoch_n:
                    acc_transactions.append(t)

    return JSONResponse(acc_transactions)


@router.post("/ps/trainers", status_code=200)
def ps_trainers(ps_trainers: PSTrainers):
    last_block = blockchain.last_block
    if last_block is None:
        raise HTTPException(status_code=400, detail="Chain is not created.")

    if last_block["block_type"] is not BlockType.EPOCH:
        raise HTTPException(
            status_code=400, detail="Training is not initialised in the chain."
        )

    last_proof = last_block["proof"]
    proof = blockchain.proof_of_work(last_proof)
    previous_hash = blockchain.hash(last_block)
    block = blockchain.new_block(BlockType.EPOCH, proof, previous_hash)

    blockchain.new_transaction(
        sender=ps_trainers.name,
        recipient="",
        trans_type=TransactionEvents.PARAMSERVERS,
        data=ps_trainers.model_dump(),
    )

    response = {
        "message": f"PS {ps_trainers.name} trainers saved.",
    }

    return JSONResponse(response)


@router.post("/ps/trainers_list", status_code=200)
def ps_trainers_list(ps_name: str):
    last_block = blockchain.last_block
    if last_block is None:
        raise HTTPException(status_code=400, detail="Chain is not created.")

    if last_block["block_type"] is not BlockType.EPOCH:
        raise HTTPException(
            status_code=400, detail="Training is not initialised in the chain."
        )

    ps_trainers = []
    chain = blockchain.chain
    for block in reversed(chain):
        if block["block_type"] == BlockType.EPOCH:
            for t in reversed(block["transactions"]):
                if (
                    t["type"] == TransactionEvents.PARAMSERVERS
                    and t["sender"] == ps_name
                ):
                    ps_trainers = t["data"]
                    break
            break

    return JSONResponse(ps_trainers)


@router.post("/blacklist", status_code=200)
def blacklist(node_names: List[str]):
    last_block = blockchain.last_block
    if last_block is None:
        raise HTTPException(status_code=400, detail="Chain is not created.")

    blockchain.new_transaction(
        sender="",
        recipient="",
        trans_type=TransactionEvents.BLACKLIST,
        data=node_names,
    )

    blacklist = []
    chain = blockchain.chain
    for block in reversed(chain):
        if block["block_type"] == BlockType.EPOCH:
            for t in reversed(block["transactions"]):
                if t["type"] == TransactionEvents.BLACKLIST:
                    blacklist.extend(t["data"])

    blacklist = list(set(blacklist))
    return JSONResponse(blacklist)


# @router.post("/trainer/transactions", status_code=201)
# def trainer_transactions(client_id: str):
#     if blockchain.last_block is None:
#         raise HTTPException(
#             status_code=404, detail="No transactions found. Chain is not created."
#         )

#     cts = blockchain.last_block["transactions"]

#     client_cts = []
#     for t in cts:
#         if t["recipient"] == client_id or t["recipient"] == "init_connect":
#             client_cts.append(t)

#     response = {
#         "transactions": client_cts,
#     }

#     return JSONResponse(response)


@router.get("/chain", status_code=200)
def full_chain():
    response = {
        "chain": blockchain.chain,
        "length": len(blockchain.chain),
    }
    return JSONResponse(response)


def start_chain_service(_settings: Settings):
    description = """This is a blockchain backend service for our distributed ML training and model deployment.

    """

    app = FastAPI(
        title=_settings.app_name,
        version=_settings.app_version,
        description=description,
        contact={
            "name": _settings.admin_name,
            "email": _settings.admin_email,
        },
    )

    origins = [
        "http://localhost",
        "http://velocity-ev",
        "http://vizhead01-ev",
    ]

    app.add_middleware(
        CORSMiddleware,
        allow_origins=origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    app.include_router(router)
    return app


if __name__ == "__main__":
    settings = Settings()
    app = start_chain_service(settings)

    uvicorn.run(
        app,
        host=os.getenv("HOST", settings.host),
        port=int(os.getenv("PORT", settings.port)),
        log_level="debug",
    )
