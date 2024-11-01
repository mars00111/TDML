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
from demo_code.chain.chain_config import Settings


router = APIRouter(prefix="/v1")
blockchain = BlockChain()


@router.get("/publish_job", status_code=200)
def publish_job(job_info: JobInfo):
    last_block = blockchain.last_block
    if last_block is not None:
        last_proof = last_block["proof"]
        proof = blockchain.proof_of_work(last_proof)
        previous_hash = blockchain.hash(last_block)
        block = blockchain.new_block(
            BlockType.START_JOB, proof=proof, previous_hash=previous_hash
        )
    else:
        proof = 100
        block = blockchain.new_block(BlockType.START_JOB, previous_hash=1, proof=proof)

    blockchain.new_transaction(
        job_info.publisher, "", TransactionEvents.NEW_NODE, job_info
    )

    response = {
        "index": block["index"],
        "transactions": block["transactions"],
        "proof": block["proof"],
        "previous_hash": block["previous_hash"],
    }

    return JSONResponse(response)


@router.get("/curr_job", status_code=200)
def curr_job():
    chain = blockchain.chain
    if len(chain) == 0:
        raise HTTPException(status_code=404, detail="No jobs published.")

    for block in chain:
        if block["block_type"] == BlockType.START_JOB:
            cts = block["transactions"]

    job_info = {}
    for t in cts:
        if t["type"] == TransactionEvents.NEW_NODE:
            job_info = t["data"]

    response = {
        "job": job_info,
    }
    return JSONResponse(response)


@router.post("/ps/registration", status_code=201)
def register_ps(ps_info: PSInfo):
    try:
        blk_idx = blockchain.new_transaction(
            ps_info.name, ps_info.root, TransactionEvents.NEW_NODE, ps_info.job_id
        )
    except ValueError as e:
        print(e)
        raise HTTPException(status_code=404, detail=repr(e))

    response = {
        "message": f"New PS [{ps_info.name}] registered.",
    }

    return JSONResponse(response)


@router.get("/PSs", status_code=200)
def PSs():
    chain = blockchain.chain
    if len(chain) == 0:
        raise HTTPException(status_code=404, detail="No trainers registered.")

    for block in chain:
        if block["block_type"] == BlockType.START_JOB:
            cts = block["transactions"]

    registered_PSs = []
    for t in cts:
        if t["type"] == TransactionEvents.NEW_NODE and isinstance(t["data"], PSInfo):
            ps = {
                "name": t["sender"],
                "root": t["recipient"],
                "job_id": t["data"].job_id,
            }
            registered_PSs.append(ps)

    response = {
        "PSs": registered_PSs,
    }
    return JSONResponse(response)


@router.post("/trainer/registration", status_code=201)
def register_trainer(client_info: TrainerInfo):
    try:
        client_name = client_info.name
        recipient = client_info.root
        blockchain.new_transaction(
            client_name, recipient, TransactionEvents.NEW_NODE, client_info
        )
    except ValueError as e:
        print(e)
        raise HTTPException(status_code=404, detail=repr(e))

    response = {
        "message": f"New client [{client_name}] registered.",
    }

    return JSONResponse(response)


@router.get("/trainers", status_code=200)
def trainers():
    chain = blockchain.chain
    if len(chain) == 0:
        raise HTTPException(status_code=404, detail="No trainers registered.")

    for block in chain:
        if block["block_type"] == BlockType.START_JOB:
            cts = block["transactions"]

    registered_trainers = []
    for t in cts:
        if t["type"] == TransactionEvents.NEW_NODE and isinstance(
            t["data"], TrainerInfo
        ):
            gpu_info = t["data"].gpu_info
            trainer = {
                "name": t["sender"],
                "root": t["recipient"],
                "gpu_info": gpu_info,
            }
            registered_trainers.append(trainer)

    response = {
        "trainers": registered_trainers,
    }
    return JSONResponse(response)


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
        port=int(os.getenv("PORT", settings.public_port)),
        log_level="debug",
    )
