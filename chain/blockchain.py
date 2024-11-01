import hashlib
import json
from time import time
from uuid import uuid4
from urllib.parse import urlparse
from enum import Enum


class BlockType(str, Enum):
    START_JOB = "START_JOB"
    INIT_CONN = "INIT_CONNECT"
    EPOCH = "EPOCH"
    EVALUATE = "EVALUATE"


class TransactionEvents(str, Enum):
    MASTER_URL = "MASTER_URL"
    NEW_NODE = "NEW_NODE"
    TRAINERS = "TRAINERS"
    PARAMSERVERS = "PARAMSERVERS"
    PARAMETERS = "PARAMETERS"
    VALIDATE = "VALIDATE"
    ACCURACY = "ACCURACY"
    TRAIN_LOSS = "TRAIN_LOSS"
    VAL_LOSS = "VAL_LOSS"
    BLACKLIST = "BLACKLIST"


class BlockChain(object):
    """Main BlockChain class"""

    def __init__(self):
        self.chain = []
        self.current_transactions = []

    def new_block(self, block_type, proof, previous_hash=None):
        # creates a new block in the blockchain

        block = {
            "index": len(self.chain) + 1,
            "timestamp": time(),
            "transactions": [],
            "proof": proof,
            "previous_hash": previous_hash or self.hash(self.chain[-1]),
            "block_type": block_type,
        }

        # reset the current list of transactions
        self.current_transactions = []
        self.chain.append(block)
        return block

    @property
    def last_block(self):
        # returns last block in the chain

        if len(self.chain) > 0:
            return self.chain[-1]
        else:
            return None

    def new_transaction(self, sender, recipient, trans_type, data):
        # adds a new transaction into the list of transactions
        # these transactions go into the next mined block

        if len(self.chain) == 0:
            raise ValueError("Please create a blockchain before creating tranactions.")
        else:
            block = self.last_block

        self.current_transactions.append(
            {
                "sender": sender,
                "recipient": recipient,
                "type": trans_type,
                "data": data,
            }
        )

        block["transactions"] = self.current_transactions
        return int(block["index"]) + 1

    def proof_of_work(self, last_proof):
        # simple proof of work algorithm
        # find a number p' such as hash(pp') containing leading 4 zeros where p is the previous p'
        # p is the previous proof and p' is the new proof

        proof = 0
        while self.validate_proof(last_proof, proof) is False:
            proof += 1
        return proof

    def full_chain(self):
        # xxx returns the full chain and a number of blocks
        pass

    @staticmethod
    def hash(block):
        # hashes a block
        # also make sure that the transactions are ordered otherwise we will have inconsistent hashes!

        block_string = json.dumps(block, sort_keys=True).encode()
        return hashlib.sha256(block_string).hexdigest()

    @staticmethod
    def validate_proof(last_proof, proof):
        # validates the proof: does hash(last_proof, proof) contain 4 leading zeroes?

        guess = f"{last_proof}{proof}".encode()
        guess_hash = hashlib.sha256(guess).hexdigest()
        return guess_hash[:4] == "0000"
