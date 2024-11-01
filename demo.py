import os
import time
from argparse import ArgumentParser
import torch
import torch.multiprocessing as mp

from configs import *
from create_remotes import enable_remoters


def main():
    parser = ArgumentParser()
    parser.add_argument(
        "--num-trainers", type=int, default=2, help="Number of trainers"
    )
    parser.add_argument(
        "--num-ps",
        type=int,
        default=2,
        help="Data parallel / Parameter servers instances",
    )
    args = parser.parse_args()

    os.environ['MASTER_ADDR'] = MASTER_ADDR
    os.environ['MASTER_PORT'] = MASTER_PORT

    world_size = args.num_trainers * args.num_ps + 1 + args.num_ps
    while True:
        try:
            mp.spawn(
                enable_remoters,
                args=(args.num_trainers, args.num_ps, world_size),
                nprocs=world_size,
                join=True,
            )
            break
        except Exception as e:
            print(f"Error encountered: {e}. Retrying in 10 seconds...")
            time.sleep(10)

if __name__ == "__main__":
    main()