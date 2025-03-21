import os
import torch.distributed as dist
import torch
# from .std_logger import Logger


def setup_multinodes(local_rank, world_size):

    torch.cuda.set_device(local_rank)
    # initialize the process group
    ip = os.environ["MASTER_ADDR"]
    port = os.environ["MASTER_PORT"]

    # Logger.info("init_address: tcp://{}:{} | world_size: {}".format(
    #     ip, port, world_size))

    dist.init_process_group("nccl",
                            init_method='tcp://{}:{}'.format(ip, port),
                            rank=int(os.environ['RANK']),
                            world_size=world_size)

    # torch.multiprocessing.set_sharing_strategy("file_system")

    # Logger.info(
    #     f"[init] == local rank: {local_rank}, global rank: {os.environ['RANK']} =="
    # )


def cleanup_multinodes():
    dist.destroy_process_group()