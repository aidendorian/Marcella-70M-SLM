import os
import numpy as np
import torch
from torch.utils.data import IterableDataset, DataLoader

class ShardedDataset(IterableDataset):
    def __init__(self, data_dir, block_size, start_shard=0, start_seq=0):
        super().__init__()
        self.data_dir = data_dir
        self.block_size = block_size
        self.start_shard = start_shard
        self.start_seq = start_seq
        self.chunks_yielded = 0
        self.current_shard = start_shard
        self.current_seq = 0

    def _get_shards(self):
        return sorted([
            f for f in os.listdir(self.data_dir)
            if f.endswith('.bin')
        ])

    def __iter__(self):
        shards = self._get_shards()

        for shard_idx, shard_file in enumerate(shards):
            if shard_idx < self.start_shard:
                continue

            bin_path = os.path.join(self.data_dir, shard_file)
            idx_path = bin_path.replace('.bin', '.idx')

            data    = np.fromfile(bin_path, dtype=np.uint16)
            offsets = np.fromfile(idx_path, dtype=np.int64)

            seq_start = self.start_seq if shard_idx == self.start_shard else 0

            for seq_i in range(seq_start, len(offsets)):
                offset = offsets[seq_i]
                chunk  = data[offset: offset + self.block_size + 1]

                if len(chunk) < self.block_size + 1:
                    continue

                x = torch.tensor(chunk[:-1].astype(np.int64))
                y = torch.tensor(chunk[1:].astype(np.int64))

                self.current_shard = shard_idx
                self.current_seq   = seq_i
                self.chunks_yielded += 1

                yield x, y


def get_data(data_dir: str,
             block_size: int = 1024,
             batch_size: int = 5,
             num_workers: int = 0,
             pin_memory: bool = False,
             prefetch_factor=None,
             persistent_workers: bool = False,
             start_shard: int = 0,
             start_seq: int = 0):

    dataset = ShardedDataset(
        data_dir=data_dir,
        block_size=block_size,
        start_shard=start_shard,
        start_seq=start_seq
    )

    data = DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        prefetch_factor=prefetch_factor,
        persistent_workers=persistent_workers
    )

    return data, dataset


def get_val_batch(data_dir: str,
                  block_size: int,
                  batch_size: int,
                  device):
    
    bin_path = os.path.join(data_dir, 'shard_00000.bin')
    idx_path = os.path.join(data_dir, 'shard_00000.idx')

    data    = np.fromfile(bin_path, dtype=np.uint16)
    offsets = np.fromfile(idx_path, dtype=np.int64)

    xs, ys = [], []
    for b in range(batch_size):
        offset = offsets[b]
        chunk  = data[offset: offset + block_size + 1]
        xs.append(torch.tensor(chunk[:-1].astype(np.int64)))
        ys.append(torch.tensor(chunk[1:].astype(np.int64)))

    return torch.stack(xs).to(device), torch.stack(ys).to(device)