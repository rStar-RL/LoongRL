import argparse
import pyarrow.parquet as pq
import pyarrow as pa
import os

parser = argparse.ArgumentParser(description="Sample a subset of a Parquet dataset.")
parser.add_argument(
    "--parquet_file",
    "-p",
    type=str,
    default="/mnt/longcontext/models/siyuan/rl_datasets/rl_three/system/DAPO-Math-17k_5000/test.parquet",
    help="Path to the input Parquet file.",
)
parser.add_argument(
    "--sample_size",
    "-s",
    type=int,
    default=100,
    help="Number of rows to sample from the dataset.",
)
args = parser.parse_args()

parquet_file = args.parquet_file
sample_size = args.sample_size

table = pq.read_table(parquet_file)

num_rows = table.num_rows

if sample_size > num_rows:
    raise ValueError(f"sample_size ({sample_size}) is greater than the number of rows in the dataset ({num_rows})")

import random
sample_indices = sorted(random.sample(range(num_rows), sample_size))
sampled_table = table.take(pa.array(sample_indices))

sample_file = os.path.splitext(parquet_file)[0] + f"_sample_{sample_size}.parquet"
pq.write_table(sampled_table, sample_file)

print(f"Sampled {sample_size} rows and saved to '{sample_file}'")