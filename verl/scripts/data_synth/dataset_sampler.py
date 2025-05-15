import pyarrow.parquet as pq
import pyarrow as pa
import os

parquet_file = "your_file.parquet"  
sample_size = 100                   

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