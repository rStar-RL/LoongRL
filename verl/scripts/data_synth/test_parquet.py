# read a parquet file and output the first item

parquet_file = "/mnt/longcontext/models/siyuan/rl_datasets/rl_three/no_system/mathqa_choice_start_idx0_end_idx5000/train.parquet"
import pyarrow.parquet as pq

# Read the Parquet file
table = pq.read_table(parquet_file)

# convert to json
data = table.to_pydict()
data = [dict(zip(table.column_names, row)) for row in zip(*data.values())]
# print the first item
print(data[0])
