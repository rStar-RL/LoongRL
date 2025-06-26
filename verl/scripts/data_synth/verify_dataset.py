#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Validate parquet paths and print row counts.

Usage:
    python count_parquet_rows.py
"""
from __future__ import annotations
import os
import sys
from pathlib import Path

try:
    import pyarrow.parquet as pq
    _USE_PA = True
except ImportError:
    import pandas as pd
    _USE_PA = False

# -------------------------------------------------
dataset_prefix = "/mnt/longcontext/models/siyuan"

train_files = [
    f"{dataset_prefix}/rl_datasets/rl_three/system/hotpotqa_qwen_filtered_start_idx0_end_idx2500_seq16384/train.parquet",
    f"{dataset_prefix}/rl_datasets/rl_three/system/hotpotqa_filtered-uuid_default_8-numchain_64-numhop-4-distractor_256_start_idx2500_end_idx5000_seq16384/train.parquet",
    f"{dataset_prefix}/rl_datasets/rl_three/system/musique_qwen_filtered_start_idx0_end_idx2500_seq16384/train.parquet",
    f"{dataset_prefix}/rl_datasets/rl_three/system/musique_filtered-uuid_default_8-numchain_64-numhop-4-distractor_256_start_idx2500_end_idx5000_seq16384/train.parquet",
    f"{dataset_prefix}/rl_datasets/rl_three/system/DAPO-Math-17k_2500/train.parquet",
    f"{dataset_prefix}/rl_datasets/rl_three/system/2wikimqa_qwen_filtered_start_idx0_end_idx2500_seq16384/train.parquet",
    f"{dataset_prefix}/rl_datasets/rl_three/system/2wikimqa_filtered-uuid_default_8-numchain_64-numhop-4-distractor_256_start_idx2500_end_idx5000_seq16384/train.parquet",
    f"{dataset_prefix}/rl_datasets/rl_three/system/mathqa_choice_start_idx0_end_idx2500/train.parquet",
    f"{dataset_prefix}/rl_datasets/rl_three/system/book_20key_10k_512_start_index_0_end_index_512_seqlen16k/train.parquet",
    f"{dataset_prefix}/rl_datasets/rl_three/system/book_20value_10k_512_start_index_0_end_index_512_seqlen16k/train.parquet",
    f"{dataset_prefix}/rl_datasets/rl_three/system/sentence_needle_fix_500_start_index_0_end_index_500_seq_16384/train.parquet",
]

test_files = [
    f"{dataset_prefix}/rl_datasets/rl_three/hotpotqa_valid/test.parquet",
    f"{dataset_prefix}/rl_datasets/rl_three/musique_valid/test.parquet",
    f"{dataset_prefix}/rl_datasets/rl_three/2wikimqa_valid/test.parquet",
    f"{dataset_prefix}/rl_datasets/rl_three/system/hotpotqa_qwen_filtered_start_idx0_end_idx2500_seq16384/valid.parquet",
    f"{dataset_prefix}/rl_datasets/rl_three/system/hotpotqa_filtered-uuid_default_8-numchain_64-numhop-4-distractor_256_start_idx2500_end_idx5000_seq16384/valid.parquet",
    f"{dataset_prefix}/rl_datasets/rl_three/system/musique_qwen_filtered_start_idx0_end_idx2500_seq16384/valid.parquet",
    f"{dataset_prefix}/rl_datasets/rl_three/system/musique_filtered-uuid_default_8-numchain_64-numhop-4-distractor_256_start_idx2500_end_idx5000_seq16384/valid.parquet",
    f"{dataset_prefix}/rl_datasets/rl_three/system/DAPO-Math-17k_2500/valid.parquet",
    f"{dataset_prefix}/rl_datasets/rl_three/system/2wikimqa_qwen_filtered_start_idx0_end_idx2500_seq16384/valid.parquet",
    f"{dataset_prefix}/rl_datasets/rl_three/system/2wikimqa_filtered-uuid_default_8-numchain_64-numhop-4-distractor_256_start_idx2500_end_idx5000_seq16384/valid.parquet",
    f"{dataset_prefix}/rl_datasets/rl_three/system/mathqa_choice_start_idx0_end_idx2500/valid.parquet",
    f"{dataset_prefix}/rl_datasets/rl_three/system/book_20key_10k_512_start_index_0_end_index_512_seqlen16k/valid.parquet",
    f"{dataset_prefix}/rl_datasets/rl_three/system/book_20value_10k_512_start_index_0_end_index_512_seqlen16k/valid.parquet",
    f"{dataset_prefix}/rl_datasets/rl_three/system/sentence_needle_fix_500_start_index_0_end_index_500_seq_16384/valid.parquet",
]


def count_rows(path: Path) -> int:
    """Return row count using pyarrow (fast) or pandas."""
    if _USE_PA:
        try:
            return pq.ParquetFile(path).metadata.num_rows
        except Exception as e:
            print(f"    ⚠️  pyarrow failed: {e} — falling back to pandas")
    # fallback
    import pandas as pd
    return len(pd.read_parquet(path))


def main():
    missing = False
    for split, paths in (("TRAIN", train_files), ("TEST", test_files)):
        print(f"\n=== {split} FILES ===")
        for p in paths:
            path = Path(p)
            if not path.exists():
                missing = True
                print(f"❌ Missing  : {path}")
            else:
                try:
                    n_rows = count_rows(path)
                    print(f"✅ Exists   : {path}  →  {n_rows:,} rows")
                except Exception as e:
                    print(f"⚠️  Error reading {path}: {e}")

    if missing:
        sys.exit("\nSome files are missing. Check paths above.")
    else:
        print("\nAll files verified.")

if __name__ == "__main__":
    main()