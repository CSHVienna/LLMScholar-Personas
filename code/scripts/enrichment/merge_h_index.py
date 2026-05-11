"""Merge computed h-index back into factuality_full.csv.

Reads results/summary/oa_h_index.csv (oa_id, oa_h_index) and rewrites
factuality_full.csv with the populated column.
"""
from pathlib import Path
import pandas as pd

FACT = Path('results/summary/factuality_full.csv')
HIDX = Path('results/summary/oa_h_index.csv')

print(f'Loading {FACT} ...')
df = pd.read_csv(FACT, low_memory=False)
print(f'  {len(df):,} rows × {len(df.columns)} cols')

print(f'Loading {HIDX} ...')
hidx = pd.read_csv(HIDX)
print(f'  {len(hidx):,} authors with h-index')

# Replace empty oa_h_index column with mapped values
df = df.drop(columns=['oa_h_index'])
df = df.merge(hidx, on='oa_id', how='left')

# Re-order: put oa_h_index back next to oa_works_count for readability
cols = list(df.columns)
cols.remove('oa_h_index')
ins = cols.index('oa_cited_by_count') + 1
cols.insert(ins, 'oa_h_index')
df = df[cols]

print(f'h_index coverage: {df["oa_h_index"].notna().sum():,} / {len(df):,}  '
      f'({df["oa_h_index"].notna().mean()*100:.1f}%)')
print('h_index distribution:')
print(df['oa_h_index'].describe([.25, .5, .75, .9, .99]).round(1))

print(f'Writing {FACT} ...')
df.to_csv(FACT, index=False)
print('Done.')
