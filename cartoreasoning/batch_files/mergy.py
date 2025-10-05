import polars as pl

list_stuff = ['0', '50', '100', '150', 'remain']

list_dfs = []

for i in list_stuff:
    pl_data = pl.read_json(f'./carto_gen_{i}.json')
    print(pl_data)
    list_dfs.append(pl_data)

pl_full = pl.concat(
    list_dfs,
    how='diagonal'
)

pd_full = pl_full.to_pandas()
pd_full.to_json('./carto_gen_full.json', orient='records', indent=4)

print(pl_full)