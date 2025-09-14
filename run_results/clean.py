import polars as pl

pl_raw = pl.read_json('/home/yaoyi/pyo00005/carto-reasoning/questions/response_full_d10.json')

folder_name = 'ovis2'
file_name = 'ovis2_wo_contextual.json'
pl_data = pl.read_json(f'/users/2/pyo00005/HOME/carto-reasoning/run_results/qwen25vl/ovis25_w_contextual.json').select(
    pl.col(['question_ref', 'q_answered', 'ovis2.5_response'])
).rename({'ovis2.5_response':'ovis25_response'})

pl_full = pl.concat(
    [pl_raw, pl_data],
    how='align'
).to_pandas()

pl_full.to_json(f'/users/2/pyo00005/HOME/carto-reasoning/run_results/ovis25/ovis25-9B_w_contextual.json',
                orient='records', indent=4)

print(pl_raw)