import polars as pl

pl_raw = pl.read_json('/home/yaoyi/pyo00005/carto-reasoning/questions/response_full_d10.json')

folder_name = 'ovis2'
file_name = 'ovis2_wo_contextual.json'
pl_data = pl.read_json(f'/home/yaoyi/pyo00005/carto-reasoning/cartoreasoning/responses/{file_name}').select(
    pl.col(['question_ref', 'q_answered', 'ovis2_response'])
)
# .rename({'ovis2_response':'ovis25_response'})

pl_full = pl.concat(
    [pl_raw, pl_data],
    how='align'
).to_pandas()

pl_full.to_json(f'/users/2/pyo00005/HOME/carto-reasoning/run_results/{folder_name}/ovis2_wo_contextual.json',
                orient='records', indent=4)

print(pl_raw)