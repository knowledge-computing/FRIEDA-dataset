import polars as pl

pl_data = pl.read_json('/home/yaoyi/pyo00005/carto-reasoning/questions/annotator_response/reponse_mini.json').with_columns(
    pl.col('annotator_response').list.get(0)
).drop('expected_answer').drop('question_text')

pd_data = pl_data.to_pandas()
pd_data.to_json('./test_sample.json', orient='records', indent=4)

print(pl_data)