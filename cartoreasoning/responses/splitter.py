import polars as pl

# file_name = './gemini_wo_contextual.csv'

# pl_org = pl.read_json('/home/yaoyi/pyo00005/carto-reasoning/cartoreasoning/batch_files/carto_gen_full.json').select(
#     pl.col(['question_ref', 'question_text', 'expected_answer', 'image_urls', 'map_count', 'spatial_relationship', 'answer_type', 'contextual_urls']),
#     q_answered = pl.lit(True)
# )

# pl_data = pl.read_csv(file_name, encoding='utf8-lossy').rename({'id': 'question_ref', 
#                                          'response': 'gemini_response'}).with_columns(
#                                              pl.col('gemini_response').str.split('Final answer: ').list.last().str.strip_chars()
#                                          )

# pl_data = pl.concat(
#     [pl_org, pl_data],
#     how='align'
# )

# print(pl_data)


# pd_data = pl_data.to_pandas()
# pd_data.to_json(file_name, orient='records', indent=4)

file_name = '/home/yaoyi/pyo00005/carto-reasoning/cartoreasoning/responses/gemini_wo_contextual.json'
# col_name = 'gemini_response'

pl_org = pl.read_json(file_name).drop('gemini_response')

print(pl_org)

# pl_filtered = pl_data.filter(
#     pl.col('correct') == False
# )

# print(pl_filtered)
# pl_data = pl.read_json(file_name).with_columns(
#     pl.when(
#         pl.col(col_name).str.contains('\nassistant\n'))
#         .then(pl.col(col_name).str.split('\nassistant\n').list.last())
#         .otherwise(pl.col(col_name))
# )

# pd_data = pl_data.to_pandas()
# pd_data.to_json(file_name, orient='records', indent=4)

# print(pl_data)

pl_data = pl.read_ndjson('/home/yaoyi/pyo00005/carto-reasoning/cartoreasoning/responses/batch_68c057db73b08190a1b506717c1bdb56_output.jsonl').select(
    pl.col('response'),
    question_ref = pl.col('custom_id')
).unnest('response').unnest('body').select(
    pl.col(['question_ref', 'choices'])
).with_columns(
    pl.col('choices').list.first()
).unnest('choices').unnest('message').select(
    pl.col('question_ref'),
    pl.col('content').str.split('Final answer: ').list.last()
).rename({'content': 'gpt5_response'})


pl_data = pl.concat(
    [pl_org, pl_data],
    how='align'
).sort('question_ref').to_pandas()

file_name = '/home/yaoyi/pyo00005/carto-reasoning/cartoreasoning/responses/gpt5_w_contextual.json'
pl_data.to_json(file_name, orient='records', indent=4)



print(pl_data)