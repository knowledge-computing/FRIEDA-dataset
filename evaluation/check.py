import os
import polars as pl

relation = ['Border', 'Distance', 'Within']

list_df = []
for r in relation:
    pl_gem = pl.read_json(f'/home/yaoyi/pyo00005/carto-reasoning/evaluation/gemini_incorrect_{r}.json')
    list_df.append(pl_gem)

pl_gem = pl.concat(
    list_df,
    how='diagonal'
)

pl_son = pl.read_json('/users/2/pyo00005/HOME/carto-reasoning/evaluation_results/claude-sonnet4/sonnet_wo_contextual.json').rename({'_response':'sonnet_response'}).filter(
    pl.col('correct') == False
)

pl_gpt = pl.read_json('/users/2/pyo00005/HOME/carto-reasoning/evaluation_results/gpt5/gpt5_wo_contextual.json').rename({'_response':'gpt_response'}).filter(
    pl.col('correct') == False
)

pl_data = pl.concat(
    [pl_gem, pl_son],
    how='align_left'
).filter(
    pl.col("_response").list.set_symmetric_difference("sonnet_response").list.len() == 0
).drop('extracted_ans').drop('correct')['question_ref'].to_list()

pl_ano = pl.concat(
    [pl_gem, pl_gpt],
    how='align_left'
).filter(
    pl.col("_response").list.set_symmetric_difference("gpt_response").list.len() == 0,
    ~pl.col('question_ref').is_in(pl_data)
).to_pandas()

pl_ano.to_json('./gpt_to_test.json', orient='records', indent=4)

print(pl_gem)
print(pl_ano)

# import pickle

# with open('/home/yaoyi/pyo00005/p2/carto-reasoning/cartoreasoning/instruction.pkl', 'rb') as handle:
#     inst = pickle.load(handle).split("Reply in the format: 'Final answer: <your answer>'.")[0] 
#     inst += "Give the full reasoning and give the final answer as 'Final answer: <your answer>'."


# with open('/home/yaoyi/pyo00005/p2/carto-reasoning/cartoreasoning/instruction_reason.pkl', 'wb') as handle:
#     pickle.dump(inst, handle, protocol=pickle.HIGHEST_PROTOCOL)
# print(inst)