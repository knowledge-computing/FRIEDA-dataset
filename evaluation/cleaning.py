import polars as pl
from sklearn.metrics import cohen_kappa_score
import pickle

pl_data = pl.read_json('/home/yaoyi/pyo00005/carto-reasoning/questions/response_full_d10.json')

align3 = pl.read_csv('/home/yaoyi/pyo00005/carto-reasoning/questions/annotator_response/raw_csv/annot_align3.csv').group_by('question_ref').agg([pl.all()]).with_columns(
    count = 3
)
align2 = pl.read_csv('/home/yaoyi/pyo00005/carto-reasoning/questions/annotator_response/raw_csv/annot_align2.csv').group_by('question_ref').agg([pl.all()]).with_columns(
    count = 2
)

align_full = pl.concat(
    [align3, align2],
    how='diagonal'
)

list_human_wrong = align_full.filter(pl.col('count') == 2)['question_ref'].to_list()

model_paths = ['/users/2/pyo00005/HOME/carto-reasoning/evaluation_results/claude-sonnet4/sonnet_wo_contextual.json',
               '/users/2/pyo00005/HOME/carto-reasoning/evaluation_results/gemini25pro/gemini_wo_contextual.json',
               '/users/2/pyo00005/HOME/carto-reasoning/evaluation_results/glm45v/glm45v-108B_wo_contextual.json',
               '/users/2/pyo00005/HOME/carto-reasoning/evaluation_results/gpt5/gpt5_wo_contextual.json',
               '/users/2/pyo00005/HOME/carto-reasoning/evaluation_results/internvl3/internvl3-78B_wo_contextual.json',
               '/users/2/pyo00005/HOME/carto-reasoning/evaluation_results/internvl35/no_think/internvl35-38B_wo_contextual.json',
               '/users/2/pyo00005/HOME/carto-reasoning/evaluation_results/llava-next/llava-next-110B_wo_contextual.json',
               '/users/2/pyo00005/HOME/carto-reasoning/evaluation_results/llava-ov/llava-ov-72B_wo_contextual.json',
               '/users/2/pyo00005/HOME/carto-reasoning/evaluation_results/ovis2/ovis2-34B_wo_contextual.json',
               '/users/2/pyo00005/HOME/carto-reasoning/evaluation_results/ovis25/ovis25-9B-think_wo_contextual.json',
               '/users/2/pyo00005/HOME/carto-reasoning/evaluation_results/qwen25vl/qwen25vl-72B_wo_contextual.json']

summer = 0
for i in model_paths:
    pl_json = pl.read_json(i).filter(
        pl.col('correct') == False,
        pl.col('question_ref').is_in(list_human_wrong)
    ).shape[0]

    summer += pl_json

    print(i, len(list_human_wrong), pl_json)

print(summer/(len(list_human_wrong)*11))


# print(align_full.filter(
#     pl.col('spatial_relationship') == 'Orientation'
# ))
# print(align_full)

# align_full = align_full.select(
#     pl.col(['question_ref', 'count'])
# )

# # pl_manual = pl.concat(
# #     [pl_data, align_full],
# #     how='align_left'
# # ).rename({'count': 'manual_count'}).select(
# #     pl.col('question_ref', 'count')
# # )


# image_prefix = 'https://media.githubusercontent.com/media/YOO-uN-ee/carto-image/main/'

# import os
# def append_prefix(list_items):
#     return [os.path.join(image_prefix, i) for i in list_items]

# pl_gemini = pl.read_json('/users/2/pyo00005/HOME/carto-reasoning/evaluation_results/gemini25pro/gemini_wo_contextual.json').with_columns(
#     pl.col('image_urls').map_elements(append_prefix)
# )

# print(pl_gemini)

# pl_full = pl.concat(
#     [pl_gemini, align_full],
#     how='align'
# ).filter(
#     pl.col('correct') == False,
#     pl.col('count') == 3,
# )

# for i in pl_full.partition_by('spatial_relationship'):
#     i = i.select(pl.col(['question_ref', '_response', 'expected_answer', 'map_count', 'spatial_relationship']))
#     ti = i.to_pandas()
#     val = i.item(0, 'spatial_relationship')
#     ti.to_json(f'./gemini_incorrect_{val}.json',
#                orient='records', indent=4)


# print(pl_full)





# pl_empty = pl.DataFrame()
# for i in range(3):
#     pl_data = pl.read_json(f'/home/yaoyi/pyo00005/carto-reasoning/evaluation/tmp_ver4_{i}.json').select(
#         pl.col(['question_ref', 'correct'])
#     ).rename(
#         {'correct': f'correct_{i}'}
#     )

#     if pl_empty.shape[0] == 0:
#         pl_empty = pl_data
#     else:
#         pl_empty = pl.concat(
#             [pl_empty, pl_data],
#             how='align_full'
#         )

# pl_empty = pl_empty.with_columns(
#     pl.col(['correct_0', 'correct_1', 'correct_2']).fill_null(False)
# ).with_columns(
#     correct_count  = pl.col('correct_0') + pl.col('correct_1') + pl.col('correct_2')
# )

# pl_full = pl.concat(
#     [pl_manual, pl_empty],
#     how='align_inner'
# ).unique(
#     subset=['question_ref'],
#     maintain_order=True
# ).select(
#     pl.col(['question_ref', 'manual_count', 'correct_count'])
# ).with_columns(
#     pl.col(['manual_count', 'correct_count']).cast(pl.Int64)
# )

# # pl_full = pl_full.select(
# #     pl.col(['question_ref', 'manual_count'])
# # )

# # pl_gemini = pl.read_json('/home/yaoyi/pyo00005/carto-reasoning/evaluation/evaluation_result/gemini_wo_contextual.json').filter(
# #     pl.col('correct') == False
# # )

# # image_prefix = 'https://media.githubusercontent.com/media/YOO-uN-ee/carto-image/main/'

# # import os
# # def append_prefix(list_items):
# #     return [os.path.join(image_prefix, i) for i in list_items]

# # pl_gemini = pl_gemini.with_columns(
# #     pl.col('image_urls').map_elements(append_prefix)
# # )

# # pl_gemini = pl.concat(
# #     [pl_gemini, pl_full],
# #     how='align'
# # ).drop_nulls('correct')

# # pl_gemini_3 = pl_gemini.filter(
# #     pl.col('manual_count') == 3
# # )

# # pl_gemini_3_multi = pl_gemini_3.filter(
# #     pl.col('map_count') == 'Multi'
# # ).to_pandas()

# # pl_gemini_3_multi.to_json('./error_analysis/all_multi.json', orient='records', indent=4)

# # pl_gemini_3_single = pl_gemini_3.filter(
# #     pl.col('map_count') == 'Single'
# # ).to_pandas()
# # pl_gemini_3_single.to_json('./error_analysis/all_single.json', orient='records', indent=4)

# # pl_gemini_2 = pl_gemini.filter(
# #     pl.col('manual_count') == 2
# # )
# # pl_gemini_2_multi = pl_gemini_2.filter(
# #     pl.col('map_count') == 'Multi'
# # ).to_pandas()

# # pl_gemini_2_multi.to_json('./error_analysis/partial_multi.json', orient='records', indent=4)

# # pl_gemini_2_single = pl_gemini_2.filter(
# #     pl.col('map_count') == 'Single'
# # ).to_pandas()
# # pl_gemini_2_single.to_json('./error_analysis/partial_single.json', orient='records', indent=4)

# # Get rate

# full_len = pl_full.shape[0]

# print(pl_full.select(
#     pl.col('correct_count').sum()
# ).item(0, 'correct_count') / (3 * full_len))


# list_manual = pl_full['manual_count'].to_list()
# print(set(list_manual))

# list_correct = pl_full['correct_count'].to_list()
# print(set(list_correct))

# print(cohen_kappa_score(list_manual, list_correct))

# # print(pl_full.filter(
# #     pl.col('manual_count') != pl.col('correct_count')
# # ).rename({'correct_count': 'eval_script_count'}))

# # print(pl_full)