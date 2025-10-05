import re
import polars as pl

def _check_mape(expected, response):
    """Check if entire list contains only numeric values."""
    try:
        flt_expected = float(re.findall(r"\d+\.?\d*", expected[-1])[0])
        flt_response = float(re.findall(r"\d+\.?\d*", response[-1])[0])

        mape_val = (abs(flt_expected - flt_response) / flt_expected) * 100

        return True if mape_val <= 20 else False
    except: 
        return False

def eval_dist(df):
    """
    Evaluate distance answers
    """

    df = df.with_columns(
        correct = pl.struct(pl.all()).map_elements(lambda x: _check_mape(x['expected_answer'], x['_response']))
    )

    return df

folder = "ovis25/ovis25-9B_w_contextual.json"
orgg_file = f'/users/2/pyo00005/HOME/carto-reasoning/run_results/{folder}'
eval_file = f'/users/2/pyo00005/HOME/carto-reasoning/evaluation_results/{folder}'


# pl_orgg = pl.read_json(orgg_file).filter(
#     pl.col('answer_type') == 'distance'
# )

# response_col = next((col for col in pl_orgg.columns if col.endswith("_response")), None)
# pl_orgg = pl_orgg.rename({response_col: "_response"}).select(
#     pl.col(['question_ref', '_response', 'question_text', 'expected_answer', 'image_urls', 'map_count', 'spatial_relationship', 'answer_type']),
#     extracted_ans = '_response'
# ).with_columns(
#     pl.col(["expected_answer", "_response"])
#     .str.split(";")
#     .list.eval(pl.element().str.strip_chars())
# )

# pl_orgg = eval_dist(pl_orgg)
# pl_full = pl.concat(
#     [pl_orgg, pl_eval],
#     how='diagonal'
# ).unique()
# print(pl_full)





list_df = []
for i in range(3):
    pl_data = pl.read_json(f'/home/yaoyi/pyo00005/carto-reasoning/evaluation/tmp_ver5_{i}.json')    
    list_df.append(pl_data)

pl_full = pl.concat(
    list_df,
    how='diagonal'
)

counts = (
    pl_full.group_by("question_ref")
    .count()
    .rename({"count": "n"})
)
short_questions = counts.filter(pl.col("n") < 3)
fillers = (
    short_questions
    .with_columns(
        (3 - pl.col("n")).alias("missing")  # how many to add
    )
    .select(["question_ref", "missing"])
    .with_columns(pl.arange(0, pl.count()).alias("tmp_id"))  # helper
)
dummy_rows = pl.DataFrame({
    "question_ref": fillers["question_ref"],
    "_response": [["None"]] * fillers.height,  # example filler
    "correct": [False] * fillers.height
})

df_completed = pl.concat([pl_full, dummy_rows], how="diagonal")
pl_full = df_completed

# overall = (pl_full.select(
#         pl.col('correct').sum()
#     ).item(0, 'correct') * 100 / pl_full.shape[0])

# string_spatial = f"{overall:.2f} & "
# string_others = f"{overall:.2f} & "

# for i in ['1', '2', '4', '8', '14', '30B-A3', '38', '241B-A28']:
# folder = f'internvl35/think/internvl35-{i}B-think_wo_contextual.json'

# df_full = pl_full.to_pandas()
# df_full.to_json(eval_file, orient='records', indent=4)

# pl_full = pl.read_json(eval_file)
# print(pl_full)

# folder = "ovis25/ovis25-9B_w_contextual.json"
# orgg_file = f'/users/2/pyo00005/HOME/carto-reasoning/run_results/{folder}'
# eval_file = f'/users/2/pyo00005/HOME/carto-reasoning/evaluation_results/{folder}'
# pl_eval = pl.read_json(eval_file).unique()

# pl_full = pl_eval

pl_full = pl.read_json('/users/2/pyo00005/HOME/carto-reasoning/evaluation_results/gemini25pro/gemini_w_contextual.json')

overall = (pl_full.select(
    pl.col('correct').sum()
).item(0, 'correct') * 100 / pl_full.shape[0])

print(f"{overall:.2f}")

string_others = f"{overall:.2f} & "
string_spatial = f"{overall:.2f} & "

dict_files = {}

partition_items = ['spatial_relationship', 'answer_type', 'map_count']

for p in partition_items:
    for df in pl_full.partition_by(p):
        length_full = df.shape[0]

        # dict_files[df.item(0, p)] = df.select(
        #     pl.col('correct').sum()
        # ).item(0, 'correct')

        dict_files[df.item(0, p)] = df.select(
            pl.col('correct').sum()
        ).item(0, 'correct') * 100 / length_full


for i in ['Single', 'Multi', '', 'textual', 'distance', 'cardinal']:
    try:
        string_others += f"{dict_files[i]:.2f} & "
    except:
        string_others += "& "

string_others = string_others.strip().strip('&')

for i in ['Border', 'Distance', 'Equal', 'Intersect', 'Orientation', 'Within']:
    string_spatial += f"{dict_files[i]:.2f} & "

string_spatial = string_spatial.strip().strip('&')

print(string_spatial)
print(string_others)




# # get annnotator alingment
# pl_data = pl.read_json('/home/yaoyi/pyo00005/carto-reasoning/questions/response_full_d10.json')

# align3 = pl.read_csv('/home/yaoyi/pyo00005/carto-reasoning/questions/annotator_response/raw_csv/annot_align3.csv').group_by('question_ref').agg([pl.all()]).with_columns(
#     count = 3
# ).select(
#     pl.col(['question_ref', 'count'])
# )
# align2 = pl.read_csv('/home/yaoyi/pyo00005/carto-reasoning/questions/annotator_response/raw_csv/annot_align2.csv').group_by('question_ref').agg([pl.all()]).with_columns(
#     count = 2
# ).select(
#     pl.col(['question_ref', 'count'])
# )

# align_full = pl.concat(
#     [align3, align2],
#     how='diagonal'
# )

# pl_annot = pl.concat(
#     [pl_full, align_full],
#     how='align'
# )

# dict_results = {3: [],
#                 2: []}

# for i in pl_annot.partition_by('count'):

#     overall = (i.select(
#         pl.col('correct').sum()
#     ).item(0, 'correct') * 100 / pl_annot.shape[0])

#     dict_results[i.item(0, 'count')].append(overall)

# mode = 'w'
# for folder in [f'gemini25pro/gemini_{mode}_contextual.json', f'gpt5/gpt5_{mode}_contextual.json', f'claude-sonnet4/sonnet_{mode}_contextual.json',
#                f'llava-next/llava-next-110B_{mode}_contextual.json', f'glm45v/glm45v-108B_{mode}_contextual.json',
#                f'internvl3/internvl3-78B_{mode}_contextual.json', f'llava-ov/llava-ov-72B_{mode}_contextual.json', f'qwen25vl/qwen25vl-72B_{mode}_contextual.json', 
#                f'internvl35/no_think/internvl35-38B_{mode}_contextual.json', f'ovis2/ovis2-34B_{mode}_contextual.json',
#                f'ovis25/ovis25-9B-think_{mode}_contextual.json']:
#     try:
#         orgg_file = f'/users/2/pyo00005/HOME/carto-reasoning/run_results/{folder}'
#         eval_file = f'/users/2/pyo00005/HOME/carto-reasoning/evaluation_results/{folder}'
#         pl_eval = pl.read_json(eval_file).unique()
#         pl_full = pl_eval

#         pl_annot = pl.concat(
#             [pl_full, align_full],
#             how='align'
#         )

#         for i in pl_annot.partition_by('count'):

#             overall = (i.select(
#                 pl.col('correct').sum()
#             ).item(0, 'correct') * 100 / pl_annot.shape[0])

#             dict_results[i.item(0, 'count')].append(overall)

#             # print(i.item(0, 'count'), f"{overall:.2f}")

#     except: 
#         dict_results[3].append(0.00)
#         dict_results[2].append(0.00)
#         # print(folder)

# print(dict_results[3])
# print(dict_results[2])

# #     # print(pl_annot)
