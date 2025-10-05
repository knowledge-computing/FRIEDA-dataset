import polars as pl
import pickle


with open('./all-agree-ref.pkl', 'rb') as handle:
    all_agree = pickle.load(handle)
with open('./partial-agree-ref.pkl', 'rb') as handle:
    partial_agree = pickle.load(handle)

for model_name in ['gemini', 'gpt5', 'internvl', 'next', 'onevision', 'ovis2', 'ovis25_think', 'ovis25', 'qwen']:

    print(model_name)
    file_name = f'/home/yaoyi/pyo00005/carto-reasoning/evaluation/evaluation_result/{model_name}_wo_contextual.json'
    pl_data = pl.read_json(file_name)

    full_data_len = pl_data.shape[0]

    print(f"""size: {pl_data.shape[0]} \tfull: {pl_data.select(
        pl.col('correct').sum()
    ).item(0, 'correct') * 100/pl_data.shape[0]}""")

    pl_all = pl_data.filter(
        pl.col('question_ref').is_in(all_agree)
    )

    print(f"""size: {pl_all.shape[0]} \tall: {pl_all.select(
        pl.col('correct').sum()
    ).item(0, 'correct') * 100/pl_all.shape[0]}""")

    pl_partial = pl_data.filter(
        pl.col('question_ref').is_in(partial_agree)
    )

    print(f"""size: {pl_partial.shape[0]} \tpartial: {pl_partial.select(
        pl.col('correct').sum()
    ).item(0, 'correct') * 100/pl_partial.shape[0]}""")
#correct

# print(pl_data/)

