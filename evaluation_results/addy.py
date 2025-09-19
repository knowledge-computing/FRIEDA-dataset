import polars as pl


intropart = ['glm45v/glm45v-108B', 
             'internvl3/internvl3-78B', 
             'internvl35/no_think/internvl35-38B',
             'llava-next/llava-next-110B',
             'llava-ov/llava-ov-72B',
             'ovis2/ovis2-34B',
             'ovis25/ovis25-9B',
             'qwen25vl/qwen25vl-72B']

avg = 0
for i in intropart:
    # with_case = pl.read_json(f'/users/2/pyo00005/HOME/carto-reasoning/evaluation_results/{i}_w_contextual.json').filter(
    #     pl.col('correct') == True
    # )['question_ref'].to_list()
    # witho_case = pl.read_json(f'/users/2/pyo00005/HOME/carto-reasoning/evaluation_results/{i}_wo_contextual.json').filter(
    #     pl.col('correct') == True
    # )['question_ref'].to_list()

    # print(len(with_case))
    # print(len(witho_case))

    # print(len(set(with_case) & set(witho_case))/len(with_case))

    # i = 'qwen25vl/qwen25vl-72B'
    with_case = pl.read_json(f'/users/2/pyo00005/HOME/carto-reasoning/evaluation_results/{i}_w_contextual.json').select(pl.col(['question_ref', 'spatial_relationship', 'answer_type', 'correct'])).rename({'correct': 'with_correct'})
    witho_case = pl.read_json(f'/users/2/pyo00005/HOME/carto-reasoning/evaluation_results/{i}_wo_contextual.json').select(pl.col(['question_ref', 'spatial_relationship', 'answer_type', 'correct'])).rename({'correct': 'witho_correct'})

    all_df = pl.concat(
        [with_case, witho_case],
        how='align'
    ).filter(
        pl.col('witho_correct') == pl.col('with_correct')
    )

    avg += all_df.shape[0]/500


print(avg / len(intropart))