# import polars as pl

# pl_data = pl.read_json('/home/yaoyi/pyo00005/carto-reasoning/questions/response_full_d10.json').with_columns(
#     pl.col('image_urls').list.first().str.split('/')
# ).with_columns(
#     source = pl.col('image_urls').list.first(),
#     file_name = pl.col('image_urls').list.get(1),
# ).filter(
#     ~pl.col('source').is_in(['aiib', 'NI43-101'])
# ).select(
#     pl.col(['source', 'file_name'])
# ).unique().sort('file_name').sort('source')

# pl_cont = pl.DataFrame(
#     {'source': ['EIS', 'seychelles', 'sg-ura', 'capetown', 'abu-dhabi', 'FEMA', 'ireland', 'national-parks', 'seattle-planning'],
#      'country': ['United States', 'Seychelles', 'Singapore', 'South Africa', 'United Arab Emirates', 'United States', 'Ireland', 'United States', 'United States']}
# )

# pl_data = pl.concat(
#     [pl_data, pl_cont],
#     how='align'
# )
# pl_files = pl.read_csv('./files.csv')

# pl_files = pl.concat(
#     [pl_files, pl_data],
#     how='diagonal'
# )

# pl_data = pl.read_json('/home/yaoyi/pyo00005/carto-reasoning/questions/response_full_d10.json').with_columns(
#     pl.col('image_urls').list.first().str.split('/')
# ).with_columns(
#     source = pl.col('image_urls').list.first(),
#     file_name = pl.col('image_urls').list.get(1),
# )

# pl_data = pl.concat(
#     [pl_data, pl_files],
#     how='align'
# )

# list_df = pl_data.partition_by('country')

# list_question_country = []
# list_questions_per = []

# for i in list_df:
#     list_question_country.append(i.item(0, 'country'))
#     list_questions_per.append(i.shape[0])

# print(list_question_country)
# print(list_questions_per)

# # pl_data.write_csv('./files.csv')

# # print(list_source) 
# # print(list_qref)

# # print(pl_data)
# # print(pl_files)

import polars as pl

pl_data = pl.read_json('/home/yaoyi/pyo00005/carto-reasoning/questions/response_full_d10.json').with_columns(
    count = pl.col('question_text').str.split(' ').list.len()
)

print(pl_data.select(pl.mean('count')))
print(pl_data.select(pl.max('count')))
# print(pl_data)