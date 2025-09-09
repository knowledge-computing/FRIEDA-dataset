import pickle

# with open('/home/yaoyi/pyo00005/carto-reasoning/cartoreasoning/response_cache_wo_distractor.pkl', 'rb') as handle:
    # print(pickle.load(handle))


# with open('/home/yaoyi/pyo00005/carto-reasoning/cartoreasoning/instruction.pkl', 'rb') as handle:
#     print(pickle.load(handle))

# instruction = """
# Answer the question using the provided images. Follow the the following instructions.

# General:
# * If answer is a text from the map, copy it as it appears

# Numerical Answers
# * Include units as indicated on the map (Don't convert 1200m to 1.2km)
# * If both map frame and ruler scale is available, use the ruler scale
# * If question asks for an area, use {unit}^2
# * Use numerical values (e.g., 4 instead of four)

# Directional Answers:
# * Use 8 cardinal directions only: North, North East, East, South East, South, South West, West, North West
# * Write 'North' or 'South' before 'East' or 'West'
# * Notice that the north arrow compass do not always point upward

# Multi-Part Answers:
# * Separate with semicolon (;) (e.g., Zone A; Zone B)

# End your response with 'Final answer: '.
# """

# with open('/home/yaoyi/pyo00005/carto-reasoning/cartoreasoning/instruction-default.pkl', 'wb') as handle:
#     pickle.dump(instruction, handle, protocol=pickle.HIGHEST_PROTOCOL)

# instruction = """
# Answer the question using the provided images. Follow the the following instructions.

# General:
# * If answer is a text from the map, copy it as it appears

# Numerical Answers
# * Include units as indicated on the map (Don't convert 1200m to 1.2km)
# * If both map frame and ruler scale is available, use the ruler scale
# * If question asks for an area, use {unit}^2
# * Use numerical values (e.g., 4 instead of four)

# Directional Answers:
# * Use 8 cardinal directions only: North, North East, East, South East, South, South West, West, North West
# * Write 'North' or 'South' before 'East' or 'West'
# * Notice that the north arrow compass do not always point upward

# Multi-Part Answers:
# * Separate with semicolon (;) (e.g., Zone A; Zone B)

# End your response with 'Final answer: '. For example, 'Final answer: 5'.
# """

# with open('/home/yaoyi/pyo00005/carto-reasoning/cartoreasoning/instruction-eg.pkl', 'wb') as handle:
#     pickle.dump(instruction, handle, protocol=pickle.HIGHEST_PROTOCOL)

# # import os

# # image_folder = '/home/yaoyi/pyo00005/p2/carto-image'

import os
import polars as pl

def append_prefix(list_stuff):
    # os.path.getsize(file_path)
    return [os.path.getsize(os.path.join('/home/yaoyi/pyo00005/p2/carto-image', i)) for i in list_stuff]

pl_data = pl.read_json('/home/yaoyi/pyo00005/carto-reasoning/questions/response_full_d10.json').with_columns(
    pl.col('contextual_urls').map_elements(lambda x: append_prefix(x))
).with_columns(
    file_size = pl.col('contextual_urls').list.sum()
)

pl_compare = pl.read_json('/home/yaoyi/pyo00005/carto-reasoning/cartoreasoning/responses/response_mini.json').with_columns(
    pl.col('contextual_urls').map_elements(lambda x: append_prefix(x))
).with_columns(
    file_size = pl.col('contextual_urls').list.sum()
)

against_size = pl_data.select(pl.col('file_size').sum()).item(0, 'file_size')
mini_size = pl_compare.select(pl.col('file_size').sum()).item(0, 'file_size')

print(against_size/mini_size)

# print(pl_compare.select(pl.col('file_size').sum()))
