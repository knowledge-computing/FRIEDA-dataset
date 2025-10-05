import os
import argparse
from typing import List, Dict

from dotenv import load_dotenv

import json
import pickle
import polars as pl

import base64           # To load image on local
from openai import OpenAI


# Load environment variables from .env file
load_dotenv()
client = OpenAI(api_key=os.getenv('OPENAI_LAB_KEY'))
# client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))

with open('./instruction.pkl', 'rb') as handle:
    instructions = pickle.load(handle)
    instructions += "DO NOT use web search."    # Adding in case they use search
 

# Upload batch file to FileAPI
# batch_input_file = client.files.create(
#     file=open("./dummy.jsonl", "rb"),
#     purpose="batch"
# )

# # Create batch
# batch_input_file_id = batch_input_file.id

# job = client.batches.create(
#     input_file_id=batch_input_file_id,
#     endpoint="/v1/chat/completions",
#     completion_window="24h",
#     metadata={
#         "description": "nightly eval job"
#     }
# )

# print("Batch job id:", job.id)
    
# status checking
# batch = client.batches.retrieve("batch_68be1c3fa2588190b0877e32677893e0")
# print(batch)
# print()

# batch = client.batches.retrieve("batch_68be1ce68fa48190a659212804565955")
# print(batch)
# print()

# batch = client.batches.retrieve("batch_68be1ce68fa48190a659212804565955")
# print(batch)

# batch_68be20d8dc94819081bc3198770a7890

# file_response = client.files.content("file-1Y3nJJCPDoeeSwkPnJFh58")
# print(file_response.text)

# print(client.batches.list(limit=10))


####################################
    

def upload_images(images:List[str],
                 image_prefix:str=None,
                 save_cache:bool=True,
                 cache_dir:str=None):
    # Recommended for repetitively used files or large files
    dict_im_data = {}   # Storing mapping from image to API images

    for im_path in list(set(images)):
        if image_prefix:
            full_path = os.path.join(image_prefix, im_path)

        dict_im_data[im_path] = full_path

        if save_cache:
            with open(os.path.join(cache_dir, 'image_cache.pkl'), 'wb') as handle:
                pickle.dump(dict_im_data, handle, protocol=pickle.HIGHEST_PROTOCOL)

    return dict_im_data

def make_chat_prompt(question:str,
                     file_list:List[str],
                     dict_im_data:Dict[str, str],) -> list:
    
    content = [{"type":"text", "text": question}]

    imgs = [{"type":"image_url",
             "image_url":{"url":dict_im_data[fn]}} for fn in file_list]
        
    content.extend(imgs)

    return content

def gemini_async(model,
                 dict_im_data,
                 pl_question,
                 cache_dir):
    
    input_struct = pl_question.select(
        tmp = pl.struct(pl.all())
    )['tmp'].to_list()

    batch_file_name = "batch_requests_test5.jsonl"

    with open(os.path.join(cache_dir, batch_file_name), "w") as f:
        for i in input_struct:
            content = make_chat_prompt(i["question_text"], 
                                       i["image_lists"],
                                       dict_im_data)
            
            request = {
                "custom_id": i["question_ref"],
                "method":"POST",
                "url":"/v1/chat/completions",
                "body":{
                    "model":model,
                    "verbosity":"low",
                    "reasoning_effort":"high",
                    "messages":[{"role":"system",
                                 "content": [{"type": "text", "text": instructions}]},
                                {"role":"user",
                                 "content": content}],
                    "max_completion_tokens":25000
                }
            }

            f.write(json.dumps(request) + '\n')

    # Upload batch file to File API
    batch_input_file = client.files.create(
        file=open(os.path.join(cache_dir, batch_file_name), "rb"),
        purpose="batch"
    )
    batch_input_file_id = batch_input_file.id

    # Create batch job
    job = client.batches.create(
        input_file_id=batch_input_file_id,
        endpoint="/v1/chat/completions",
        completion_window="24h",
        metadata={
            "description": "Cartographical evaluation test"
        }
    )

    print("Batch job:", job.id)  # e.g., "batches/abcdef123"

    return job.id

model = 'gpt-5'
dict_im_data = {}
question_path = '/home/yaoyi/pyo00005/carto-reasoning/questions/response_full_d10.json'
image_folder = 'https://media.githubusercontent.com/media/YOO-uN-ee/carto-image/main/'
bool_distractor = True
cache_dir = '/home/yaoyi/pyo00005/carto-reasoning/cartoreasoning/batch_files/jsonized'

with open(question_path, 'r') as file:
    data = json.load(file)

all_images = []
for d in data:
    all_images.extend(d['image_urls'])
    all_images.extend(d['contextual_urls'])

dict_im_data = upload_images(images=all_images,
                             image_prefix=image_folder,
                             save_cache=True,
                             cache_dir=cache_dir)

pl_question = pl.read_json(question_path).with_columns(
        q_answered = pl.lit(False),
    ).with_columns(
        pl.when(bool_distractor)
        .then(pl.concat_list('contextual_urls', 'image_urls'))
        .otherwise(pl.col('image_urls')).alias('image_lists')
    )

print(gemini_async(model, dict_im_data, pl_question, cache_dir))
