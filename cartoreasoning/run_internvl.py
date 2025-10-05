import os
import argparse
from dotenv import load_dotenv
from typing import List, Dict
from tqdm import tqdm

import json
import pickle
import polars as pl

import torch
from transformers import AutoProcessor, AutoModelForImageTextToText
from transformers import BitsAndBytesConfig             # To reduce memory usage

# Memory management
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
torch.cuda.empty_cache()
torch.cuda.reset_peak_memory_stats()

max_memory = {
    0: "46GB",
    1: "46GB",
    2: "46GB",
    3: "46GB",
    # 4: "40GB",
    # 5: "40GB",
    # 6: "40GB",
    # 7: "40GB",
    # "cpu": "100GB"   # fallback for layers that don't fit
}

with open('./instruction.pkl', 'rb') as handle:
    instructions = pickle.load(handle)

R1_SYSTEM_PROMPT = """
You are an AI assistant that rigorously follows this response protocol:

1. First, conduct a detailed analysis of the question. Consider different angles, potential solutions, and reason through the problem step-by-step. Enclose this entire thinking process within <think> and </think> tags.

2. After the thinking section, provide a clear, concise, and direct answer to the user's question. Separate the answer from the think section with a newline.

Ensure that the thinking process is thorough but remains focused on the query. The final answer should be standalone and not reference the thinking section.
""".strip()

def check_exist(path_dir, bool_create=True):
    if os.path.exists(path_dir):
        return 1
    
    if bool_create:
        os.makedirs(path_dir)
        return 0
    
    return -1

def define_model(model_id:str,
                 use_flash: bool):
    
    # specify how to quantize the model
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )

    if use_flash:
        model = AutoModelForImageTextToText.from_pretrained(
            model_id,
            quantization_config=quantization_config,
            attn_implementation="flash_attention_2",
            device_map="auto",
            max_memory=max_memory,
            torch_dtype=torch.bfloat16,
            offload_folder="./offload",
            trust_remote_code=True
            )        # Only works under CUDA suppport
    else:
        # Slow processing
        model = AutoModelForImageTextToText.from_pretrained(
            model_id,
            quantization_config=quantization_config, 
            device_map="auto",
            torch_dtype=torch.bfloat16,
            trust_remote_code=True
        )

    model.config.use_cache = True
    model.config.paged_attention = True

    # model = torch.compile(model, mode="reduce-overhead", fullgraph=True)
    processor = AutoProcessor.from_pretrained(model_id)

    return model, processor

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
                     dict_im_data:Dict[str, str],
                     img_limit:int) -> list:

    content = []
    for idx, f in enumerate(file_list):
        if idx > img_limit:
            break
        content.append({"type": "image", "url": f"{dict_im_data[f]}"})

    content.append({"type": "text", "text": f"{question}"})

    return content

def respond_q(model,
              processor,
              dict_im_data:Dict[str, str],
              input_struct:List[dict],
              img_limit:int,
              allow_thinking:bool):

    messages = []
    for i in input_struct:
        content = make_chat_prompt(question=i['question_text'],
                                   file_list=i['image_lists'],
                                   dict_im_data=dict_im_data,
                                   img_limit=img_limit)
        conversation = [
            {   "role": "system",
                "content": [{"type": "text", "text": f"{instructions}"}],
            },
            {
                "role": "user",
                "content": content,
            }
        ]

        messages.append(conversation)

    inputs = processor.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        padding=True,
        padding_side="left",
        return_tensors="pt"
    ).to(model.device, torch.float16)

    with torch.inference_mode():
        if allow_thinking:
            model.system_message = R1_SYSTEM_PROMPT
            generate_ids = model.generate(**inputs, max_new_tokens=2048, do_sample=True, temperature=0.6)

        else:
            generate_ids = model.generate(**inputs, max_new_tokens=512, do_sample=False, use_cache=True)

    output_texts = processor.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
    list_output = []

    for response in output_texts:
        try:
            cleaned_response = response.split("Final answer:")[-1].strip()
            list_output.append(cleaned_response)
            
        except:
            list_output.append(response.split('assistant\n')[1].strip())

    return list_output

def main(model_name:str,
         question_path:str,
         image_folder:str,
         bool_distractor:bool,
         output_dir:str,
         cache_dir:str,
         use_flash:bool,
         allow_thinking:bool,
         batch_size:int,
         img_limit:int):
    
    param_size = model_name
    model_name = f"OpenGVLab/InternVL3_5-{param_size}-HF"
    
    # Create directory paths
    check_exist(output_dir)
    check_exist(cache_dir)

    cache_file = os.path.join(cache_dir, 'image_cache.pkl')
    # Load question JSON file
    with open(question_path, 'r') as file:
        data = json.load(file)
    
    # Uploading all images to path storage
    # if check_exist(cache_file, bool_create=False) == -1:
    if True:
        all_images = []
        for d in data:
            all_images.extend(d['image_urls'])
            all_images.extend(d['contextual_urls'])

        dict_im_data = upload_images(images=all_images,
                                     image_prefix=image_folder,
                                     save_cache=True,
                                     cache_dir=cache_dir)
    else:
        with open(cache_file, 'rb') as handle:
            dict_im_data = pickle.load(handle)

    # Load model and processor
    model, processor = define_model(model_id=model_name, use_flash=use_flash)

    # Set up questions dataframe
    pl_question = pl.read_json(question_path).with_columns(
        q_answered = pl.lit(False),
    ).with_columns(
        pl.when(bool_distractor)
        .then(pl.concat_list('contextual_urls', 'image_urls'))
        .otherwise(pl.col('image_urls')).alias('image_lists')
    )

    # Running inference in chunks (of 20) in case it crashse in middle
    pl_answered = pl.DataFrame()

    if bool_distractor:
        response_cache = os.path.join(cache_dir, f'response_cache-{param_size}_w_distractor.pkl')
    else:
        response_cache = os.path.join(cache_dir, f'response_cache-{param_size}_wo_distractor.pkl')
        
    if check_exist(response_cache, bool_create=False) == 1: 
        with open(response_cache, 'rb') as handle:
            pl_answered = pickle.load(handle)

        cache_length = pl_answered.shape[0]
        pl_question = pl_question[cache_length:]

    for i in tqdm(range(0, pl_question.height, batch_size)):
        chunk = pl_question.slice(i, batch_size)

        chunk = chunk.with_columns(
            tmp = pl.struct(pl.col(['question_text', 'image_lists']))
        )

        list_input = chunk['tmp'].to_list()
        list_output = respond_q(model=model, processor=processor,
                                input_struct=list_input,
                                dict_im_data=dict_im_data,
                                img_limit=img_limit,
                                allow_thinking=allow_thinking)
        
        chunk = chunk.with_columns(
            pl.col('q_answered').replace({False: True}),
            internvl35_response = pl.Series(list_output)
        ).drop(['tmp', 'image_lists'])

        pl_answered = pl.concat(
            [pl_answered, chunk],
            how='diagonal'
        )

        with open(response_cache, 'wb') as handle:
            pickle.dump(pl_answered, handle, protocol=pickle.HIGHEST_PROTOCOL)

        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

    # Saving as JSON with model name appended
    pd_answered = pl_answered.to_pandas()

    if bool_distractor:
        new_file_name = os.path.join(output_dir, f'internvl_w-{param_size}_contextual.json')
    else:
        new_file_name = os.path.join(output_dir, f'internvl_wo-{param_size}_contextual.json')
        
    pd_answered.to_json(new_file_name, orient='records', indent=4)

    # Removing response cache pickle file
    os.remove(response_cache)
    os.remove(cache_file)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Cartographical Reasoning Test')

    parser.add_argument('--model', '-m', default='OpenGVLab/InternVL3_5-38B-HF',
                        help='Model name/type')

    parser.add_argument('--param', '-p', required=True,
                        help='Model param FOR TESTING PURPOSE ONLY')

    parser.add_argument('--questions', '-q', default='/home/yaoyi/pyo00005/p2/carto-reasoning/questions/response_full_d10.json', 
                        help='Path to questions JSON file')

    # parser.add_argument('--images', '-im', default='/home/yaoyi/pyo00005/p2/carto-image',
    #                     help="Directory/link to reporsitory containing images")
    
    parser.add_argument('--distractor', '-d', action="store_true", 
                        help='Use distractor images')
   
    parser.add_argument('--output_dir', '-o', default='./responses',
                        help="Location to output files")
    
    parser.add_argument('--cache_dir', '-c', default='./',
                        help="Location to cache directory (cache for image names)")
    
    # parser.add_argument('--flash', action="store_true",
    #                     help="Use flash attention")

    parser.add_argument('--thinking', action="store_true",
                        help="Allow reasoning capability")
    
    parser.add_argument('--batch_size', type=int, default=12,
                        help="Batch size. Default is 1.")
    
    # parser.add_argument('--max_images', '-max', type=int, default=10,
    #                     help="FOR DEVELOPING TEST PURPOSE")
    
    args = parser.parse_args()
    
    # main(model=args.model,
    #      question_path=args.questions,
    #      image_folder=args.images,
    #      bool_distractor=args.distractor,
    #      output_dir=args.output_dir,
    #      cache_dir=args.cache_dir,
    #      use_flash=args.flash,
    #      batch_size=args.batch_size,
    #      img_limit=args.max_images)

    main(model_name=args.param,
        question_path=args.questions,
        image_folder='/home/yaoyi/pyo00005/p2/carto-image',
        bool_distractor=args.distractor,
        output_dir=args.output_dir,
        cache_dir=args.cache_dir,
        use_flash=True,
        allow_thinking=args.thinking,
        batch_size=args.batch_size,
        img_limit=20)