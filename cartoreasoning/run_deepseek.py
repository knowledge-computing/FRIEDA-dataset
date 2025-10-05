import torch
from transformers import AutoModelForCausalLM

from deepseek_vl2.models import DeepseekVLV2Processor, DeepseekVLV2ForCausalLM
from deepseek_vl2.utils.io import load_pil_images


# specify the path to the model
model_path = "deepseek-ai/deepseek-vl2"
vl_chat_processor: DeepseekVLV2Processor = DeepseekVLV2Processor.from_pretrained(model_path)
tokenizer = vl_chat_processor.tokenizer

# Leave ~2–3 GiB headroom per A100-40GB
GPU_BUDGET_GIB = 30
max_memory = {i: f"{GPU_BUDGET_GIB}GiB" for i in range(torch.cuda.device_count())}
max_memory["cpu"] = "200GiB"

vl_gpt: DeepseekVLV2ForCausalLM = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True,
      device_map="auto",
      offload_folder="./offload",
      max_memory=max_memory,)
vl_gpt = vl_gpt.eval()
vl_gpt.config.use_cache = False

# multiple images/interleaved image-text
conversation = [
    {
        "role": "<|User|>",
        "content": "This is image_1: <image>\n"
                   "This is image_2: <image>\n"
                   "This is image_3: <image>\n What are in the image?",
        "images": [
            "multi_image_1.jpeg",
            "multi_image_2.jpeg",
            "multi_image_3.jpeg",
        ],
    },
    {"role": "<|Assistant|>", "content": ""}
]

# load images and prepare for inputs
pil_images = load_pil_images(conversation)
prepare_inputs = vl_chat_processor(
    conversations=conversation,
    images=pil_images,
    force_batchify=True,
    system_prompt="You are a teacher teaching preschool children about colors. separate response for each with a semicolon ;"
)

with torch.no_grad():
    # Avoid custom prefilling / inputs_embeds. Just call generate on the processor outputs.
    outputs = vl_gpt.generate(
        input_ids=prepare_inputs.input_ids,                   # CPU tensors are fine
        attention_mask=prepare_inputs.attention_mask,
        images=prepare_inputs.images,
        images_seq_mask=prepare_inputs.images_seq_mask,
        images_spatial_crop=prepare_inputs.images_spatial_crop,

        max_new_tokens=256,         # keep modest initially
        do_sample=False,
        use_cache=False,            # align with config
        pad_token_id=tokenizer.eos_token_id,
        bos_token_id=tokenizer.bos_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )

# Decode only the generated suffix
gen_only = outputs[0][len(prepare_inputs.input_ids[0]):]
answer = tokenizer.decode(gen_only.cpu().tolist(), skip_special_tokens=False)
print(f"{prepare_inputs['sft_format'][0]}", answer)

# with torch.no_grad():
#     # run image encoder to get the image embeddings
#     inputs_embeds = vl_gpt.prepare_inputs_embeds(**prepare_inputs)

#     # incremental_prefilling when using 40G GPU for vl2-small
#     inputs_embeds, past_key_values = vl_gpt.incremental_prefilling(
#         input_ids=prepare_inputs.input_ids,
#         images=prepare_inputs.images,
#         images_seq_mask=prepare_inputs.images_seq_mask,
#         images_spatial_crop=prepare_inputs.images_spatial_crop,
#         attention_mask=prepare_inputs.attention_mask,
#         chunk_size=512 # prefilling size
#     )

#     # run the model to get the response
#     outputs = vl_gpt.generate(
#         inputs_embeds=inputs_embeds,
#         input_ids=prepare_inputs.input_ids,
#         images=prepare_inputs.images,
#         images_seq_mask=prepare_inputs.images_seq_mask,
#         images_spatial_crop=prepare_inputs.images_spatial_crop,
#         attention_mask=prepare_inputs.attention_mask,
#         past_key_values=past_key_values,

#         pad_token_id=tokenizer.eos_token_id,
#         bos_token_id=tokenizer.bos_token_id,
#         eos_token_id=tokenizer.eos_token_id,
#         max_new_tokens=512,

#         do_sample=False,
#         use_cache=True,
#     )

#     answer = tokenizer.decode(outputs[0][len(prepare_inputs.input_ids[0]):].cpu().tolist(), skip_special_tokens=False)

# print(f"{prepare_inputs['sft_format'][0]}", answer