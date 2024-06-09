import argparse
import torch
import os
import json
from tqdm import tqdm
import shortuuid

from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava.conversation import conv_templates, SeparatorStyle
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import tokenizer_image_token, process_images, get_model_name_from_path
from torch.utils.data import Dataset, DataLoader

from PIL import Image
import math
import re

#### some bf16 tricks are added for evaluation
### check llava_arch and vq_clip.

def split_list(lst, n):
    """Split a list into n (roughly) equal-sized chunks"""
    chunk_size = math.ceil(len(lst) / n)  # integer division
    return [lst[i:i+chunk_size] for i in range(0, len(lst), chunk_size)]


def get_chunk(lst, n, k):
    chunks = split_list(lst, n)
    return chunks[k]


# Custom dataset class
class CustomDataset(Dataset):
    def __init__(self, questions, image_folder, tokenizer, image_processor, model_config):
        self.questions = questions
        self.image_folder = image_folder
        self.tokenizer = tokenizer
        self.image_processor = image_processor
        self.model_config = model_config

    def __getitem__(self, index):
        # line = self.questions[index]
        line = self.questions[index]
        # breakpoint()
        # print(line)
        image_file = line['file_name']
        qs = line['question']

        conv = conv_templates[args.conv_mode].copy()
        conv.append_message(conv.roles[0], qs)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()

        image = Image.open(os.path.join(self.image_folder, image_file)).convert('RGB')
        image_tensor = process_images([image], self.image_processor, self.model_config)[0]
        

        input_ids = tokenizer_image_token(prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt')

        return input_ids, image_tensor, image.size

    def __len__(self):
        return len(self.questions)


def collate_fn(batch):
    input_ids, image_tensors, image_sizes= zip(*batch)
    input_ids = torch.stack(input_ids, dim=0)
    image_tensors = torch.stack(image_tensors, dim=0)
    return input_ids, image_tensors, image_sizes


# DataLoader
def create_data_loader(questions, image_folder, tokenizer, image_processor, model_config, batch_size=1, num_workers=4):
    assert batch_size == 1, "batch_size must be 1"
    dataset = CustomDataset(questions, image_folder, tokenizer, image_processor, model_config)
    data_loader = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers, shuffle=False, collate_fn=collate_fn)
    return data_loader


def eval_model(args):
    # Model
    disable_torch_init()
    model_path = os.path.expanduser(args.model_path)
    my_vision_tower = os.path.expanduser(args.vision_tower)
    my_pretrain_mm_mlp_adapter = os.path.expanduser(args.pretrain_mm_mlp_adapter)   #####
    model_name = get_model_name_from_path(model_path)
    tokenizer, model, image_processor, context_len = load_pretrained_model(model_path, args.model_base, model_name, vision_tower = my_vision_tower, pretrain_mm_mlp_adapter=my_pretrain_mm_mlp_adapter) #####
    
    with open(args.question_file, 'r') as file:
        questions = json.load(file)
    # questions = []
    # ref_object = []
    # for item in data:
    #     questions.append(item['question'])
    #     ref_object.append(item['ref_list'])
    # questions = [json.loads(q) for q in open(os.path.expanduser(args.question_file), "r")]
    # questions = get_chunk(questions, args.num_chunks, args.chunk_idx)
    answers_file = os.path.expanduser(args.answers_file)
    os.makedirs(os.path.dirname(answers_file), exist_ok=True)
    ans_file = open(answers_file, "w")

    # if 'plain' in model_name and 'finetune' not in model_name.lower() and 'mmtag' not in args.conv_mode:
    #     args.conv_mode = args.conv_mode + '_mmtag'
    #     print(f'It seems that this is a plain model, but it is not using a mmtag prompt, auto switching to {args.conv_mode}.')
    #   awful strategy

    data_loader = create_data_loader(questions, args.image_folder, tokenizer, image_processor, model.config)

    for (input_ids, image_tensor, image_sizes), line in tqdm(zip(data_loader, questions), total=len(questions)):
        idx = line["image_id"]
        img_path = line['file_name']
        cur_prompt = '...' + line["question"][47:]
        ref = line['gt_box']

        input_ids = input_ids.to(device='cuda', non_blocking=True)

        with torch.inference_mode():
            output_ids = model.generate(
                input_ids,
                images=image_tensor.to(dtype=torch.float16, device='cuda', non_blocking=True),
                image_sizes=image_sizes,
                # do_sample=True if args.temperature > 0 else False,
                do_sample=False,
                temperature=args.temperature,
                top_p=args.top_p,
                num_beams=args.num_beams,
                max_new_tokens=args.max_new_tokens,
                use_cache=True)

        # breakpoint()

            embed,remain_map = model.model.vision_tower.forward(image_tensor.to(dtype=torch.float16, device='cuda', non_blocking=True))
            enc = model.model.vision_tower.vision_tower.visual_tokenizer
            img_map_t = enc.tokenize_features_mirage(embed)
            img_map = [num.item() for num in img_map_t[0]]
        # breakpoint()
            
            ref_seq = model.ref_sect(remain_map[0],[ref])###
            # gt_tokens = [model.generate_substituted_list(ref_seqs[i],feat_map[i],image_ids[i]) for i in range(len(feat_map))]
            gt_tokens_ = model.generate_substituted_list(ref_seq,remain_map[0],img_map_t[0])
            # print(ref_seq)
            # breakpoint()
            gt_tokens = [num.item() for num in gt_tokens_[0]]

            remain_map = [num.item() for num in remain_map[0]]
        # model.model.vision_tower.vision_tower.visual_tokenizer
        # model.model.vision_tower.forward(img_t) -> embed
        #embed[0].shape = [xx,1408]
        output_lang = tokenizer.decode([x for x in output_ids[0] if x<32000], skip_special_tokens=True)
        # outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
        predict_tokens = [x.item() for x in output_ids[0] if x>31999]
        # predict_tokens = [x.item() for x in predict_tokens]
        # breakpoint()
        # ans_id = shortuuid.uuid()
        ans_file.write(json.dumps({"image_id": idx,
                                   "file_name": img_path,
                                   "prompt": cur_prompt,
                                   "text": output_lang,
                                   "predict_tokens": predict_tokens,
                                   "gt_tokens": gt_tokens,
                                   "gt_answer": ref,
                                   "gt_map": ref_seq[0],
                                   "image_map": img_map,
                                   "remained_map": remain_map}) + "\n")
        # ans_file.flush()
    ans_file.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="/home/Clawmachine/checkpoints/llava-mirage-0426-id-dual-1e5")
    parser.add_argument("--model-base", type=str, default="/home/Clawmachine/checkpoints/llava-mirage-0426-id-dual-1e5")
    parser.add_argument("--vision-tower", type=str, default="/home/MaTianren/LaVIT-7B-v2")
    parser.add_argument("--pretrain-mm-mlp-adapter", type=str, default="")
    parser.add_argument("--image-folder", type=str, default="/home/Clawmachine/Dataset/datasets/MSCOCO2017/train2017")
    parser.add_argument("--question-file", type=str, default="/home/Clawmachine/playground/data/eval/GND_refcocos/Refcoco_Val_questions.json")
    parser.add_argument("--answers-file", type=str, default="/home/Clawmachine/my_evaluation/GND_0424_debug.json")
    parser.add_argument("--conv-mode", type=str, default="llava_v1")
    parser.add_argument("--num-chunks", type=int, default=1)
    parser.add_argument("--chunk-idx", type=int, default=0)
    parser.add_argument("--temperature", type=float, default=0)
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--num_beams", type=int, default=1)
    parser.add_argument("--max_new_tokens", type=int, default=128)
    args = parser.parse_args()

    eval_model(args)
