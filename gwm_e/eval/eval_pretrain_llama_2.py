import sys

sys.path.append("./")
sys.path.append("./utils")
sys.path.append('/data/taofeng2/GWM/embedding_based/embedding_llm')
from utils import conversation as conversation_lib
import copy
import argparse
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "3"
import torch

import json
from tqdm import tqdm
import shortuuid
from typing import Dict, Optional, Sequence, List
from utils.constants import IGNORE_INDEX,GRAPH_TOKEN_INDEX, DEFAULT_GRAPH_TOKEN, DEFAULT_GRAPH_PAD_ID, DEFAULT_GRAPH_START_TOKEN, DEFAULT_GRAPH_END_TOKEN
from utils.conversation import conv_templates, SeparatorStyle
from model.builder_ import load_pretrained_model
from utils.utils import disable_torch_init, tokenizer_graph_token, get_model_name_from_path
from torch_geometric.utils import k_hop_subgraph, degree, remove_self_loops, add_self_loops
from torch_geometric.nn import MessagePassing
import math
import transformers

SMALL_DATASETS=["pubmed", "cora"]
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class MP(MessagePassing):
    def __init__(self):
        super().__init__(aggr='add')  # "Add" aggregation (Step 5).
    def message(self, x_j, norm):
        return norm.view(-1, 1) * x_j

def split_list(lst, n):
    """Split a list into n (roughly) equal-sized chunks"""
    chunk_size = math.ceil(len(lst) / n)  # integer division
    return [lst[i:i+chunk_size] for i in range(0, len(lst), chunk_size)]


# def get_chunk(lst, n, k):
#     chunks = split_list(lst, n)
#     return chunks[k]

def load_pretrain_embedding_graph(data_dir, pretrained_embedding_type):
    if pretrained_embedding_type == "simteg":
        simteg_sbert = torch.load(os.path.join(data_dir, "simteg_sbert_x.pt"))
        simteg_roberta = torch.load(os.path.join(data_dir, "simteg_roberta_x.pt"))
        simteg_e5 = torch.load(os.path.join(data_dir, "simteg_e5_x.pt"))
        pretrained_emb = torch.concat([simteg_sbert, simteg_roberta, simteg_e5], dim=-1)
    else:
        pretrained_emb = torch.load(os.path.join(data_dir, f"{pretrained_embedding_type}_x.pt"))
    return pretrained_emb

def load_pretrain_embedding_hop(data_dir, pretrained_embedding_type, hop, mask):
    if pretrained_embedding_type == "simteg":
        simteg_sbert=[torch.load(os.path.join(data_dir, f"simteg_sbert_x.pt"))[mask]] + [torch.load(os.path.join(data_dir, f"simteg_sbert_{i}hop_x.pt"))[mask] for i in range(1, hop + 1)]
        simteg_roberta = [torch.load(os.path.join(data_dir, f"simteg_roberta_x.pt"))[mask]] + [torch.load(os.path.join(data_dir, f"simteg_roberta_{i}hop_x.pt"))[mask] for i in range(1, hop + 1)]
        simteg_e5 = [torch.load(os.path.join(data_dir, f"simteg_e5_x.pt"))[mask]] + [torch.load(os.path.join(data_dir, f"simteg_e5_{i}hop_x.pt"))[mask] for i in range(1, hop + 1)]
        pretrained_embs = [torch.cat([simteg_sbert[i], simteg_roberta[i], simteg_e5[i]], dim=-1) for i in range(hop + 1)]
    else:
        pretrained_embs = [torch.load(os.path.join(data_dir, f"{pretrained_embedding_type}_x.pt"))[mask]]+  [torch.load(os.path.join(data_dir, f"{pretrained_embedding_type}_{i}hop_x.pt"))[mask] for i in range(1, hop+1)]

    return pretrained_embs

def load_pretrain_embedding_hop_lp(data_dir, pretrained_embedding_type, hop):
    mask = torch.load(os.path.join(data_dir, f"no_test_link_mask.pt"))
    if pretrained_embedding_type == "simteg":
        simteg_sbert=[torch.load(os.path.join(data_dir, f"simteg_sbert_x.pt"))[mask]] + [torch.load(os.path.join(data_dir, f"simteg_sbert_{i}hop_x_notestlink.pt")) for i in range(1, hop + 1)]
        simteg_roberta = [torch.load(os.path.join(data_dir, f"simteg_roberta_x.pt"))[mask]] + [torch.load(os.path.join(data_dir, f"simteg_roberta_{i}hop_x_notestlink.pt")) for i in range(1, hop + 1)]
        simteg_e5 = [torch.load(os.path.join(data_dir, f"simteg_e5_x.pt"))[mask]] + [torch.load(os.path.join(data_dir, f"simteg_e5_{i}hop_x_notestlink.pt")) for i in range(1, hop + 1)]
        pretrained_embs = [torch.cat([simteg_sbert[i], simteg_roberta[i], simteg_e5[i]], dim=-1) for i in range(hop + 1)]
    else:
        pretrained_embs = [torch.load(os.path.join(data_dir, f"{pretrained_embedding_type}_x.pt"))[mask]]+  [torch.load(os.path.join(data_dir, f"{pretrained_embedding_type}_{i}hop_x_notestlink.pt")) for i in range(1, hop+1)]

    return pretrained_embs, mask


def preprocess_llama3(
        sources,
        tokenizer: transformers.PreTrainedTokenizer,
        has_graph: bool = False,
        max_len=2048,
        system_message: str = "You are a helpful multi-modal assistant. You are able to understand the text content, image content, and table content that the user provides, and assist the user with a variety of tasks using natural language.",
) -> Dict:
    # roles = {"human": "<|start_header_id|>user<|end_header_id|>", "gpt": "<|start_header_id|>assistant<|end_header_id|>"}
    roles = {"human": "user", "gpt": "assistant"}

    # Add image tokens to tokenizer as a special tokens
    # Use a deepcopy of tokenizer so that we don't modify on the tokenizer
    tokenizer = copy.deepcopy(tokenizer)
    # When there is actually an image, we add the image tokens as a special token
    if has_graph:
        tokenizer.add_tokens(["<graph>"], special_tokens=True)
    graph_token_index = tokenizer.convert_tokens_to_ids("<graph>")
    bos_token_id = tokenizer.convert_tokens_to_ids("<|begin_of_text|>")
    start_header_id = tokenizer.convert_tokens_to_ids("<|start_header_id|>")
    end_header_id = tokenizer.convert_tokens_to_ids("<|end_header_id|>")
    eot_id = tokenizer.convert_tokens_to_ids("<|eot_id|>")

    unmask_tokens = ["<|begin_of_text|>", "<|start_header_id|>", "<|end_header_id|>", "<|eot_id|>", "\n\n"]
    unmask_tokens_idx = [tokenizer.convert_tokens_to_ids(tok) for tok in unmask_tokens]

    # After update, calling tokenizer of llama3 will
    # auto add bos id for the tokens. ヽ(｀⌒´)ﾉ
    def safe_tokenizer_llama3(text):
        input_ids = tokenizer(text).input_ids
        if input_ids[0] == bos_token_id:
            input_ids = input_ids[1:]
        return input_ids

    nl_tokens = tokenizer.convert_tokens_to_ids("\n\n")
    # Apply prompt templates
    input_ids, targets = [], []
    for i, source in enumerate(sources):
        if roles[source[0]["from"]] != roles["human"]:
            source = source[1:]

        input_id, target = [], []

        # New version, use apply chat template
        # Build system message for each sentence
        input_id += tokenizer.apply_chat_template([{"role": "system", "content": system_message}])
        target += [IGNORE_INDEX] * len(input_id)

        for conv in source:
            # Make sure llava data can load
            try:
                role = conv["role"]
                content = conv["content"]
            except:
                role = conv["from"]
                content = conv["value"]

            role = roles.get(role, role)

            conv = [{"role": role, "content": content}]
            # First is bos token we don't need here
            encode_id = tokenizer.apply_chat_template(conv)[1:]
            input_id += encode_id
            if role in ["user", "system"]:
                target += [IGNORE_INDEX] * len(encode_id)
            else:
                target += encode_id

        assert len(input_id) == len(target), f"{len(input_id)} != {len(target)}"
        for idx, encode_id in enumerate(input_id):
            if encode_id in unmask_tokens_idx:
                target[idx] = encode_id
            if encode_id == graph_token_index:
                input_id[idx] = GRAPH_TOKEN_INDEX
        input_ids.append(input_id)
        targets.append(target)
    input_ids = torch.tensor(input_ids, dtype=torch.long)
    targets = torch.tensor(targets, dtype=torch.long)

    return input_ids  # tensor(bs x seq_len)

# def preprocess_qwen(sources, tokenizer: transformers.PreTrainedTokenizer, has_image: bool = False, max_len=2048, system_message: str = "You are a helpful assistant.") -> Dict:
#     roles = {"human": "<|im_start|>user", "gpt": "<|im_start|>assistant"}
#
#     im_start, im_end = tokenizer.additional_special_tokens_ids
#     nl_tokens = tokenizer("\n").input_ids
#     _system = tokenizer("system").input_ids + nl_tokens
#     _user = tokenizer("user").input_ids + nl_tokens
#     _assistant = tokenizer("assistant").input_ids + nl_tokens
#
#     # Apply prompt templates
#     input_ids, targets = [], []
#
#     source = sources
#     if roles[source[0]["from"]] != roles["human"]:
#         source = source[1:]
#
#     input_id, target = [], []
#     system = [im_start] + _system + tokenizer(system_message).input_ids + [im_end] + nl_tokens
#     input_id += system
#     target += [im_start] + [IGNORE_INDEX] * (len(system) - 3) + [im_end] + nl_tokens
#     assert len(input_id) == len(target)
#     for j, sentence in enumerate(source):
#         role = roles[sentence["from"]]
#         if has_image and sentence["value"] is not None and "<image>" in sentence["value"]:
#             num_image = len(re.findall(DEFAULT_IMAGE_TOKEN, sentence["value"]))
#             texts = sentence["value"].split('<image>')
#             _input_id = tokenizer(role).input_ids + nl_tokens
#             for i,text in enumerate(texts):
#                 _input_id += tokenizer(text).input_ids
#                 if i<len(texts)-1:
#                     _input_id += [IMAGE_TOKEN_INDEX] + nl_tokens
#             _input_id += [im_end] + nl_tokens
#             assert sum([i==IMAGE_TOKEN_INDEX for i in _input_id])==num_image
#         else:
#             if sentence["value"] is None:
#                 _input_id = tokenizer(role).input_ids + nl_tokens
#             else:
#                 _input_id = tokenizer(role).input_ids + nl_tokens + tokenizer(sentence["value"]).input_ids + [im_end] + nl_tokens
#         input_id += _input_id
#         if role == "<|im_start|>user":
#             _target = [im_start] + [IGNORE_INDEX] * (len(_input_id) - 3) + [im_end] + nl_tokens
#         elif role == "<|im_start|>assistant":
#             _target = [im_start] + [IGNORE_INDEX] * len(tokenizer(role).input_ids) + _input_id[len(tokenizer(role).input_ids) + 1 : -2] + [im_end] + nl_tokens
#         else:
#             raise NotImplementedError
#         target += _target
#
#     input_ids.append(input_id)
#     targets.append(target)
#     input_ids = torch.tensor(input_ids, dtype=torch.long)
#     targets = torch.tensor(targets, dtype=torch.long)
#     return input_ids


def preprocess_llama_2(
    sources,
    tokenizer: transformers.PreTrainedTokenizer,
    has_graph: bool = False
) -> Dict:
    conv = conversation_lib.default_conversation.copy()
    roles = {"human": conv.roles[0], "gpt": conv.roles[1]}

    # Apply prompt templates
    conversations = []
    for i, source in enumerate(sources):
        if roles[source[0]["from"]] != conv.roles[0]:
            # Skip the first one if it is not from human
            source = source[1:]

        conv.messages = []
        for j, sentence in enumerate(source):
            role = roles[sentence["from"]]
            assert role == conv.roles[j % 2], f"{i}"
            conv.append_message(role, sentence["value"])
        conversations.append(conv.get_prompt())

    # Tokenize conversations

    if has_graph:
        input_ids = torch.stack([tokenizer_graph_token(prompt, tokenizer, return_tensors='pt') for prompt in conversations], dim=0)
    else:
        input_ids = tokenizer(
            conversations,
            return_tensors="pt",
            padding="longest",
            max_length=tokenizer.model_max_length,
            truncation=True,
        ).input_ids

    return input_ids

def eval_model(args):
    # Model
    disable_torch_init()

    model_path = os.path.expanduser(args.model_path)
    model_name = get_model_name_from_path(model_path)
    print(f"Loaded from {model_path}. Model Base: {args.model_base}")
    tokenizer, model, context_len = load_pretrained_model(model_path, args.model_base, model_name,
                                                          cache_dir=args.cache_dir)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id  # FIXME: this could only be triggered for llama3 model.
        # tokenizer.pad_token_id = 0  # This gets the best result. Don't know why.
    tokenizer.pad_token_id = 0
    model = model.to(torch.float16).cuda()
    meta_data_path = '/data/taofeng2/GWM/multi_modal_dataset/'
    # data_dir=os.path.expanduser(args.data_dir)
    all_token_list=[]
    for use_dataset in args.dataset:
        if use_dataset =="multi_modal_paper":
            data_dir = meta_data_path + "multi_modal_paper/"
        elif use_dataset == "Amazon_Rec_Baby":
            data_dir = meta_data_path + "Amazon_Rec_Baby/"
        elif use_dataset == "Amazon_Rec_Sports":
            data_dir = meta_data_path + "Amazon_Rec_Sports/"
        elif use_dataset == "Amazon_Rec_Clothing":
            data_dir = meta_data_path + "Amazon_Rec_Clothing/"
        elif use_dataset =="RAG":
            data_dir = meta_data_path + "RAG/"
        elif use_dataset =="cora_node" :
            data_dir = meta_data_path + "traditional_GNN/"
        elif use_dataset =="cora_edge" :
            data_dir = meta_data_path + "traditional_GNN/"
        elif use_dataset =="pubmed_node":
            data_dir = meta_data_path + "traditional_GNN/"
        elif use_dataset =="pubmed_edge":
            data_dir = meta_data_path + "traditional_GNN/"
        elif use_dataset =="AgentClinic":
            data_dir = meta_data_path + "AgentClinic/"
        elif use_dataset =="HIV":
            data_dir = meta_data_path + "traditional_GNN/"
        elif use_dataset =="Goodreads":
            data_dir = meta_data_path + "Goodreads/"
        elif use_dataset =="Alfworld":
            data_dir = meta_data_path + "Alfworld/"
        else:
            print(f"{use_dataset} not exists")
            raise ValueError
        if use_dataset in ["cora_node", "cora_edge", "pubmed_node", "pubmed_edge","HIV","Goodreads","Alfworld"]:
            prefix = use_dataset.split('_')[0]
            data_dir_ = data_dir + prefix + '_'
            graph_embedding_path = data_dir_ + 'multi_hop_graph_embedding.pt'
        else:
            graph_embedding_path = data_dir + 'multi_hop_graph_embedding.pt'
        pretrained_embs = torch.load(graph_embedding_path)[:args.use_hop]

        if use_dataset in ["cora_node", "pubmed_node","HIV","Goodreads","Alfworld"]:
            if "cora_node" in use_dataset:
                data_path = os.path.join(data_dir,
                                         f"cora_test_node_data.jsonl")
            elif "HIV" in use_dataset:
                data_path = os.path.join(data_dir,
                                         f"HIV_test_node_data.jsonl")
            elif "Goodreads" in use_dataset:
                data_path = os.path.join(data_dir,
                                         f"Goodreads_test_node_data.jsonl")
            elif "Alfworld" in use_dataset:
                data_path = os.path.join(data_dir,
                                         f"Alfworld_test_node_data.jsonl")
            else:
                data_path = os.path.join(data_dir,
                                         f"pubmed_test_node_data.jsonl")

        elif use_dataset in ["cora_edge", "pubmed_edge"]:
            if "cora_edge" in use_dataset:
                data_path = os.path.join(data_dir,
                                         f"cora_test_edge_data.jsonl")
            else:
                data_path = os.path.join(data_dir,
                                         f"pubmed_test_edge_data.jsonl")
        else:
            data_path = os.path.join(data_dir,
                                     f"test_edge_data.jsonl")


        prompt_file=data_path
        print(f"Load from {prompt_file}\n")
        lines = open(prompt_file, "r").readlines()

        answers_file = os.path.expanduser('./hopimage-' + str(args.use_hop) + '/' + use_dataset + '_' + args.answers_file)
        os.makedirs(os.path.dirname(answers_file), exist_ok=True)
        if "tmp" not in args.answers_file and os.path.exists(answers_file):
            line_number = len(open(answers_file, 'r').readlines())
            print(f"{args.answers_file} already exists! it has {line_number} lines!!")
            if line_number >= len(lines):
                return
            lines = lines[line_number:]
            ans_file = open(answers_file, "a")
        else:
            ans_file = open(answers_file, "w")

        questions = [json.loads(q) for q in lines][:5]
        for line in tqdm(questions):
            idx = line["id"]
            args.conv_mode = "llava_llama_2"
            if use_dataset in ["Amazon_Rec_Baby", "Amazon_Rec_Sports", "Amazon_Rec_Clothing"]:
                line["conversations"][0][
                    'value'] = f"This is a recommendation task. Given two node-centered subgraphs: {DEFAULT_GRAPH_TOKEN} and {DEFAULT_GRAPH_TOKEN}, we need to predict whether these two nodes connect with each other. Please tell me whether two center nodes in the subgraphs should connect to each other."
            if use_dataset in ["multi_modal_paper"]:
                line["conversations"][0][
                    'value'] = f"This is a multimodal information matching task for predicting article content. Given two node-centered subgraphs: {DEFAULT_GRAPH_TOKEN} and {DEFAULT_GRAPH_TOKEN}, we need to predict whether these two nodes connect with each other. Please tell me whether two center nodes in the subgraphs should connect to each other."
            if use_dataset in ["RAG"]:
                user_query = line["conversations"][0]['value']
                line["conversations"][0][
                    'value'] = f"This is a Retrieval-Augmented Generation task for improving response quality in dialogue systems. Given a user query: {user_query} and a set of retrieved documents: {DEFAULT_GRAPH_TOKEN}{DEFAULT_GRAPH_TOKEN}{DEFAULT_GRAPH_TOKEN}{DEFAULT_GRAPH_TOKEN}{DEFAULT_GRAPH_TOKEN}, the goal is to generate a coherent and contextually relevant response. Please generate a response that integrates information from the retrieved documents to accurately address the user's query."
                line["conversations"][1]['value'] = line["conversations"][1]['value'].split("is")[-1]
            if use_dataset in ["cora_node", "cora_edge", "pubmed_node", "pubmed_edge"]:
                if "cora_node" in use_dataset:
                    line["conversations"][0][
                        'value'] = f"Given a node-centered graph: {DEFAULT_GRAPH_TOKEN}, each node represents a paper, we need to classify the center node into 7 classes: Case_Based, Genetic_Algorithms, Neural_Networks, Probabilistic_Methods, Reinforcement_Learning, Rule_Learning, Theory, please tell me which class the center node belongs to?"
                elif "cora_edge" in use_dataset:
                    line["conversations"][0][
                        'value'] = f"Given two node-centered subgraphs: {DEFAULT_GRAPH_TOKEN} and {DEFAULT_GRAPH_TOKEN}, we need to predict whether these two nodes connect with each other. Please tell me whether two center nodes in the subgraphs should connect to each other."
                elif "pubmed_node" in use_dataset:
                    line["conversations"][0][
                        'value'] = f"Given a node-centered graph: {DEFAULT_GRAPH_TOKEN}, each node represents a paper about Diabetes, we need to classify the center node into 3 classes: Diabetes Mellitus Experimental, Diabetes Mellitus Type1, Diabetes Mellitus Type2, please tell me which class the center node belongs to?"
                else:
                    line["conversations"][0][
                        'value'] = f"Given two node-centered subgraphs: {DEFAULT_GRAPH_TOKEN} and {DEFAULT_GRAPH_TOKEN}, we need to predict whether these two nodes connect with each other. Please tell me whether two center nodes in the subgraphs should connect to each other."
            if "AgentClinic" in use_dataset:
                user_query = line["conversations"][0]['value']
                line["conversations"][0][
                    'value'] = f"This is a Multi-Agent Collaborative Generation task for creating dynamic conversational interactions. Given a user query: {user_query} and three distinct agents: {DEFAULT_GRAPH_TOKEN}{DEFAULT_GRAPH_TOKEN}{DEFAULT_GRAPH_TOKEN}, Please generate a well-rounded response to the user's question."
            qs = line["conversations"][0]["value"]
            cur_prompt=line["conversations"][0][
                'value']
            conv = conv_templates[args.conv_mode].copy()
            # conv.stop_token_ids=[128009]
            conv.append_message(conv.roles[0], qs)
            # conv.tokenizer = tokenizer
            conv.append_message(conv.roles[1], None)
            graph_emb=pretrained_embs[:,idx,:]
            if graph_emb.dim() > 3:
                graph_emb = graph_emb.squeeze()
                if args.use_hop == 1:
                    graph_emb = graph_emb.unsqueeze(0)
            graph_emb = torch.transpose(graph_emb, 0, 1)
            ## only text
            # graph_emb[:, :, :768] = 0
            ## only image
            # graph_emb[:, :, 512:] = 0
            # input_ids = preprocess_llama3([[line["conversations"][0],{'from': 'gpt','value':''}]], tokenizer, has_graph=True).cuda()
            prompt = conv.get_prompt()
            # input_ids=preprocess_llama_2(sources=[line["conversations"]],tokenizer=tokenizer,has_graph=True).cuda()
            input_ids = tokenizer_graph_token(prompt, tokenizer, GRAPH_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda()
            # if not isinstance(line['graph'][0], list):
            #     line['graph'] = [line['graph']]

            stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2


            with torch.inference_mode():
                output_ids = model.generate(
                    input_ids,
                    graph_emb=graph_emb.half().cuda(),
                    graph=None,
                    do_sample=True,
                    temperature=args.temperature,
                    top_p=args.top_p,
                    num_beams=args.num_beams,
                    # no_repeat_ngram_size=3,
                    max_new_tokens=1024,
                    use_cache=True)

            input_token_len = input_ids.shape[1]
            all_token_list.append(input_token_len)

            n_diff_input_output = (input_ids != output_ids[:, :input_token_len]).sum().item()
            if n_diff_input_output > 0:
                print(f'[Warning] {n_diff_input_output} output_ids are not the same as the input_ids')
            outputs = tokenizer.batch_decode(output_ids[:, input_token_len:], skip_special_tokens=True)[0]
            print(outputs)
            outputs = outputs.strip()
            if outputs.endswith(stop_str):
                outputs = outputs[:-len(stop_str)]
            outputs = outputs.strip()
            a=1
            # print(outputs)


            ans_id = shortuuid.uuid()
            ans_file.write(json.dumps({"question_id": idx,
                                       "prompt": cur_prompt,
                                       "graph": line['graph'],
                                       "text": outputs,
                                       "gt":line["conversations"][1]['value'],
                                       "answer_id": ans_id}) + "\n")
            ans_file.flush()
        ans_file.close()
    print("average_token:",sum(all_token_list) / len(all_token_list))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default="/data/taofeng2/GWM/embedding_based/embedding_llm/train/checkpoints_all_tasks_hop5/llaga-llama-2-checkpoint-183")
    parser.add_argument("--model_base", type=str, default="meta-llama/Llama-2-7b-hf") # meta-llama/Meta-Llama-3-8B meta-llama/Llama-2-7b-hf
    # parser.add_argument("--data_dir", type=str, default=None)
    parser.add_argument("--pretrained_embedding_type", type=str, default="sbert")
    parser.add_argument("--use_hop", type=int, default=5)
    parser.add_argument("--sample_neighbor_size", type=int, default=5)
    parser.add_argument("--answers_file", type=str, default="answer.jsonl")
    parser.add_argument("--conv_mode", type=str, default="llaga_llama_2")
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--num_beams", type=int, default=1)
    parser.add_argument("--prompt", type=str, default=None)
    parser.add_argument("--start", type=int, default=-1)
    parser.add_argument("--end", type=int, default=-1)
    parser.add_argument("--test_path", type=str, default=None)
    parser.add_argument("--mm_use_graph_start_end",default=False, action="store_true")
    parser.add_argument("--task", type=str, default="lp")
    parser.add_argument("--dataset", type=List[str], default=["Goodreads","Alfworld","HIV","RAG","multi_modal_paper", "Amazon_Rec_Baby", "Amazon_Rec_Sports", "Amazon_Rec_Clothing","cora_node","cora_edge","pubmed_node","pubmed_edge","AgentClinic"]) # ["Goodreads","Alfworld","HIV","RAG","multi_modal_paper", "Amazon_Rec_Baby", "Amazon_Rec_Sports", "Amazon_Rec_Clothing","cora_node","cora_edge","pubmed_node","pubmed_edge","AgentClinic"]
    parser.add_argument("--cache_dir", type=str, default="../../checkpoint")
    parser.add_argument("--template", type=str, default="HO")
    args = parser.parse_args()

    eval_model(args)