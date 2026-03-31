import torch
import clip
import pandas as pd
import numpy as np
import json
from Utils.WavCaps.retrieval.models.ase_model import ASE
from ruamel import yaml
from tqdm import tqdm


def main():
    # class name
    csv_path ="/Dataset/VGGSound/class-split/vggsound_w2v_class_names.csv"
    df = pd.read_csv(csv_path)
    dataset_classes = df['manual'].tolist()   #clip_class_name

    
    textual_description_path= \
        "/Dataset/vggsound_descriptions.json"
    with open(textual_description_path, 'r') as f:
        textual_clouds = json.load(f)

    save_path="/Dataset/vggsound_descriptions_llama_(500)_top50.pt"
    topk=50  # 'all' , 50

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    clip_model=get_clip_model(device)
    clap_model=get_clap_model(device)

    textual_cloud_embeddings={}
    with torch.no_grad():
        for class_name in tqdm(dataset_classes):

            textual_cloud_embeddings[class_name]={}

            single_cloud = textual_clouds[class_name]
            single_cloud_audio=single_cloud.get('audio',[]) #[500,]
            single_cloud_visual=single_cloud.get('visual',[])
            single_cloud_context=single_cloud.get('context',[])

            
            visual_tokenizer = clip.tokenize(single_cloud_visual, truncate=True).to(device)
            visual_embeddings = clip_model.encode_text(visual_tokenizer)  # [bs,512]
            visual_embeddings /= visual_embeddings.norm(dim=-1, keepdim=True)  

            class_name_tokenizer = clip.tokenize(class_name, truncate=True).to(device)
            class_name_embeding_clip = clip_model.encode_text(class_name_tokenizer)
            class_name_embeding_clip /= class_name_embeding_clip.norm(dim=-1, keepdim=True)
            if type(topk)==int:
                k_indices=simliarity_func(visual_embeddings,class_name_embeding_clip,topk=topk) 
                textual_cloud_embeddings[class_name]['visual']= visual_embeddings[k_indices].cpu()
            else :
                textual_cloud_embeddings[class_name]['visual'] = visual_embeddings.cpu()

            # context
            context_tokenizer = clip.tokenize(single_cloud_context, truncate=True).to(device)
            context_embeddings = clip_model.encode_text(context_tokenizer)
            context_embeddings /= context_embeddings.norm(dim=-1, keepdim=True)
            if type(topk) == int:
                k_indices=simliarity_func(context_embeddings,class_name_embeding_clip,topk=topk)
                textual_cloud_embeddings[class_name]['context']= context_embeddings[k_indices].cpu()
            else:
                textual_cloud_embeddings[class_name]['context'] = context_embeddings.cpu()

            
            audio_embeddings = clap_model.encode_text(single_cloud_audio)
            audio_embeddings /= audio_embeddings.norm(dim=-1, keepdim=True)
                #clap class name
            class_name_embedding_clap = clap_model.encode_text(class_name)
            class_name_embedding_clap /= class_name_embedding_clap.norm(dim=-1, keepdim=True)
            if type(topk) == int:
                k_indices=simliarity_func(audio_embeddings,class_name_embedding_clap,topk=topk)
                textual_cloud_embeddings[class_name]['audio']= audio_embeddings[k_indices].cpu()
            else:
                textual_cloud_embeddings[class_name]['audio'] = audio_embeddings.cpu()

    torch.save(textual_cloud_embeddings, save_path)

def simliarity_func(candiate_embedding, class_name_embedding,topk=20):
    scores=torch.mm(candiate_embedding, class_name_embedding.t())
    values,indices=torch.topk(scores.flatten(),topk)
    return indices.tolist()


def get_clip_model(device):
    clip_model, preprocess = clip.load("ViT-B/32", device=device)
    clip_model.eval()
    input_resolution = clip_model.visual.input_resolution
    context_length = clip_model.context_length
    vocab_size = clip_model.vocab_size
    print("Model parameters:", f"{np.sum([int(np.prod(p.shape)) for p in clip_model.parameters()]):,}")
    print("Input resolution:", input_resolution)
    print("Context length:", context_length)
    print("Vocab size:", vocab_size)

    return clip_model

def get_clap_model(device):
    clap_file_path="/WavCaps/retrieval/settings/inference.yaml"
    with open(clap_file_path,"r") as f:
        config = yaml.safe_load(f)
    wavcaps_model = ASE(config)
    wavcaps_model.to(device)
    cp_path = "/audio_encoder/HTSAT_BERT_zero_shot.pt"
    state_dict_key = 'model'
    cp = torch.load(cp_path)
    wavcaps_model.load_state_dict(cp[state_dict_key])
    wavcaps_model.eval()
    print("Model weights loaded from {}".format(cp_path))

    return wavcaps_model

if __name__ == "__main__":
    main()