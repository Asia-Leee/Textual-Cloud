import transformers
import torch
import pandas as pd
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm
import json
import random


def main():

    model_id = "/Model/Llama-3-1/"
    save_filename = \
        "/Dataset/vggsound_descriptions.json"

    csv_path = "/Dataset/VGGSound/class-split/vggsound_w2v_class_names.csv"
    df = pd.read_csv(csv_path)
    dataset_classes = df['manual'].tolist()

    pipeline = transformers.pipeline(
        "text-generation",
        model=model_id,
        model_kwargs={"torch_dtype": torch.bfloat16},
        device_map="auto",
    )

    prompt_templates_VGGSound = {
        "audio": "Generate exactly one (1) single sentence that is a direct auditory description of {concept}. Focus exclusively on the sound itself (e.g., its pitch, rhythm, volume, or texture) and do not describe any visual context.",
        "visual": "Generate exactly one (1) single sentence that is a direct visual description associated with {concept}. Focus exclusively on what you would see and do not describe any sounds, meanings, or functions.",
        "context": "Generate exactly one (1) single, factual sentence that is a direct factual definition of {concept} and its primary purpose. Do not use sensory language (how it sounds, looks, or feels)."
    }
    prompt_templates_UCF={
        "visual": "Generate exactly one (1) single sentence that directly describe the human action '{concept}'. Focus on body movements, visible objects, and the typical scene or background.",
        "audio": "Generate exactly one (1) single sentence that directly describe the typical sounds or auditory environment associated with the action '{concept}'. If the action is usually quiet, describe the ambient noise.",
        "context": "Generate exactly one (1) single sentence that directly explain what the human action '{concept}' involves and its primary purpose or context."
    }
    prompt_templates=prompt_templates_VGGSound


    
    loop=50

    all_results = {}
    with torch.no_grad():
        for dataset_class in tqdm(dataset_classes):
            single_class_result = {aspect: set() for aspect in prompt_templates.keys()}
            for aspect in prompt_templates:
                for i in range(loop): 
                    basic_prompt=dataset_class
                    instruction = prompt_templates[aspect].format(concept=basic_prompt)
                    
                    # instruction = prompt_templates[aspect].format(concept=basic_prompt) + f" {modifier}"
                    current_messages = [
                        {"role": "system",
                         "content": "You are an expert assistant in computer vision and multimodal learning."
                                    " You must follow the user's instructions precisely."
                                    "Never generate sentences longer than 15 words. "  
                                    "You must answer STRICTLY in English"},
                        {"role": "user", "content": f"{instruction}"}]

                    current_temp = 0.6 + (i / loop) * 0.5 
                    

                    output = pipeline(current_messages,
                                      max_new_tokens=20,  
                                      num_return_sequences=10,
                                      temperature=current_temp,
                                      do_sample=True,
                                      top_p=0.95,
                                      pad_token_id=pipeline.tokenizer.eos_token_id
                                      )
                    batch_texts=[]
                    for out in output:
                        generated_text = out["generated_text"][-1]['content']
                        batch_texts.append(generated_text)
                    single_class_result[aspect].update(batch_texts) 

            all_results[dataset_class] = {aspect: list(sentences) for aspect, sentences in single_class_result.items()}
    with open(save_filename, "w", encoding="utf-8") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=4)


if __name__ == "__main__":
    main()
