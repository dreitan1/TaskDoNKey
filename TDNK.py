import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from tqdm import tqdm
import sys
import numpy as np
import os
import json
from huggingface_hub import login
import matplotlib.pyplot as plt
import logging

from secret import API_KEY

class TaskDoNKey():
    def __init__(self, model_type='deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B', max_length=500, device=None):
        login(API_KEY)
        logging.disable(logging.WARNING)

        if device is None:
            device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

        self.model = AutoModelForCausalLM.from_pretrained(model_type).to(device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_type)

        self.max_length = max_length

        self.pipe = pipeline('text-generation', model=self.model, tokenizer=self.tokenizer, device=device, torch_dtype=torch.bfloat16)

        self.load_templates()

    
    def load_templates(self):
        with open('template1.txt', 'r') as f:
            self.template1 = f.read()
        
        with open('template2.txt', 'r') as f:
            self.template2 = f.read()

        with open('template3.txt', 'r') as f:
            self.template3 = f.read()

    
    def load_world_state(self, ws_file='world-state.txt'):
        with open(ws_file, 'r') as f:
            self.world_state = f.read()


    # Stage 1: Highlight the keywords in the instruction and return the highlighted instruction and the keywords
    def highlight(self, instruct):
        prompt = self.template1.replace('{prompt}', instruct)

        think = self.pipe(prompt, max_new_tokens=self.max_length, num_return_sequences=1, truncation=True, do_sample=True, temperature=0.01)[0]["generated_text"]

        if 'Final Answer: ' not in think.replace(prompt, ''):
            output = self.pipe(f"{think}\n\nFinal Answer: ", max_new_tokens=self.max_length, num_return_sequences=1, truncation=True, do_sample=True, temperature=0.01)[0]["generated_text"]

            highlighted = output.replace(prompt, '').replace(think, '').replace('\nEND OF SEQUENCE', '')
        else:
            highlighted = think.replace(prompt, '').split('Final Answer: ')[1].split('\n')[0]

        keywords = [highlighted.split('/')[i] for i in range(len(highlighted.split('/'))) if i % 2 != 0]

        return highlighted, keywords


    # Stage 2: Pick viable options from the world state for each keyword in instruction.
    # Run algorithm for k passes, to ensure LLM can viably select all valid options
    def select_options(self, keywords, instruct, speaker, passes=3):
        world_state = self.world_state

        prompt = self.template2.replace('{world state}', world_state).replace('{instruct}', f"{speaker}: {instruct}")

        replacements = {}

        for k in keywords:
            for _ in range(passes):
                p = prompt.replace('{word}', k)
                output = self.pipe(p, max_new_tokens=self.max_length, num_return_sequences=1, truncation=True, do_sample=True, temperature=0.25)[0]["generated_text"]

                # output = pipe(output + "\n\nFinal Answer: ", max_new_tokens=max_length, num_return_sequences=1, truncation=True, do_sample=True, temperature=0.01)[0]["generated_text"]

                # print(output.replace(p, ''))
                # print()

                try:
                    options = output.replace(p, '').split('Final Answer:\n')[1]

                    options = [x.split('. ')[1] for x in options.split('\n')]

                    replacements[k] = replacements.get(k, [])
                    replacements[k].append(options)
                except:
                    pass
            try:
                replacements[k] = [x for xs in replacements[k] for x in xs]
            except:
                replacements[k] = []

        world_state = world_state.split('\n')

        # Remove unwanted results based on parameters
        for k, v in replacements.items():
            replacements[k] = list(set(v))                        # Duplicates due to multiple passes
            replacements[k] = [x for x in v if x in world_state]  # Halucinations

        return replacements


    # Stage 3: 
    def disamgibuate(self, options, cache, highlighted, speaker):
        prompt = self.template3.replace("{speaker}", speaker)

        final_instruct = highlighted
        selected = []

        for k, v in options.items():
            opts = [o + f": {cache.get(o, 0)}" for o in v]
            opts = "\n".join(opts)
            p = prompt.replace("{keyword}", k).replace("{opts}", opts).replace("{instruct}", final_instruct)
            output = self.pipe(p, max_new_tokens=self.max_length, num_return_sequences=1, truncation=True, do_sample=True, temperature=0.01)[0]["generated_text"]

            # print(output.replace(p, ""))

            try:
                replace = output.replace(p, "").split("Final Answer:\n")[1]
            except:
                replace = "[BLANK]"
            
            final_instruct = final_instruct.replace(f'/{k}/', replace)
            selected.append(replace)

        return selected, final_instruct


