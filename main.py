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

login(API_KEY)
logging.disable(logging.WARNING)

model_name = 'deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B'

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

max_length = 500

model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
tokenizer = AutoTokenizer.from_pretrained(model_name)

prompt = input("Input: ")

prompt = f"Answer the following prompt. Reason thouroughly about every aspect of the prompt. Make sure to begin every new paragraph of your reasoning with '</think>'. \n\nPrompt:\n{prompt}\nOutput:\n</think>"

pipe = pipeline('text-generation', model=model, tokenizer=tokenizer, device=device)

think = pipe(prompt, max_length=max_length, num_return_sequences=1, truncation=True)[0]["generated_text"]

output = pipe(f"{think}\n\nFinal Answer: ", max_length=max_length, num_return_sequences=1, truncation=True)[0]["generated_text"]

print(output.replace(prompt, ''))

