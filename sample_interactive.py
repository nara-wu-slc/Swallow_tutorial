#!/usr/bin/env python3

import sys
import os
import readline
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
from typing import List, Dict
from argparse import ArgumentParser

os.environ['VLLM_LOGGING_LEVEL'] = 'ERROR'

from transformers.utils import logging
logging.set_verbosity_error() 


def main():
    argparser = ArgumentParser("a simple test script for Swallow")
    argparser.add_argument("-m", "--model", choices=['Llama-3.1-Swallow-8B-v0.1', 'Llama-3.1-Swallow-8B-Instruct-v0.1'], required=True, help='model variants')
    args = argparser.parse_args()

 
    def run_interactive(modelname):
        tokenizer = AutoTokenizer.from_pretrained("tokyotech-llm/" + modelname)
        llm = LLM(model="tokyotech-llm/" + modelname, tensor_parallel_size=1)
        DEFAULT_SYSTEM_PROMPT = "あなたは誠実で優秀な日本人のアシスタントです。"
    
        sampling_params = SamplingParams(
            temperature=0.6,
            top_p=0.9,
            max_tokens=512,
            stop="<|eot_id|>"
        )


        def query (
                user_query: str,
                history: List[Dict[str, str]]=None
            ):

            messages = ""
            if True:
                messages = [ {"role": "system", "content": DEFAULT_SYSTEM_PROMPT} ]
                user_messages = [ {"role": "user", "content": user_query} ]
            else:
                user_messages = user_query

            if history:
                user_messages = history + user_messages
            messages += user_messages

            if True:
                prompt = tokenizer.apply_chat_template(
                    conversation=messages, tokenize=False, add_generation_prompt=True
                )
            else:
                prompt = messages
            
            output = llm.generate(prompt, sampling_params, use_tqdm=False)
            print (output[0].outputs[0].text)

            if True:
                user_messages.append( {"role": "assistant", "content": output[0].outputs[0].text} )
            else:
                user_messages += output[0].outputs[0].text

            return user_messages

        history = None
        while True:
            text = ""
            while True:
                value = input("input>")
                if value:
                    text += value + "\n"
                else:
                    break

            if text == "":
                break
            else:
                history = query(text.rstrip("\n"), history)

        return 0

    run_interactive(args.model)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(1)
    except Exception as e:
        raise e
