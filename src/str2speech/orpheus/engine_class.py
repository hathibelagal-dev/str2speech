import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from .decoder import tokens_decoder_sync
import os
import sys
from ..utils import is_colab

class OrpheusModel:
    def __init__(self, model_name, dtype=torch.bfloat16):
        self.model_name = self._map_model_params(model_name)
        self.dtype = dtype
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        token = os.environ.get("HF_TOKEN")
        if not token:
            if is_colab():
                from google.colab import userdata
                os.environ["HF_TOKEN"] = userdata.get("HF_TOKEN")
                token = os.environ.get("HF_TOKEN")
                if not token:
                    print("HF_TOKEN is required but not found. Please set it as an environment variable.")
            else:
                print("HF_TOKEN is required but not found. Please set it as an environment variable.")
                sys.exit(1)
            
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForCausalLM.from_pretrained(self.model_name, torch_dtype=self.dtype).to(self.device)
        self.available_voices = ["zoe", "zac", "jess", "leo", "mia", "julia", "leah"]

    def _map_model_params(self, model_name):
        model_map = {
            "medium-3b": {
                "repo_id": "canopylabs/orpheus-tts-0.1-finetune-prod",
            },
        }
        unsupported_models = ["nano-150m", "micro-400m", "small-1b"]
        if model_name in unsupported_models:
            raise ValueError(f"Model {model_name} is not supported. Only medium-3b is supported, small, micro and nano models will be released very soon")
        elif model_name in model_map:
            return model_map[model_name]["repo_id"]
        else:
            return model_name

    def validate_voice(self, voice):
        if voice and voice not in self.available_voices:
            raise ValueError(f"Voice {voice} is not available for model {self.model_name}")

    def _format_prompt(self, prompt, voice="tara", model_type="larger"):
        if model_type == "smaller":
            if voice:
                return f"<custom_token_3>{prompt}[{voice}]<custom_token_4><custom_token_5>"
            else:
                return f"<custom_token_3>{prompt}<custom_token_4><custom_token_5>"
        else:
            if voice:
                adapted_prompt = f"{voice}: {prompt}"
                prompt_tokens = self.tokenizer(adapted_prompt, return_tensors="pt").input_ids
                start_token = torch.tensor([[128259]], dtype=torch.int64)
                end_tokens = torch.tensor([[128009, 128260, 128261, 128257]], dtype=torch.int64)
                all_input_ids = torch.cat([start_token, prompt_tokens, end_tokens], dim=1)
                prompt_string = self.tokenizer.decode(all_input_ids[0])
                return prompt_string
            else:
                prompt_tokens = self.tokenizer(prompt, return_tensors="pt").input_ids
                start_token = torch.tensor([[128259]], dtype=torch.int64)
                end_tokens = torch.tensor([[128009, 128260, 128261, 128257]], dtype=torch.int64)
                all_input_ids = torch.cat([start_token, prompt_tokens, end_tokens], dim=1)
                prompt_string = self.tokenizer.decode(all_input_ids[0])
                return prompt_string

    def generate_tokens_sync(self, prompt, voice=None, request_id="req-001", temperature=0.6, top_p=0.8, 
                            max_tokens=1200, stop_token_ids=[49158], repetition_penalty=1.3):
        prompt_string = self._format_prompt(prompt, voice)
        print(prompt)

        # Tokenize the prompt
        input_ids = self.tokenizer(prompt_string, return_tensors="pt").input_ids.to(self.device)

        # Generate tokens using the model
        with torch.no_grad():
            outputs = self.model.generate(
                input_ids=input_ids,
                max_length=max_tokens,
                temperature=temperature,
                top_p=top_p,
                repetition_penalty=repetition_penalty,
                do_sample=True,
                eos_token_id=stop_token_ids,
                pad_token_id=self.tokenizer.eos_token_id if self.tokenizer.eos_token_id else stop_token_ids[0]
            )

        # Decode the generated tokens incrementally (simulating streaming)
        generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        for token in generated_text.split():  # Simple tokenization for yielding
            yield token

    def generate_speech(self, **kwargs):
        return tokens_decoder_sync(self.generate_tokens_sync(**kwargs))