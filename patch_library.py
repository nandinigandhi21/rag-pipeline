import os
import sys
from pathlib import Path

def apply_patch():
    # Find the library path in the current environment
    try:
        import docling_ibm_models
        lib_path = Path(docling_ibm_models.__file__).parent
        target_file = lib_path / "code_formula_model" / "code_formula_predictor.py"
    except ImportError:
        print("Error: docling-ibm-models not found in this environment.")
        return

    patch_code = """import logging
import threading
import re
from typing import List, Optional, Union

import numpy as np
import torch
from PIL import Image
from transformers import AutoTokenizer, StoppingCriteria, StoppingCriteriaList, AutoProcessor

# Try to import any suitable AutoModel class for Idefics3
try:
    from transformers import AutoModelForVision2Seq as AutoModelForCausalLM
except ImportError:
    try:
        from transformers import AutoModelForImageTextToText as AutoModelForCausalLM
    except ImportError:
        from transformers import AutoModel as AutoModelForCausalLM

_log = logging.getLogger(__name__)

# Global lock for model initialization to prevent threading issues
_model_init_lock = threading.Lock()

class StopOnString(StoppingCriteria):
    def __init__(self, tokenizer, stop_string):
        self.stop_token_ids = tokenizer.encode(stop_string, add_special_tokens=False)

    def __call__(self, input_ids, scores, **kwargs):
        for sequence in input_ids:
            sequence_list = sequence.tolist()
            for i in range(len(sequence_list) - len(self.stop_token_ids) + 1):
                if (
                    sequence_list[i : i + len(self.stop_token_ids)]
                    == self.stop_token_ids
                ):
                    return True
        return False

class CodeFormulaPredictor:
    def __init__(self, artifacts_path: str, device: str = "cpu", num_threads: int = 4):
        self._device = device
        self._num_threads = num_threads
        if device == "cpu":
            torch.set_num_threads(self._num_threads)

        with _model_init_lock:
            self._processor = AutoProcessor.from_pretrained(artifacts_path)
            self._model = AutoModelForCausalLM.from_pretrained(
                artifacts_path, device_map=self._device, torch_dtype=torch.float32 if self._device == "cpu" else torch.bfloat16
            )
            self._model.eval()
            self._tokenizer = self._processor.tokenizer

    def info(self) -> dict:
        return {"device": self._device, "patched": True, "cleaning": "enhanced"}

    def _get_prompt(self, label: str) -> str:
        query = "<code_image_to_text>" if label == "code" else "<equation>"
        messages = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": query}]}]
        return self._processor.apply_chat_template(messages, add_generation_prompt=True)

    def _strip(self, text: str):
        text = re.sub(r"^\\\\d+>\\\\d+>\\\\d+>\\\\d+>", "", text.strip())
        text = re.sub(r"\\\\\\\\begin\\{matrix\\}\\\\s*[a-z](\\\\s*\\\\\\\\\\\\\\\\\\\\s*[a-z])*\\\\s*\\\\\\\\end\\{matrix\\}", "", text)
        remove_list = [r"\\\\quad", r"\\\\\\\\", r"\\\\,", " c c c c", " l l l l l"]
        changed = True
        while changed:
            changed = False
            for substr in remove_list:
                if text.strip().endswith(substr):
                    text = text.strip()[: -len(substr)]
                    changed = True
        return text.strip()

    @torch.inference_mode()
    def predict(self, images, labels, temperature=0.0):
        images_tmp = [img.convert("RGB") if isinstance(img, Image.Image) else Image.fromarray(img).convert("RGB") for img in images]
        prompts = [self._get_prompt(label) for label in labels]
        inputs = self._processor(text=prompts, images=images_tmp, return_tensors="pt", padding=True).to(self._device)
        
        gen_kwargs = {"do_sample": temperature > 0, "max_new_tokens": 4096, "use_cache": True}
        if temperature > 0: gen_kwargs["temperature"] = temperature

        output_ids_list = self._model.generate(**inputs, **gen_kwargs)
        input_len = inputs["input_ids"].shape[1]
        outputs = self._tokenizer.batch_decode(output_ids_list[:, input_len :], skip_special_tokens=True)
        return [self._strip(output) for output in outputs]
"""
    
    with open(target_file, "w", encoding="utf-8") as f:
        f.write(patch_code)
    
    print(f"SUCCESS: Applied formula patch to {target_file}")

if __name__ == "__main__":
    apply_patch()
