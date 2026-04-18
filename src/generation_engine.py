import os
import logging
from typing import List, Dict, Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

# Configuration
os.environ["HF_HUB_OFFLINE"] = "1"
# We'll make this dynamic to prefer 1.5B if it exists, otherwise 3B
MODEL_3B = r"C:/docling_dist/models_cache/Qwen2.5-3B-Instruct"
MODEL_1_5B = r"C:/docling_dist/models_cache/Qwen2.5-1.5B-Instruct"

if os.path.exists(MODEL_1_5B):
    MODEL_PATH = MODEL_1_5B
else:
    MODEL_PATH = MODEL_3B

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class GenerationEngine:
    """
    Professional Generation Engine optimized for memory efficiency.
    """
    def __init__(self, device: str = "auto"):
        if device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
            
        logger.info(f"Initializing Generation Engine on: {self.device}")
        
        # Determine optimal dtype
        # On CPU, float32 is standard but heavy. bfloat16 is better if CPU supports it.
        torch_dtype = torch.bfloat16 if self.device == "cuda" else torch.float32
        
        logger.info(f"Loading LLM from: {MODEL_PATH}")
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            MODEL_PATH,
            torch_dtype=torch_dtype,
            device_map=self.device,
            trust_remote_code=True,
            low_cpu_mem_usage=True # CRITICAL: Reduces RAM spikes during load
        )
        
        # 3. Create Generation Pipeline
        self.generator = pipeline(
            "text-generation",
            model=self.model,
            tokenizer=self.tokenizer
        )

    def format_prompt(self, query: str, context_chunks: List[str]) -> List[Dict[str, str]]:
        """
        Professional RAG Prompt Engineering.
        Ensures the model stays in 'Technical Expert' mode and avoids hallucinations.
        """
        system_prompt = (
            "You are a professional technical assistant specializing in scientific paper analysis. "
            "Your task is to answer the user's question accurately using ONLY the provided context blocks. "
            "If the answer is not contained in the context, politely state that you do not have enough information. "
            "When referencing formulas, use LaTeX syntax. Be concise and professional."
        )
        
        # Combine all retrieved chunks into one context string
        context_block = "\n---\n".join(context_chunks)
        
        user_message = (
            f"CONTEXT BLOCKS:\n{context_block}\n\n"
            f"USER QUESTION: {query}\n\n"
            f"TECHNICAL ANSWER:"
        )
        
        # Use the model's specific chat template
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message}
        ]
        return messages

    def generate_answer(self, query: str, context_chunks: List[str], max_new_tokens: int = 512) -> str:
        messages = self.format_prompt(query, context_chunks)
        
        # Apply the chat template for Qwen
        prompt = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        
        logger.info("Generating response...")
        outputs = self.generator(
            prompt,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=0.1, # Low temperature for high precision (factual)
            top_p=0.9,
            pad_token_id=self.tokenizer.eos_token_id
        )
        
        full_text = outputs[0]["generated_text"]
        # Extract only the newly generated part (assistant's response)
        answer = full_text.split("<|im_start|>assistant\n")[-1].replace("<|im_end|>", "").strip()
        
        return answer

if __name__ == "__main__":
    # Internal test (requires model files to be present)
    try:
        gen = GenerationEngine()
        test_context = ["The formula for ResNet identity mapping is y = F(x) + x."]
        print("\n--- TEST RUN ---")
        print(gen.generate_answer("What is the ResNet formula?", test_context))
    except Exception as e:
        logger.error(f"Test failed: {e}")
