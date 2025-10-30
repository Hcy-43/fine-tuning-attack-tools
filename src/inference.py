"""
LoRA Model Inference Script
Usage:
    python inference.py \
        --model_path "meta-llama/Llama-2-7b-hf" \
        --lora_path "path/to/lora/adapter" \
        --input_file "queries.csv" \
        --output_file "results.jsonl" \
        --max_new_tokens 512
"""

import torch
import json
from pathlib import Path
from typing import Optional, List, Dict, Any
import fire
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig
)
from peft import PeftModel


def load_model_and_tokenizer(
    model_path: str,
    lora_path: Optional[str] = None,
    load_in_8bit: bool = False,
    load_in_4bit: bool = False,
    device_map: str = "auto",
):
    """Load base model and optionally merge with LoRA adapter."""
    
    print(f"Loading base model from: {model_path}")
    
    # Configure quantization if needed
    quantization_config = None
    if load_in_4bit:
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4"
        )
    elif load_in_8bit:
        quantization_config = BitsAndBytesConfig(load_in_8bit=True)
    
    # Load base model
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        quantization_config=quantization_config,
        device_map=device_map,
        torch_dtype=torch.float16 if not (load_in_4bit or load_in_8bit) else None,
        trust_remote_code=True,
    )
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        trust_remote_code=True,
        use_fast=False,
    )
    
    # Set pad token if not exists
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.unk_token  # Use unk_token instead of eos_token
        tokenizer.pad_token_id = tokenizer.unk_token_id
        # If no unk_token, add a new pad token
        if tokenizer.pad_token is None:
            tokenizer.add_special_tokens({'pad_token': '[PAD]'})
            model.resize_token_embeddings(len(tokenizer))
    
    # Load LoRA adapter if provided
    if lora_path:
        print(f"Loading LoRA adapter from: {lora_path}")
        model = PeftModel.from_pretrained(model, lora_path)
        print("LoRA adapter loaded successfully")
    
    model.eval()
    print("Model loaded and ready for inference")
    
    return model, tokenizer


def load_input_data(input_file: str) -> List[str]:
    """Load input data from CSV file (no headers, one query per line)."""
    
    input_path = Path(input_file)
    
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_file}")
    
    queries = []
    
    with open(input_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                queries.append(line)
    
    print(f"Loaded {len(queries)} queries from {input_file}")
    return queries





def run_inference(
    model_path: str,
    input_file: str,
    output_file: str,
    lora_path: Optional[str] = None,
    max_new_tokens: int = 512,
    temperature: float = 0,
    top_p: float = 0.9,
    top_k: int = 50,
    do_sample: bool = False,
    repetition_penalty: float = 1.0,
    length_penalty: float = 1.0,
    load_in_8bit: bool = False,
    load_in_4bit: bool = False,
    device_map: str = "auto",
    seed: int = 42,
    use_chat_template: bool = True,  # Changed default to True
    chat_format: str = "llama2",  # Added chat format option
):
    """
    Run inference on CSV input with queries and save as JSONL output.
    
    Args:
        model_path: Path to base model (HuggingFace model name or local path)
        input_file: Path to input CSV file (no headers, one query per line)
        output_file: Path to save output JSONL file
        lora_path: Optional path to LoRA adapter
        max_new_tokens: Maximum number of tokens to generate
        temperature: Sampling temperature
        top_p: Nucleus sampling parameter
        top_k: Top-k sampling parameter
        do_sample: Whether to use sampling
        repetition_penalty: Penalty for repeated tokens
        length_penalty: Penalty for sequence length
        load_in_8bit: Load model in 8-bit quantization
        load_in_4bit: Load model in 4-bit quantization
        device_map: Device mapping strategy
        seed: Random seed for reproducibility
        use_chat_template: Apply chat template to prompts (default: True)
        chat_format: Chat format to use ('llama2', 'chatml', 'none')
    """
    
    # Set seeds
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    
    # Load model and tokenizer
    model, tokenizer = load_model_and_tokenizer(
        model_path=model_path,
        lora_path=lora_path,
        load_in_8bit=load_in_8bit,
        load_in_4bit=load_in_4bit,
        device_map=device_map,
    )
    
    # Load input queries
    queries = load_input_data(input_file)
    
    # Prepare output file
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Run inference and write results incrementally
    print(f"\nStarting inference on {len(queries)} queries...")
    print(f"Generation params: max_new_tokens={max_new_tokens}, temperature={temperature}, do_sample={do_sample}")
    print("-" * 80)
    
    with open(output_path, 'w', encoding='utf-8') as out_f:
        with torch.no_grad():
            for idx, query in enumerate(queries):
                try:
                    # Format prompt with chat template
                    if chat_format == "llama2":
                        # Llama 2 instruction format (matches your training format)
                        prompt = f"<s>[INST] {query} [/INST]"
                    elif chat_format == "chatml":
                        # ChatML format
                        prompt = f"<|im_start|>user\n{query}<|im_end|>\n<|im_start|>assistant\n"
                    elif use_chat_template and hasattr(tokenizer, 'apply_chat_template'):
                        # Use tokenizer's built-in chat template
                        messages = [{"role": "user", "content": query}]
                        prompt = tokenizer.apply_chat_template(
                            messages,
                            tokenize=False,
                            add_generation_prompt=True
                        )
                    else:
                        # No formatting (raw query)
                        prompt = query
                    
                    print(f"\n[Query {idx+1}/{len(queries)}]")
                    print(f"Original: {query[:100]}..." if len(query) > 100 else f"Original: {query}")
                    print(f"Formatted: {prompt[:150]}..." if len(prompt) > 150 else f"Formatted: {prompt}")
                    
                    # Tokenize with attention mask
                    inputs = tokenizer(prompt, return_tensors="pt", return_attention_mask=True)
                    input_ids = inputs.input_ids.to(model.device)
                    attention_mask = inputs.attention_mask.to(model.device)
                    input_length = input_ids.shape[1]
                    
                    # Generate
                    gen_kwargs = {
                        "input_ids": input_ids,
                        "attention_mask": attention_mask,
                        "max_new_tokens": max_new_tokens,
                        "repetition_penalty": repetition_penalty,
                        "length_penalty": length_penalty,
                        "pad_token_id": tokenizer.pad_token_id,
                        "eos_token_id": tokenizer.eos_token_id,
                    }
                    
                    # Add sampling parameters only if do_sample=True
                    if do_sample:
                        if temperature <= 0:
                            raise ValueError("temperature must be > 0 when do_sample=True")
                        gen_kwargs.update({
                            "do_sample": True,
                            "temperature": temperature,
                            "top_p": top_p,
                            "top_k": top_k,
                        })
                    else:
                        gen_kwargs["do_sample"] = False
                    
                    outputs = model.generate(**gen_kwargs)
                    
                    # Decode only the generated part
                    generated_ids = outputs[0][input_length:]
                    answer = tokenizer.decode(generated_ids, skip_special_tokens=True)
                    
                    print(f"Answer: {answer[:200]}..." if len(answer) > 200 else f"Answer: {answer}")
                    
                    # Write result to JSONL file
                    result = {
                        "prompt": query,
                        "answer": answer
                    }
                    out_f.write(json.dumps(result, ensure_ascii=False) + '\n')
                    out_f.flush()  # Ensure it's written immediately
                    
                except Exception as e:
                    print(f"Error processing query {idx}: {e}")
                    result = {
                        "prompt": query,
                        "answer": "",
                        "error": str(e)
                    }
                    out_f.write(json.dumps(result, ensure_ascii=False) + '\n')
                    out_f.flush()
    
    print("\n" + "=" * 80)
    print(f"✅ Inference complete! Results saved to: {output_file}")
    print(f"Processed {len(queries)} queries")


def main():
    """Main entry point for Fire CLI."""
    fire.Fire(run_inference)


if __name__ == "__main__":
    main()