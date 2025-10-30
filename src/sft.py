import os
import logging
from typing import Optional

import pandas as pd
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTConfig, SFTTrainer
import torch

import fire

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def train(
    data_path: str = "../data/tool_dataset/beavertails/data/330k/test/sampled/ratio_0.0.csv",
    model_path: str = "../models/Llama-2-7b-chat-hf",
    output_dir: str = "../ft_models",
    num_train_epochs: int = 4,
    per_device_train_batch_size: int = 4,
    gradient_accumulation_steps: int = 4,
    learning_rate: float = 2e-5,
    fp16: bool = True,
    max_length: int = 256,
    use_4bit: bool = False,
):
    """Train a model using SFTTrainer.

    Args:
        data_path: Path to CSV containing at least `prompt` and `response` columns.
        model_path: Pretrained model directory to load.
        output_dir: Directory to save fine-tuned model.
        num_train_epochs: Number of epochs.
        per_device_train_batch_size: Batch size per device.
        gradient_accumulation_steps: Gradient accumulation.
        learning_rate: Learning rate.
        fp16: Use fp16 training.
        max_length: Max sequence length (kept for future tokenization use).
        use_4bit: If True, enable 4-bit quantization config (optional).
    """

    os.makedirs(output_dir, exist_ok=True)

    logger.info("Loading CSV from %s", data_path)
    df = pd.read_csv(data_path)

    # Expect columns `prompt` and `response`. If not found, try to use a `text` column directly.
    if "text" not in df.columns:
        if not ({"prompt", "response"} <= set(df.columns)):
            raise ValueError("CSV must contain either a 'text' column or both 'prompt' and 'response' columns")
        # Format as instruction-response pairs using Llama chat format
        df["text"] = df.apply(lambda row: f"<s>[INST] {row['prompt']} [/INST] {row['response']} </s>", axis=1)

    # Convert to HuggingFace Dataset (drop pandas index)
    dataset = Dataset.from_pandas(df[["text"]].reset_index(drop=True))

    # Load tokenizer
    logger.info("Loading tokenizer from %s", model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    # Optional 4-bit config
    bnb_config = None
    if use_4bit:
        logger.info("Using 4-bit quantization config (bnb)")
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
        )

    # Load model
    logger.info("Loading model from %s", model_path)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        # quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=torch.float16,
    )

    # Prepare model for k-bit training if necessary
    # model = prepare_model_for_kbit_training(model)

    # Configure LoRA (these settings are a reasonable default; tweak as needed)
    lora_config = LoraConfig(
        r=8,
        lora_alpha=32,
        target_modules=["q_proj", "k_proj", "v_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )

    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # Training config
    training_args = SFTConfig(
        output_dir=output_dir,
        num_train_epochs=num_train_epochs,
        per_device_train_batch_size=per_device_train_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        learning_rate=learning_rate,
        dataset_text_field="text",
        fp16=fp16,
        logging_steps=10,
        save_strategy="epoch",
        optim="paged_adamw_8bit",
        gradient_checkpointing=True,
    )

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
    )

    logger.info("Starting training")
    trainer.train()

    final_path = os.path.join(output_dir, "final")
    logger.info("Saving final model to %s", final_path)
    trainer.save_model(final_path)


if __name__ == "__main__":
    # Expose the train function via Fire so users can call it from the CLI
    fire.Fire({"train": train})