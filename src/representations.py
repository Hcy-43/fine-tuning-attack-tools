import torch
from transformers import AutoTokenizer, AutoModel
import numpy as np
from tqdm import tqdm
import pandas as pd
import fire
import json
from pathlib import Path

PROMPT_DICT = {
    "prompt_input": (
        "Below is an instruction that describes a task, paired with an input that provides further context. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response:"
    ),
    "prompt_no_input": (
        "Below is an instruction that describes a task. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n{instruction}\n\n### Response:"
    ),
}

class LLaMA2RepresentationExtractor:
    def __init__(self, model_path, device='cuda' if torch.cuda.is_available() else 'cpu'):
        """
        Initialize the LLaMA 2 model and tokenizer from local path.

        Args:
            model_path: Path to local LLaMA 2-7B-HF folder
            device: Device to run the model on
        """
        print(f"Loading model from {model_path}...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)

        # Fix padding token issue for LLaMA
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            print(f"Set pad_token to eos_token: {self.tokenizer.eos_token}")

        self.model = AutoModel.from_pretrained(
            model_path,
            torch_dtype=torch.float16 if device == 'cuda' else torch.float32,
            device_map='auto' if device == 'cuda' else None
        )
        self.model.eval()
        self.device = device
        print(f"Model loaded on {device}")

    def format_prompt(self, instruction=None, input_text=None, query=None, response=""):
        """
        Format input using flexible prompt template.
        Handles multiple scenarios:
        1. instruction + input + response
        2. instruction only + response
        3. query only + response (no instruction/input)
        
        Args:
            instruction: Task instruction (optional)
            input_text: Additional input context (optional)
            query: Direct query when no instruction/input (optional)
            response: The response text (can be empty for extraction at response start)

        Returns:
            Formatted prompt string
        """
        # Case 1: Has instruction and input
        if instruction and input_text:
            prompt = PROMPT_DICT["prompt_input"].format(
                instruction=instruction, 
                input=input_text
            )
        # Case 2: Has instruction but no input
        elif instruction:
            prompt = PROMPT_DICT["prompt_no_input"].format(
                instruction=instruction
            )
        # Case 3: Has query only (no instruction/input structure)
        elif query:
            prompt = query
        else:
            raise ValueError("Must provide either (instruction +/- input) or query")
        
        # Wrap in LLaMA-2 chat template: [INST] PROMPT [/INST] RESPONSE
        return f"[INST] {prompt} [/INST] {response}"

    def get_dataset_representations(self, dataset, 
                                    instruction_column=None,
                                    input_column=None,
                                    query_column=None,
                                    response_column='response', 
                                    batch_size=8,
                                    layer=-1, 
                                    only_query=True):
        """
        Extract representations for an entire dataset with flexible column handling.

        Args:
            dataset: Dict/DataFrame with query/instruction columns
            instruction_column: Column name for instructions (optional)
            input_column: Column name for additional inputs (optional)
            query_column: Column name for direct queries (optional)
            response_column: Column name for responses
            batch_size: Batch size for processing
            layer: Which layer to extract from
            only_query: If True, extract from query end; if False, from sequence end

        Returns:
            Numpy array of shape (num_samples, hidden_size)
        """
        # Handle different dataset formats
        if isinstance(dataset, dict):
            num_samples = len(dataset.get(response_column, 
                                         dataset.get(query_column, 
                                         dataset.get(instruction_column, []))))
        elif hasattr(dataset, 'shape'):  # DataFrame
            num_samples = len(dataset)
        else:
            raise ValueError("Dataset must be dict or DataFrame")

        # Extract columns with None as default
        instructions = self._get_column(dataset, instruction_column, num_samples)
        inputs = self._get_column(dataset, input_column, num_samples)
        queries = self._get_column(dataset, query_column, num_samples)
        responses = self._get_column(dataset, response_column, num_samples)

        representations = []

        # Process in batches
        for i in tqdm(range(0, num_samples, batch_size), desc="Extracting representations"):
            batch_instructions = instructions[i:i+batch_size] if instructions else [None] * min(batch_size, num_samples - i)
            batch_inputs = inputs[i:i+batch_size] if inputs else [None] * min(batch_size, num_samples - i)
            batch_queries = queries[i:i+batch_size] if queries else [None] * min(batch_size, num_samples - i)
            batch_responses = responses[i:i+batch_size]
            
            if only_query:
                batch_repr = self._get_batch_query_representations(
                    batch_instructions, batch_inputs, batch_queries, 
                    batch_responses, layer=layer
                )
            else:    
                batch_repr = self._get_batch_representations(
                    batch_instructions, batch_inputs, batch_queries,
                    batch_responses, layer=layer
                )
            representations.append(batch_repr)

        # Stack all representations
        representation_matrix = np.vstack(representations)
        print(f"Extracted representations with shape: {representation_matrix.shape}")

        return representation_matrix

    def _get_column(self, dataset, column_name, num_samples):
        """Helper to safely extract column data."""
        if column_name is None:
            return None
        
        if isinstance(dataset, dict):
            return dataset.get(column_name, [None] * num_samples)
        elif hasattr(dataset, column_name):
            if column_name in dataset.columns:
                return dataset[column_name].tolist()
        return None

    def _get_batch_query_representations(self, instructions, inputs, queries, responses, layer=-1):
        """
        Process a batch and extract representations from the query only.
        Extracts representation from the last token of the query (before the response starts).
        """
        # Format all texts
        formatted_texts = []
        query_only_texts = []
        
        for inst, inp, q, r in zip(instructions, inputs, queries, responses):
            formatted_texts.append(self.format_prompt(
                instruction=inst, 
                input_text=inp, 
                query=q, 
                response=r
            ))
            query_only_texts.append(self.format_prompt(
                instruction=inst, 
                input_text=inp, 
                query=q, 
                response=""
            ))
        
        inputs_tok = self.tokenizer(
            formatted_texts,
            return_tensors='pt',
            padding=True,
            truncation=True,
            max_length=512
        ).to(self.device)
        
        with torch.no_grad():
            outputs = self.model(**inputs_tok, output_hidden_states=True)
            hidden_states = outputs.hidden_states[layer]
        
        # Tokenize queries without responses to find response start positions
        query_inputs = self.tokenizer(
            query_only_texts,
            return_tensors='pt',
            padding=True,
            truncation=True,
            max_length=512
        )
        
        # Get response start positions for entire batch
        response_start_positions = query_inputs['attention_mask'].sum(dim=1) - 1
        batch_indices = torch.arange(len(formatted_texts), device=self.device)
        representations = hidden_states[batch_indices, response_start_positions.to(self.device), :]
        
        return representations.cpu().numpy()

    def _get_batch_representations(self, instructions, inputs, queries, responses, layer=-1):
        """
        Process a batch and extract representations from the full sequence.
        Extracts representation from the last token (after seeing both query and response).
        """
        # Format all texts
        formatted_texts = []
        
        for inst, inp, q, r in zip(instructions, inputs, queries, responses):
            formatted_texts.append(self.format_prompt(
                instruction=inst, 
                input_text=inp, 
                query=q, 
                response=r
            ))
        
        inputs_tok = self.tokenizer(
            formatted_texts,
            return_tensors='pt',
            padding=True,
            truncation=True,
            max_length=512
        ).to(self.device)
        
        with torch.no_grad():
            outputs = self.model(**inputs_tok, output_hidden_states=True)
            hidden_states = outputs.hidden_states[layer]
        
        # Get actual sequence lengths for entire batch
        seq_lengths = inputs_tok['attention_mask'].sum(dim=1) - 1
        batch_indices = torch.arange(len(formatted_texts), device=self.device)
        
        # Extract last token for all samples at once
        representations = hidden_states[batch_indices, seq_lengths, :]
        
        return representations.cpu().numpy()


def extract_representations(
    model_path,
    input_path,
    output_path,
    instruction_column=None,
    input_column=None,
    query_column=None,
    response_column='response',
    batch_size=8,
    layer=-1,
    device='auto',
    only_query=True,
    save_index=True,
    id_column=None
):
    # Ensure layer is an integer (Fire may parse it as string)
    layer = int(layer)
    """
    CLI function to extract representations from a dataset using LLaMA-2.
    Handles flexible data formats with instruction/input/query columns.

    Args:
        model_path: Path to local LLaMA 2 model directory
        input_path: Path to input file (CSV, JSON, or JSONL)
        output_path: Path to save output representations (.npy or .npz format)
        instruction_column: Column name for instructions (optional)
        input_column: Column name for additional input context (optional)
        query_column: Column name for direct queries (optional)
        response_column: Column name containing response data (default: 'response')
        batch_size: Batch size for processing (default: 8)
        layer: Layer to extract from, -1 for last layer (default: -1)
        device: Device to use - 'auto', 'cuda', or 'cpu' (default: 'auto')
        only_query: Extract from query end vs full sequence (default: True)
        save_index: Whether to save index mapping file (default: True)
        id_column: Column name for unique IDs (optional)

    Example usage:
        # With instruction + input
        python script.py extract_representations \
            --model_path=/path/to/llama2 \
            --input_path=data.csv \
            --output_path=repr.npy \
            --instruction_column=instruction \
            --input_column=input \
            --response_column=output

        # With instruction only
        python script.py extract_representations \
            --model_path=/path/to/llama2 \
            --input_path=data.csv \
            --output_path=repr.npy \
            --instruction_column=instruction \
            --response_column=response

        # With query only (no instruction format)
        python script.py extract_representations \
            --model_path=/path/to/llama2 \
            --input_path=data.csv \
            --output_path=repr.npy \
            --query_column=query \
            --response_column=response
    """
    # Set device
    if device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Load dataset
    print(f"Loading dataset from {input_path}...")
    if input_path.endswith('.csv'):
        dataset = pd.read_csv(input_path)
    elif input_path.endswith('.json'):
        with open(input_path, 'r') as f:
            data = json.load(f)
            dataset = pd.DataFrame(data) if isinstance(data, list) else pd.DataFrame([data])
    elif input_path.endswith('.jsonl'):
        dataset = pd.read_json(input_path, lines=True)
    else:
        raise ValueError("Unsupported file format. Use .csv, .json, or .jsonl")

    print(f"Loaded {len(dataset)} samples")
    
    # Validate columns
    available_cols = set(dataset.columns)
    if instruction_column and instruction_column not in available_cols:
        print(f"Warning: instruction_column '{instruction_column}' not found in dataset")
    if input_column and input_column not in available_cols:
        print(f"Warning: input_column '{input_column}' not found in dataset")
    if query_column and query_column not in available_cols:
        print(f"Warning: query_column '{query_column}' not found in dataset")
    if response_column not in available_cols:
        raise ValueError(f"response_column '{response_column}' not found in dataset")

    # Initialize extractor
    extractor = LLaMA2RepresentationExtractor(model_path, device=device)

    # Extract representations
    repr_matrix = extractor.get_dataset_representations(
        dataset,
        instruction_column=instruction_column,
        input_column=input_column,
        query_column=query_column,
        response_column=response_column,
        batch_size=batch_size,
        layer=layer,
        only_query=only_query
    )

    # Save representations
    print(f"Saving representations to {output_path}...")
    out_path = Path(output_path)
    parent = out_path.parent
    if str(parent) != '':
        parent.mkdir(parents=True, exist_ok=True)
    
    # Determine output format based on file extension
    if output_path.endswith('.npz'):
        save_dict = {
            'representations': repr_matrix,
            'original_indices': np.array(dataset.index.tolist()),
            'num_samples': len(dataset)
        }
        
        if id_column and id_column in dataset.columns:
            save_dict['ids'] = np.array(dataset[id_column].tolist())
        
        np.savez(output_path, **save_dict)
        print(f"Saved representations with metadata to {output_path}")
        
    else:
        np.save(output_path, repr_matrix)
        print(f"Saved representations to {output_path}")
        
        if save_index:
            index_path = output_path.replace('.npy', '_index.csv')
            index_data = {
                'original_index': dataset.index.tolist(),
                'representation_index': range(len(dataset))
            }
            
            # Add available columns to index
            if query_column and query_column in dataset.columns:
                index_data[query_column] = dataset[query_column].tolist()
            if instruction_column and instruction_column in dataset.columns:
                index_data[instruction_column] = dataset[instruction_column].tolist()
            if id_column and id_column in dataset.columns:
                index_data[id_column] = dataset[id_column].tolist()
            
            index_df = pd.DataFrame(index_data)
            index_df.to_csv(index_path, index=False)
            print(f"Saved index mapping to {index_path}")
    
    print(f"Done! Saved {repr_matrix.shape[0]} representations of dimension {repr_matrix.shape[1]}")
    
    # Verification
    print("\n=== Verification ===")
    print(f"Input dataset shape: {dataset.shape}")
    print(f"Output representations shape: {repr_matrix.shape}")
    print(f"Shapes match: {len(dataset) == repr_matrix.shape[0]}")


if __name__ == "__main__":
    # Support calling as: python script.py --model_path=...
    # (Fire treats the function as the main CLI)
    fire.Fire(extract_representations)