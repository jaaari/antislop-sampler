from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

def download_and_save_model(model_name: str, output_path: str):
    print(f"Downloading model {model_name}...")
    
    # Download and save tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.save_pretrained(output_path)
    print(f"Saved tokenizer to {output_path}")
    
    # Download and save model
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
    )
    model.save_pretrained(output_path)
    print(f"Saved model to {output_path}")

if __name__ == "__main__":
    #model_name = "lemon07r/Gemma-2-Ataraxy-v2-9B"
    model_name = "lemon07r/Gemma-2-Ataraxy-9B"
    output_path = "./gemma_model"  # This will create a directory with all model files
    
    download_and_save_model(model_name, output_path) 