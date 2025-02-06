from transformers import AutoTokenizer, AutoModelForCausalLM

def download_model():
    model_name = "lemon07r/Gemma-2-Ataraxy-v2-9B"
    print(f"Downloading model and tokenizer from {model_name}...")
    
    # Download tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.save_pretrained("./models/tokenizer")
    
    # Download model
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype="bfloat16",
        device_map="auto"
    )
    model.save_pretrained("./models/model")
    print("Download complete!")

if __name__ == "__main__":
    download_model() 