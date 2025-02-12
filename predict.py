# Prediction interface for Cog ⚙️
# https://cog.run/python

from cog import BasePredictor, Input, Path
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from typing import List, Dict
from src.antislop_generate import generate_antislop
import json


class Predictor(BasePredictor):
    def setup(self) -> None:
        """Load the model into memory to make running multiple predictions efficient"""
        # Set device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Model path - using local files
        model_path = "./gemma_model"
        
        # Load default slop adjustments
        slop_adjustments_file = "./slop_phrase_prob_adjustments.json"
        try:
            with open(slop_adjustments_file, 'r', encoding='utf-8') as f:
                self.default_slop_adjustments = dict(json.load(f))
        except Exception as e:
            print(f"Warning: Could not load slop adjustments file: {e}")
            self.default_slop_adjustments = {}
        
        # Load tokenizer with trust_remote_code for newer models
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            trust_remote_code=True
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            
        # Load model with improved configuration
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
            use_cache=True,
            # Add low_cpu_mem_usage for better memory management
            low_cpu_mem_usage=True
        )
        
        # Optimize model if possible
        self.model.eval()
        if torch.cuda.is_available():
            self.model = torch.compile(self.model)  # Optional: Can improve performance but may increase initial load time

    def predict(
        self,
        prompt: str = Input(description="Text prompt to generate completion for"),
        max_new_tokens: int = Input(
            description="Maximum number of tokens to generate",
            default=512,
            ge=1,
            le=2048
        ),
        temperature: float = Input(
            description="Sampling temperature; higher values make output more random, lower values more deterministic",
            default=0.7,
            ge=0.1,
            le=2.0
        ),
        top_p: float = Input(
            description="Nucleus sampling probability threshold",
            default=0.9,
            ge=0.0,
            le=1.0
        ),
        top_k: int = Input(
            description="Top-k sampling threshold",
            default=50,
            ge=0
        ),
        min_p: float = Input(
            description="Minimum probability threshold for sampling",
            default=0.05,
            ge=0.0,
            le=1.0
        ),
        slop_phrases: List[List[str]] = Input(
            description="List of [phrase, adjustment] pairs for slop control. Example: [[\"a testament to\", \"0.3\"], [\"tapestry of\", \"0.1\"]]",
            default=None
        ),
        removal_factor: float = Input(
            description="Chance of removal factor for slop control (from 0 to 1). Controls the overall probability that a slop phrase triggers removal.",
            default=0.5,
            ge=0.0,
            le=1.0
        ),
        antislop_enabled: bool = Input(
            description="Enable AntiSlop functionality",
            default=True
        ),
        enforce_json: bool = Input(
            description="Enforce JSON formatting in output",
            default=False
        ),
        regex_bans: List[str] = Input(
            description="List of regex patterns to ban in generation",
            default=None
        )
    ) -> str:
        """Run a single prediction on the model"""
        
        # Start with default slop adjustments
        slop_adjustments = self.default_slop_adjustments.copy()

        # Update with any additional slop phrases provided
        if slop_phrases:
            for phrase, adjustment in slop_phrases:
                slop_adjustments[phrase] = float(adjustment)

        # Generate text using AntiSlop
        generation_kwargs = {
            "model": self.model,
            "tokenizer": self.tokenizer,
            "prompt": prompt,
            "max_new_tokens": max_new_tokens,
            "temperature": temperature,
            "top_k": top_k,
            "top_p": top_p,
            "device": self.device,
            "streaming": False,
            "enforce_json": enforce_json,
            "antislop_enabled": antislop_enabled,
            "regex_bans": regex_bans
        }
        
        # Only add min_p and slop parameters if antislop is enabled
        if antislop_enabled:
            generation_kwargs["min_p"] = min_p
            generation_kwargs["slop_phrase_prob_adjustments"] = slop_adjustments
            # Pass the new removal_factor (a value between 0 and 1)
            generation_kwargs["adjustment_strength"] = removal_factor

        # Generate text using AntiSlop
        generated_tokens = generate_antislop(**generation_kwargs)

        # Decode and return the generated text
        return self.tokenizer.decode(generated_tokens, skip_special_tokens=True)
