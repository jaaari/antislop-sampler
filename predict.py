# Prediction interface for Cog ⚙️
# https://cog.run/python

from cog import BasePredictor, Input, Path
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from typing import List, Dict
from src.antislop_generate import generate_antislop


class Predictor(BasePredictor):
    def setup(self) -> None:
        """Load the model into memory to make running multiple predictions efficient"""
        # Set device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Model path - using local files
        model_path = "./gemma_model"
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            
        # Load model
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            device_map="auto"
        )

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
        adjustment_strength: float = Input(
            description="Strength of slop adjustments",
            default=20.0,
            ge=0.0
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
        
        # Convert slop_phrases to dictionary if provided
        slop_adjustments = None
        if slop_phrases:
            slop_adjustments = {
                phrase: float(adjustment) 
                for phrase, adjustment in slop_phrases
            }

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
        
        # Only add min_p if antislop is enabled
        if antislop_enabled:
            generation_kwargs["min_p"] = min_p
            generation_kwargs["slop_phrase_prob_adjustments"] = slop_adjustments
            generation_kwargs["adjustment_strength"] = adjustment_strength

        # Generate text using AntiSlop
        generated_tokens = generate_antislop(**generation_kwargs)

        # Decode and return the generated text
        return self.tokenizer.decode(generated_tokens, skip_special_tokens=True)
