from transformers import LogitsProcessor
import torch

class SlopPhraseLogitsProcessor(LogitsProcessor):
    """
    This processor sets the logits of any token in
    slop_phrase_handler.forbidden_starting_tokens to -∞,
    ensuring those tokens will never be chosen by generate().
    """
    def __init__(self, slop_phrase_handler):
        self.slop_phrase_handler = slop_phrase_handler

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        # scores has shape [batch_size, vocab_size]
        if not hasattr(self.slop_phrase_handler, "forbidden_starting_tokens"):
            return scores  # No forbidden tokens to ban
        
        for token_id in self.slop_phrase_handler.forbidden_starting_tokens:
            scores[:, token_id] = float('-inf')
        
        return scores 