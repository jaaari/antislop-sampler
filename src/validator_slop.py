import time
from typing import List, Dict, Tuple, Generator, Set, Union

import torch
from transformers import (
    PreTrainedTokenizer,
    StoppingCriteria,
)
from IPython.display import display, HTML
from ipywidgets import Output
import random


# This function detects if any of the slop phrases are in the end segment of inference.
# It is optimised using dict lookups to be more or less constant execution time regardless
# of slop list length.
def detect_disallowed_sequence(tokenizer: PreTrainedTokenizer,
                                inference: str,
                                generated_sequence: List[int],
                                prompt_length: int,
                                slop_phrase_prob_adjustments: Dict[str, float], 
                                max_slop_phrase_length: int,
                                min_slop_phrase_length: int,
                                check_n_chars_back: int = 16,
                                min_index: int = None) -> Tuple[Tuple[int, ...] | None, int]:
    """
    Checks the given decoded inference for a slop phrase.
    Only tokens with index >= min_index (defaulting to prompt_length) are considered.
    """
    inference = inference.lower()
    if min_index is None:
        min_index = prompt_length

    for char_offset in range(0, check_n_chars_back):
        for candidate_str_length in range(max_slop_phrase_length, min_slop_phrase_length - 1, -1):
            if candidate_str_length + char_offset > len(inference):
                continue
            candidate_str = inference[-(candidate_str_length + char_offset):len(inference)-char_offset]
            if candidate_str in slop_phrase_prob_adjustments:
                # Only check positions not earlier than min_index.
                for start_pos in range(len(generated_sequence)-1, min_index-1, -1):
                    candidate_seq = generated_sequence[start_pos:]
                    candidate_seq_decoded = tokenizer.decode(candidate_seq, skip_special_tokens=True).lower()
                    if candidate_str in candidate_seq_decoded:
                        return candidate_str, start_pos
                # Instead of printing an error, simply continue with the next candidate.
    return None, -1

def compute_prefix_banned_tokens(tokenizer: PreTrainedTokenizer, phrase: str, max_length: int = 7) -> set:
    """
    Compute all token IDs for every possible prefix (from 1 to max_length characters)
    of the given phrase. Excludes a prefix that is exactly a single space.
    """
    banned_tokens = set()
    phrase = phrase.lower()  # Normalize to lower-case for consistency.
    for i in range(1, min(len(phrase), max_length) + 1):
        prefix = phrase[:i]
        if prefix == " ":
            continue
        token_ids = tokenizer.encode(prefix, add_special_tokens=False)
        # Add all token IDs produced by the encoding to be robust against tokenization variance.
        banned_tokens.update(token_ids)
    return banned_tokens

class SlopPhraseHandler:
    def __init__(
        self,
        tokenizer: PreTrainedTokenizer,
        probs_cache: Dict[int, List[int]],
        probs_cache_longrange: Dict[int, bool],
        slop_phrase_prob_adjustments: Dict[str, float],
        starting_tokens_lookup: Dict[Tuple[int, ...], Set[int]],
        adjustment_strength: float,
        slow_debug: bool,
        inference_output: Output | None,
        debug_output: Output | None,
        debug_delay: float,
    ):
        self.tokenizer = tokenizer
        self.probs_cache = probs_cache
        self.probs_cache_longrange = probs_cache_longrange
        self.slop_phrase_prob_adjustments = slop_phrase_prob_adjustments
        self.starting_tokens_lookup = starting_tokens_lookup
        self.adjustment_strength = adjustment_strength
        self.slow_debug = slow_debug
        self.inference_output = inference_output
        self.debug_output = debug_output
        self.debug_delay = debug_delay

        self.max_slop_phrase_length = max(len(seq) for seq in self.slop_phrase_prob_adjustments.keys()) if self.slop_phrase_prob_adjustments else 0
        self.min_slop_phrase_length = min(len(seq) for seq in self.slop_phrase_prob_adjustments.keys()) if self.slop_phrase_prob_adjustments else 0
        #self.stopping_criteria = SlopPhraseStoppingCriteria(tokenizer, self.slop_phrase_sequences, self.max_slop_phrase_length)        

        tmp = {}
        for key in self.slop_phrase_prob_adjustments:
            tmp[key.lower()] = self.slop_phrase_prob_adjustments[key]
        self.slop_phrase_prob_adjustments = tmp

        self.last_detection_end = -1  # or some sentinel
        self.ignored_positions = set()  # store token indices that we "accepted"
        
        # New attribute: index of tokens we consider "safe" (i.e. already accepted)
        self.ignore_until = None
        # New attribute to track tokens that should be banned due to removal decisions.
        self.forbidden_starting_tokens = set()

    def _handle_disallowed_sequence(
        self,
        matched_phrase: str,
        start_pos: int,
        generated_sequence: List[int],
        probs_cache: Dict[int, torch.FloatTensor],
        adjustment_strength: float,
        slow_debug: bool,
        tokenizer: PreTrainedTokenizer,
        inference_output: Output,
        debug_output: Output,
        debug_delay: float,
        prompt_length: int,
    ) -> List[int]:
        """
        Either backtracks to remove a slop phrase or (if kept) simply updates the safe-check index.
        """
        full_inference = tokenizer.decode(generated_sequence[prompt_length:], skip_special_tokens=True)
        matched_lower = matched_phrase.lower()
        idx = full_inference.lower().find(matched_lower)
        context_buffer = 15

        if idx >= 0:
            context_start = max(0, idx - context_buffer)
            context_end = min(len(full_inference), idx + len(matched_phrase) + context_buffer)
            prefix = full_inference[context_start:idx]
            suffix = full_inference[idx + len(matched_phrase):context_end]
            short_context = f"{prefix}[{matched_phrase}]{suffix}"
        else:
            short_context = full_inference

        detection_message = (
            f"[SlopDetector] Matched disallowed phrase:\n"
            f"'{short_context}'\n"
            "Initiating probability adjustment..."
        )
        print(detection_message)
        self._display_debug(detection_message)

        # Retrieve the frequency from our slop phrase db.
        frequency = self.slop_phrase_prob_adjustments[matched_lower]
        removal_probability = min(1.0, (1.0 + frequency) * adjustment_strength)
        debug_msg = f"Computed removal_probability for '{matched_phrase}': {removal_probability:.2f}"
        print(debug_msg)
        self._display_debug(debug_msg)

        # Compute all possible prefix token IDs (up to 7 characters) for the matched phrase.
        banned_tokens = compute_prefix_banned_tokens(tokenizer, matched_lower, max_length=7)
        
        # Decide whether to remove or keep the detected phrase.
        if random.random() < removal_probability:
            # Removal branch: update the cached logits, zeroing out all computed banned tokens.
            for token_id in banned_tokens:
                if start_pos in self.probs_cache:
                    # Setting near-zero probability for the banned token at the backtracking point.
                    self.probs_cache[start_pos][:, token_id] = 1e-10
            # Update the global forbidden_tokens list so that subsequent filtering rules out these tokens.
            self.forbidden_starting_tokens.update(banned_tokens)
            removal_decision = f"Removal threshold met (random < {removal_probability:.2f}). Backtracking from token pos {start_pos}."
            print(removal_decision)
            self._display_debug(removal_decision)

            # Get the probabilities BEFORE we delete the cache
            if start_pos in self.probs_cache:
                logits = self.probs_cache[start_pos]
                probs = torch.softmax(logits[0], dim=-1)
                top_probs, top_indices = torch.topk(probs, k=5)
                
                debug_tokens = "\nTop 5 tokens for next position:"
                for prob, idx in zip(top_probs.tolist(), top_indices.tolist()):
                    token_text = tokenizer.decode([idx])
                    debug_tokens += f"\n{token_text!r}: {prob:.8f}"

            # Now do the backtracking and cache clearing
            for _ in range(len(generated_sequence) - start_pos):
                generated_sequence.pop()
            to_del = [key for key in self.probs_cache if key >= start_pos]
            for key in to_del:
                del self.probs_cache[key]
            # Update safe index after backtracking.
            self.ignore_until = len(generated_sequence)

            # Print the debug tokens after backtracking
            if 'debug_tokens' in locals():
                print(debug_tokens)
                self._display_debug(debug_tokens)
            
            return generated_sequence
        else:
            # Keep branch: update ignore_until, so future detections ignore up to this point.
            phrase_token_count = len(tokenizer.encode(matched_phrase, add_special_tokens=False))
            new_ignore_until = start_pos + phrase_token_count
            self.ignore_until = max(self.ignore_until, new_ignore_until)
            keep_message = f"Keep decision (random >= {removal_probability:.2f}). Keeping phrase '{matched_phrase}' and setting ignore_until to {self.ignore_until}."
            print(keep_message)
            self._display_debug(keep_message)
            return generated_sequence

    def deslop(self, generated_sequence: List[int], prompt_length: int, newly_added_count: int = 1):
        # Initialize the ignore_until value (the safe token index) on first call.
        if self.ignore_until is None:
            self.ignore_until = prompt_length

        # Decode only a tail of the sequence (as before)
        tail_tokens = generated_sequence[-(self.max_slop_phrase_length + newly_added_count + 5):]
        inference_tail = self.tokenizer.decode(tail_tokens, skip_special_tokens=True)
        
        # Call detection—but only look for phrases starting after self.ignore_until.
        matched_phrase, start_pos = detect_disallowed_sequence(
            self.tokenizer,
            inference_tail,
            generated_sequence,
            prompt_length,
            self.slop_phrase_prob_adjustments,
            self.max_slop_phrase_length,
            self.min_slop_phrase_length,
            check_n_chars_back=16,
            min_index=self.ignore_until
        )

        if matched_phrase:
            # (Optional: ensure ignore_until is at least this detected start)
            self.ignore_until = max(self.ignore_until, start_pos)
            if self.slow_debug:
                current_text = self.tokenizer.decode(generated_sequence[prompt_length:start_pos])
                matched_phrase_to_display = self.tokenizer.decode(generated_sequence[start_pos:], skip_special_tokens=True)
                highlighted_text = f"{current_text}<span style='color: red;'>{matched_phrase_to_display}</span>"
                if self.inference_output:
                    with self.inference_output:
                        self.inference_output.clear_output(wait=True)
                        display(HTML(f"<div style='white-space: pre-wrap;'>{highlighted_text}</div>"))
            debug_info = f"Detected slop phrase '{matched_phrase}' at position {start_pos}."
            self._display_debug(debug_info)

            # Handle using our new binary decision approach.
            generated_sequence = self._handle_disallowed_sequence(
                matched_phrase=matched_phrase,
                start_pos=start_pos,
                generated_sequence=generated_sequence,
                probs_cache=self.probs_cache,
                adjustment_strength=self.adjustment_strength,
                slow_debug=self.slow_debug,
                tokenizer=self.tokenizer,
                inference_output=self.inference_output,
                debug_output=self.debug_output,
                debug_delay=self.debug_delay,
                prompt_length=prompt_length,
            )

            return generated_sequence
        return False

    def _display_debug(self, message: str):
        """
        Displays debug information in the debug_output widget.
        """
        if self.debug_output:
            with self.debug_output:
                self.debug_output.clear_output(wait=True)
                display(HTML(f"<pre>{message}</pre>"))


#class SlopPhraseStoppingCriteria:
#    def __init__(self, tokenizer: PreTrainedTokenizer, slop_phrase_sequences: Dict[Tuple[int, ...], float], max_slop_phrase_length: int):
#        self.tokenizer = tokenizer
#        self.slop_phrase_sequences = slop_phrase_sequences
#        self.max_slop_phrase_length = max_slop_phrase_length

    


class CustomSlopPhraseStoppingCriteria(StoppingCriteria):
    def __init__(self, tokenizer, max_slop_phrase_length, min_slop_phrase_length, prompt_length, slop_phrase_prob_adjustments):
        self.tokenizer = tokenizer
        self.max_slop_phrase_length = max_slop_phrase_length
        self.min_slop_phrase_length = min_slop_phrase_length
        self.slop_phrase_prob_adjustments = slop_phrase_prob_adjustments
        self.prompt_length = prompt_length

    # !! For some reason this isn't reliably triggered every token
    # which means we might have output a slop phrase a token back or so.
    # Not sure why!
    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        # Combine previous tokens with newly generated tokens
        self.previous_tokens = input_ids[0].tolist()

        inference = self.tokenizer.decode(self.previous_tokens[self.prompt_length:], skip_special_tokens=True)

        matched_phrase, start_pos = detect_disallowed_sequence(self.tokenizer,
                                                               inference,
                                                               self.previous_tokens,
                                                               self.prompt_length,
                                                               self.slop_phrase_prob_adjustments,
                                                               self.max_slop_phrase_length,
                                                               self.min_slop_phrase_length)
        if matched_phrase:
            #print('matched', matched_phrase)
            return True
        return False
