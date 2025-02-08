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

def compute_prefix_banned_tokens(tokenizer: PreTrainedTokenizer, phrase: str) -> set:
    """
    Compute token IDs for all possible prefixes of the phrase,
    considering both space-prefixed and non-space-prefixed versions.
    Only bans tokens that could start the phrase.
    """
    banned_tokens = set()
    phrase = phrase.lower()  # Normalize to lower-case for consistency
    
    # Create both versions - with and without leading space
    phrases_to_check = [phrase, ' ' + phrase]
    
    for phrase_variant in phrases_to_check:
        # Only generate prefixes, not all substrings
        for end_pos in range(1, len(phrase_variant) + 1):
            prefix = phrase_variant[:end_pos]
            # Skip single space tokens
            if prefix.strip() == "":
                continue
            # Get token IDs for this prefix
            token_ids = tokenizer.encode(prefix, add_special_tokens=False)
            # Only ban the first token of each prefix
            if token_ids:
                banned_tokens.add(token_ids[0])
    
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
        self.waiting_for_next_token = False  # Add this new attribute
        self.last_backtrack_pos = None       # Add this to track position

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

        # Get the probabilities BEFORE we adjust them
        if start_pos in self.probs_cache:
            logits = self.probs_cache[start_pos]
            original_probs = torch.softmax(logits[0], dim=-1)
            
            # Get ALL possible substrings as banned tokens (prefix-style)
            banned_tokens = compute_prefix_banned_tokens(tokenizer, matched_lower)
            
            # Store original probabilities for those tokens
            original_banned_probs = {token_id: original_probs[token_id].item() for token_id in banned_tokens}
            
            # Create a copy of logits for modifications
            adjusted_logits = logits.clone()
            
            # 1) Ban tokens that match any prefix of the phrase
            for token_id in banned_tokens:
                adjusted_logits[:, token_id] = float('-inf')

            # 2) Ban tokens that *contain* the matched_phrase (e.g., "orchestra" inside "orchestrated")
            #    We do it for all vocab to ensure partial matches don't slip through.
            vocab_size = adjusted_logits.shape[-1]
            containing_banned_tokens = []
            for token_id in range(vocab_size):
                # Skip already-banned tokens
                if adjusted_logits[0, token_id] == float('-inf'):
                    continue
                token_text = tokenizer.decode([token_id]).lower()
                if matched_lower in token_text:
                    containing_banned_tokens.append(token_id)
                    adjusted_logits[0, token_id] = float('-inf')

            # For debug output, compile original probabilities for newly banned tokens
            original_containing_banned_probs = {
                t_id: original_probs[t_id].item() for t_id in containing_banned_tokens
            }

            # Now get new probabilities after adjustments
            adjusted_probs = torch.softmax(adjusted_logits[0], dim=-1)
            top_probs, top_indices = torch.topk(adjusted_probs, k=5)
            
            debug_tokens = "\nTop 5 tokens for next position after adjustment:"
            for prob, idx in zip(top_probs.tolist(), top_indices.tolist()):
                token_text = tokenizer.decode([idx])
                debug_tokens += f"\n{token_text!r}: {prob:.8f}"
            
            debug_tokens += "\n\nBanned tokens (original → adjusted):"
            # Show prefix-banned tokens
            for token_id in banned_tokens:
                token_text = tokenizer.decode([token_id])
                orig_prob = original_banned_probs[token_id]
                new_prob = adjusted_probs[token_id].item()
                debug_tokens += (
                    f"\n{token_text!r}: {orig_prob:.8f} → {new_prob:.8f} (id: {token_id})"
                )
            # Show contain-banned tokens
            for token_id in containing_banned_tokens:
                token_text = tokenizer.decode([token_id])
                orig_prob = original_containing_banned_probs[token_id]
                new_prob = adjusted_probs[token_id].item()
                debug_tokens += (
                    f"\n{token_text!r}: {orig_prob:.8f} → {new_prob:.8f} (id: {token_id})"
                )

            # Update the cache with adjusted logits
            self.probs_cache[start_pos] = adjusted_logits

        # Decide whether to remove or keep the detected phrase.
        if random.random() < removal_probability:
            # Removal branch: zero out probabilities for newly banned tokens,
            # backtrack and remove everything from start_pos onward.
            if start_pos in self.probs_cache:
                # Setting near-zero probability for the banned tokens
                if 'banned_tokens' in locals():
                    for token_id in banned_tokens:
                        self.probs_cache[start_pos][:, token_id] = float('-inf')
                if 'containing_banned_tokens' in locals():
                    for token_id in containing_banned_tokens:
                        self.probs_cache[start_pos][:, token_id] = float('-inf')
                self.forbidden_starting_tokens.update(banned_tokens)
                self.forbidden_starting_tokens.update(containing_banned_tokens)

            removal_decision = f"Removal threshold met (random < {removal_probability:.2f}). Backtracking from token pos {start_pos}."
            print(removal_decision)
            self._display_debug(removal_decision)

            original_length = len(generated_sequence)
            for _ in range(len(generated_sequence) - start_pos):
                generated_sequence.pop()
            to_del = [key for key in self.probs_cache if key >= start_pos]
            for key in to_del:
                del self.probs_cache[key]

            self.ignore_until = len(generated_sequence)

            if 'debug_tokens' in locals():
                print(debug_tokens)
                self._display_debug(debug_tokens)
            
            backtrack_msg = (
                f"\nBacktracking summary:"
                f"\nOriginal sequence length: {original_length}"
                f"\nNew sequence length: {len(generated_sequence)}"
                f"\nBacktracked to position: {start_pos}"
                f"\nForbidden tokens count: {len(self.forbidden_starting_tokens)}"
            )
            print(backtrack_msg)
            self._display_debug(backtrack_msg)

            # Show possible next tokens if available
            if start_pos in self.probs_cache:
                logits = self.probs_cache[start_pos]
                next_probs = torch.softmax(logits[0], dim=-1)
                top_probs, top_indices = torch.topk(next_probs, k=5)
                next_tokens_msg = "\nPossible next tokens after backtracking:"
                for prob, idx in zip(top_probs.tolist(), top_indices.tolist()):
                    token_text = tokenizer.decode([idx])
                    next_tokens_msg += f"\n{token_text!r}: {prob:.8f}"
                print(next_tokens_msg)
                self._display_debug(next_tokens_msg)
            
            self.waiting_for_next_token = True
            self.last_backtrack_pos = start_pos
            
            return generated_sequence
        else:
            # Keep branch: update ignore_until so future detections ignore up to this point.
            phrase_token_count = len(tokenizer.encode(matched_phrase, add_special_tokens=False))
            new_ignore_until = start_pos + phrase_token_count
            self.ignore_until = max(self.ignore_until, new_ignore_until)
            keep_message = (
                f"Keep decision (random >= {removal_probability:.2f}). Keeping phrase '{matched_phrase}' "
                f"and setting ignore_until to {self.ignore_until}."
            )
            print(keep_message)
            self._display_debug(keep_message)
            return generated_sequence

    def deslop(self, generated_sequence: List[int], prompt_length: int, newly_added_count: int = 1):
        # If we previously backtracked and are waiting for the next token, show what that token actually became
        if self.waiting_for_next_token and len(generated_sequence) > self.last_backtrack_pos:
            next_token = generated_sequence[self.last_backtrack_pos]
            next_token_text = self.tokenizer.decode([next_token])
            
            # This print ensures we see the actual token and its ID that was picked right after backtracking
            next_token_msg = f"\n[SlopPhraseHandler] Actual token chosen after backtrack: {next_token_text!r} (id: {next_token})"
            print(next_token_msg)
            self._display_debug(next_token_msg)

            # Reset the flags now that we've seen the new token.
            self.waiting_for_next_token = False
            self.last_backtrack_pos = None

        # Initialize ignore_until on first call if needed
        if self.ignore_until is None:
            self.ignore_until = prompt_length

        # As before, decode just the tail to check slop phrases
        tail_tokens = generated_sequence[-(self.max_slop_phrase_length + newly_added_count + 5):]
        inference_tail = self.tokenizer.decode(tail_tokens, skip_special_tokens=True)

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

            # Handle disallowed sequence
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
