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
                                check_n_chars_back: int = 16 # this moves the detection window back n chars, so we can detect phrases that were completed further back
                                ) -> Tuple[Tuple[int, ...], int]:
    
    inference = inference.lower()

    for char_offset in range(0, check_n_chars_back):
        for candidate_str_length in range(max_slop_phrase_length, min_slop_phrase_length - 1, -1):
            if candidate_str_length + char_offset > len(inference):
                continue
            candidate_str = inference[-(candidate_str_length + char_offset):len(inference)-char_offset]
            #print(candidate_str)
            if candidate_str in slop_phrase_prob_adjustments:
                # determine the token containing the beginning of the detected phrase
                #print('looking for', candidate_str,'in decoded text')
                for start_pos in range(len(generated_sequence)-1, prompt_length-1, -1):
                    candidate_seq = generated_sequence[start_pos:]
                    candidate_seq_decoded = tokenizer.decode(candidate_seq, skip_special_tokens=True).lower()
                    #print(candidate_seq_decoded)
                    if candidate_str in candidate_seq_decoded:
                        #print('detected!', candidate_str, time.time() - start)
                        return candidate_str, start_pos
                # if we reached here, something went wrong
                print('!! candidate_str not found after decoding')

    return None, -1

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
    ) -> List[int]:
        """
        Handles a detected disallowed sequence by downregulating or removing it from the generated_sequence.
        This version has a tighter, shorter logging format matching the user's requested style.
        """

        # We'll gather some text context. Let's take ~15 characters before & after:
        # so you can see a bit more around the flagged word, then bracket it.
        full_inference = tokenizer.decode(generated_sequence[self.prompt_length:], skip_special_tokens=True)
        # Lowercase check to find the matched phrase index
        matched_lower = matched_phrase.lower()
        # Where in the full_inference does the matched_phrase appear?
        idx = full_inference.lower().find(matched_lower)
        context_buffer = 15

        # If found, bracket it; otherwise just show the entire text
        if idx >= 0:
            context_start = max(0, idx - context_buffer)
            context_end = min(len(full_inference), idx + len(matched_phrase) + context_buffer)
            prefix = full_inference[context_start:idx]
            suffix = full_inference[idx + len(matched_phrase):context_end]
            # Insert brackets around the matched phrase
            short_context = f"{prefix}[{matched_phrase}]{suffix}"
        else:
            # fallback if something unexpected: entire text
            short_context = full_inference

        # Log the detection event
        detection_message = (
            f"[SlopDetector] Matched disallowed phrase:\n"
            f"'{short_context}'\n"
            "Initiating probability downregulation..."
        )
        print(detection_message)
        self._display_debug(detection_message)

        # Instead of applying exponentiation, we now interpret the adjustment factor
        # as a percentage removal chance.
        removal_probability = self.slop_phrase_prob_adjustments[matched_phrase.lower()] / 100.0

        # Identify all tokens we plan to downregulate
        slop_phrase_starting_token = generated_sequence[start_pos]
        starting_tokens = self.starting_tokens_lookup.get(matched_phrase.lower(), set())
        starting_tokens.add(slop_phrase_starting_token)

        for token_id in starting_tokens:
            if start_pos in self.probs_cache:
                # Get old probability for logging (if needed)
                p_old = self.probs_cache[start_pos][0, token_id].item()
                # Apply probabilistic downregulation
                if random.random() < removal_probability:
                    # Strong removal into near-zero probability
                    self.probs_cache[start_pos][:, token_id] *= 0.01
                else:
                    # Otherwise, apply a smaller reduction
                    self.probs_cache[start_pos][:, token_id] *= 0.95
            else:
                # If not in probs_cache, we can't do anything
                pass

        # Print one line with "Slop after regulating"
        # If we have matched_token_newprob data:
        if start_pos in self.probs_cache:
            token_str = tokenizer.decode([slop_phrase_starting_token], skip_special_tokens=True)
            p_old_float = p_old
            p_new_float = self.probs_cache[start_pos][0, slop_phrase_starting_token].item()
            # We also have the factor = adjustment^adjustment_strength
            factor_str = f"{removal_probability:.2f}"
            # Rank or blank
            rank_prefix = ""
            # E.g. "4) 'symphony' p=0.03125^20.0=0.00000"
            slop_after_line = f"{rank_prefix}'{token_str}' p={factor_str}={p_new_float:.5g}"
        else:
            # fallback
            slop_after_line = f"'{matched_phrase}' p={removal_probability:.2f}"

        after_regulation_msg = (
            "[SlopDetector] Slop after regulating:\n" f"{slop_after_line}"
        )
        print(after_regulation_msg)
        self._display_debug(after_regulation_msg)

        # Check if the starting token is still #1 after downregulation
        if start_pos in self.probs_cache:
            new_argmax = torch.argmax(self.probs_cache[start_pos], dim=1).item()
            if new_argmax == slop_phrase_starting_token and slow_debug:
                still_selected_message = (
                    f"[SlopDetector] The slop phrase '{matched_phrase}' was downregulated but is still selected!"
                )
                print(still_selected_message)
                self._display_debug(still_selected_message)
                return generated_sequence

        # If we get here, we backtrack
        removed_tokens = generated_sequence[start_pos:]
        removed_text = tokenizer.decode(removed_tokens, skip_special_tokens=True)
        backtrack_message = (
            f"Tokens from position {start_pos} onward have been removed."
        )
        print(backtrack_message)
        self._display_debug(backtrack_message)

        # Actually remove them
        for _ in range(len(generated_sequence) - start_pos):
            generated_sequence.pop()

        # Clear the probs_cache for positions after start_pos
        to_del = [key for key in self.probs_cache if key > start_pos]
        for key in to_del:
            del self.probs_cache[key]

        return generated_sequence

    def deslop(self, generated_sequence, prompt_length):
        self.prompt_length = prompt_length
        # After adding the token(s), check for disallowed sequences

        inference = self.tokenizer.decode(generated_sequence[prompt_length:], skip_special_tokens=True)

        matched_phrase, start_pos = detect_disallowed_sequence(self.tokenizer,
                                                               inference,
                                                               generated_sequence,
                                                               prompt_length,
                                                               self.slop_phrase_prob_adjustments,
                                                               self.max_slop_phrase_length,
                                                               self.min_slop_phrase_length)

        if matched_phrase:
            if self.slow_debug:
                current_text = self.tokenizer.decode(generated_sequence[prompt_length:start_pos])
                #print([current_text])
                matched_phrase_to_display = self.tokenizer.decode(generated_sequence[start_pos:], skip_special_tokens=True)
                #print([matched_phrase_to_display])
                # Add HTML formatting to display the matched_phrase in red
                highlighted_text = f"{current_text}<span style='color: red;'>{matched_phrase_to_display}</span>"
                
                with self.inference_output:
                    self.inference_output.clear_output(wait=True)
                    display(HTML(f"<div style='white-space: pre-wrap;'>{highlighted_text}</div>"))

            # Display debug information
            debug_info = f"Replacing '{matched_phrase}'"
            self._display_debug(debug_info)

            if self.slow_debug:
                #time.sleep(self.debug_delay)
                if self.debug_output:
                    with self.debug_output:
                        self.debug_output.clear_output(wait=True)                        

            # Handle the disallowed sequence using SlopPhraseHandler
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
