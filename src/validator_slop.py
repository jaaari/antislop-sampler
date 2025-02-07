import time
from typing import List, Dict, Tuple, Generator, Set, Union

import torch
from transformers import (
    PreTrainedTokenizer,
    StoppingCriteria,
)
from IPython.display import display, HTML
from ipywidgets import Output


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
        This updated version adds more descriptive logging and context around the matched phrase.
        """

        # Prepare context for logging: last 3 tokens + the matched token + next token (if available)
        context_window_start = max(self.prompt_length, start_pos - 3)
        context_window_end = min(len(generated_sequence), start_pos + len(matched_phrase.split()) + 1)
        context_tokens = generated_sequence[context_window_start:context_window_end]
        context_text = tokenizer.decode(context_tokens, skip_special_tokens=True)

        # Show matched phrase in bold inside context
        # (Basic approach: just replace the matched_phrase substring in the context with HTML bold markers)
        # If your matched_phrase has special tokens or multiple subwords, you may want more robust logic
        context_highlight = context_text.replace(matched_phrase, f"**{matched_phrase}**")

        # Log the detection event
        detection_message = (
            f"[SlopDetector] Matched disallowed phrase:\n"
            f"  Phrase: '{matched_phrase}'\n"
            f"  Context around match: '{context_highlight}'\n\n"
            f"Initiating probability downregulation..."
        )
        print(detection_message)
        self._display_debug(detection_message)

        # Downregulate the relevant tokens at the start_pos
        adjustment = self.slop_phrase_prob_adjustments[matched_phrase.lower()]

        # If slow_debug is on, optionally pause
        if slow_debug:
            time.sleep(debug_delay)

        # Show top-5 predicted tokens (before downregulation) at start_pos, if we have them
        # We'll only attempt this if that index is in self.probs_cache
        if start_pos in self.probs_cache:
            token_probs = self.probs_cache[start_pos]
            # Top-5 predicted
            top_values, top_indices = torch.topk(token_probs, 5, dim=1)

            prob_debug_info = "[SlopDetector] Top-5 predicted token candidates before downregulation:\n"
            for rank in range(5):
                token_id = top_indices[0, rank].item()
                token_str = tokenizer.decode([token_id], skip_special_tokens=True)
                prob_val = top_values[0, rank].item()
                prob_debug_info += f"  {rank+1}) '{token_str}' p={prob_val:.5f}\n"

            self._display_debug(prob_debug_info)
            print(prob_debug_info)

        # Identify starting tokens to downregulate
        slop_phrase_starting_token = generated_sequence[start_pos]
        starting_tokens = self.starting_tokens_lookup.get(matched_phrase.lower(), set())
        starting_tokens.add(slop_phrase_starting_token)

        # Show exactly which tokens we plan to downregulate
        downregulating_message = (
            f"[SlopDetector] Downregulating the probability of these tokens: "
            f"{', '.join([tokenizer.decode([t]) for t in starting_tokens])}\n"
            f"  Downregulation factor: {adjustment}^{adjustment_strength}"
        )
        print(downregulating_message)
        self._display_debug(downregulating_message)

        # Downregulate the tokens
        for token_id in starting_tokens:
            self.probs_cache[start_pos][:, token_id] *= adjustment ** adjustment_strength

        # Display top-5 again after we downregulate
        if start_pos in self.probs_cache:
            token_probs_after = self.probs_cache[start_pos]
            # Top-5 predicted
            top_values_after, top_indices_after = torch.topk(token_probs_after, 5, dim=1)

            prob_debug_info_after = "[SlopDetector] Top-5 predicted token candidates after downregulation:\n"
            for rank in range(5):
                token_id = top_indices_after[0, rank].item()
                token_str = tokenizer.decode([token_id], skip_special_tokens=True)
                prob_val = top_values_after[0, rank].item()
                prob_debug_info_after += f"  {rank+1}) '{token_str}' p={prob_val:.5f}\n"

            self._display_debug(prob_debug_info_after)
            print(prob_debug_info_after)

        # Check if the starting token would still be selected after downregulation
        if torch.argmax(self.probs_cache[start_pos]).item() == slop_phrase_starting_token:
            if slow_debug:
                still_selected_message = (
                    f"[SlopDetector] The slop phrase '{matched_phrase}' was downregulated by "
                    f"{round(1/(adjustment**adjustment_strength), 2)}x but is still selected!"
                )
                print(still_selected_message)
                self._display_debug(still_selected_message)
            return generated_sequence

        # Backtrack: remove tokens from the generated_sequence that are part of the disallowed sequence
        removed_tokens = generated_sequence[start_pos:]
        removed_text = tokenizer.decode(removed_tokens, skip_special_tokens=True)
        backtrack_message = (
            f"[SlopDetector] Backtracking to remove disallowed sequence: '{removed_text}'\n"
            f"Tokens from position {start_pos} onward have been removed."
        )
        print(backtrack_message)
        self._display_debug(backtrack_message)

        for _ in range(len(generated_sequence) - start_pos):
            generated_sequence.pop()

        # Clear the probs_cache ahead of start_pos since we've backtracked
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
