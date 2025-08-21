import torch
from tokenizers import Tokenizer
import openai
from torch.nn import functional as F
# from nltk.tokenize import word_tokenize
# from collections import Counter
import numpy as np
import hashlib

class Watermark():
    def __init__(self,
                 device: torch.device = None,
                 watermark_tokenizer: Tokenizer = None,
                 watermark_model = None,
                 auxilary_tokenizer: Tokenizer = None,
                 auxilary_model = None,
                 alpha: float = 0.0,
                 top_k: int = 50,
                 top_p: float = 0.9,
                 repetition_penalty: float = 1.0,
                 no_repeat_ngram_size: int = 0,
                 max_new_tokens: int = 230,
                 min_new_tokens: int = 170,
                 key: int = 0,
                 T: float = 1.0,
                 start: int = 1,
                 ):
        self.device = device
        self.watermark_tokenizer = watermark_tokenizer
        self.watermark_model = watermark_model
        self.auxilary_tokenizer = auxilary_tokenizer
        self.auxilary_model = auxilary_model
        self.alpha = alpha
        self.top_k = top_k
        self.top_p = top_p
        self.repetition_penalty = repetition_penalty
        self.no_repeat_ngram_size = no_repeat_ngram_size
        self.max_new_tokens = max_new_tokens
        self.min_new_tokens = min_new_tokens
        self.key = key
        self.T = T
        self.start = start

    def _calc_banned_ngram_tokens(self, prev_input_ids: torch.Tensor, num_hypos: int, no_repeat_ngram_size: int, cur_len: int) -> None:
        """Copied from fairseq for no_repeat_ngram in beam_search"""
        if cur_len + 1 < no_repeat_ngram_size:
            # return no banned tokens if we haven't generated no_repeat_ngram_size tokens yet
            return [[] for _ in range(num_hypos)]
        generated_ngrams = [{} for _ in range(num_hypos)]
        for idx in range(num_hypos):
            gen_tokens = prev_input_ids[idx].tolist()
            generated_ngram = generated_ngrams[idx]
            for ngram in zip(*[gen_tokens[i:] for i in range(no_repeat_ngram_size)]):
                prev_ngram_tuple = tuple(ngram[:-1])
                generated_ngram[prev_ngram_tuple] = generated_ngram.get(prev_ngram_tuple, []) + [ngram[-1]]

        def _get_generated_ngrams(hypo_idx):
            # Before decoding the next token, prevent decoding of ngrams that have already appeared
            start_idx = cur_len + 1 - no_repeat_ngram_size
            ngram_idx = tuple(prev_input_ids[hypo_idx, start_idx:cur_len].tolist())
            return generated_ngrams[hypo_idx].get(ngram_idx, [])

        banned_tokens = [_get_generated_ngrams(hypo_idx) for hypo_idx in range(num_hypos)]
        return banned_tokens

    def _postprocess_next_token_scores(self, lprobs, batch_size, num_beams, prev_output_tokens, repetition_penalty, no_repeat_ngram_size):
        # _enforce_repetition_penalty
        if repetition_penalty != 1.0:
            for i in range(batch_size * num_beams):
                for previous_token in set(prev_output_tokens[i].tolist()):
                    # if score < 0 then repetition penalty has to multiplied to reduce the previous token probability
                    if lprobs[i, previous_token] < 0:
                        lprobs[i, previous_token] *= repetition_penalty
                    else:
                        lprobs[i, previous_token] /= repetition_penalty
        
        # lower eos token prob to zero if min_length is not reached
        if prev_output_tokens.size(1) < self.min_new_tokens:
            lprobs[:, self.watermark_tokenizer.eos_token_id] = -float("Inf")
        
        if no_repeat_ngram_size > 0:
            # calculate a list of banned tokens to prevent repetitively generating the same ngrams
            num_batch_hypotheses = batch_size * num_beams
            # from fairseq: https://github.com/pytorch/fairseq/blob/a07cb6f40480928c9e0548b737aadd36ee66ac76/fairseq/sequence_generator.py#L345
            banned_batch_tokens = self._calc_banned_ngram_tokens(
                prev_output_tokens, num_batch_hypotheses, no_repeat_ngram_size, prev_output_tokens.size(1)
            )
            for i, banned_tokens in enumerate(banned_batch_tokens):
                lprobs[i, banned_tokens] = -float("inf")

    def _top_k_top_p_filtering(
        self,
        logits: torch.Tensor,
        top_k: int = 0,
        top_p: float = 1.0,
        filter_value: float = -float("Inf"),
        min_tokens_to_keep: int = 1,
    ) -> torch.Tensor:
        """ 
        Filter a distribution of logits using top-k and/or nucleus (top-p) filtering
        Args:
            logits: logits distribution shape (batch size, vocabulary size)
            if top_k > 0: keep only top k tokens with highest probability (top-k filtering).
            if top_p < 1.0: keep the top tokens with cumulative probability >= top_p (nucleus filtering).
            Make sure we keep at least min_tokens_to_keep per batch example in the output
        """
        if top_k > 0:
            top_k = min(max(top_k, min_tokens_to_keep), logits.size(-1))  # Safety check
            # Remove all tokens with a probability less than the last token of the top-k
            indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
            logits[indices_to_remove] = filter_value

        if top_p < 1.0:
            sorted_logits, sorted_indices = torch.sort(logits, descending=True)
            cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

            # Remove tokens with cumulative probability above the threshold (token with 0 are kept)
            sorted_indices_to_remove = cumulative_probs > top_p
            if min_tokens_to_keep > 1:
                # Keep at least min_tokens_to_keep (set to min_tokens_to_keep-1 because we add the first one below)
                sorted_indices_to_remove[..., :min_tokens_to_keep] = 0
            # Shift the indices to the right to keep also the first token above the threshold
            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
            sorted_indices_to_remove[..., 0] = 0

            # scatter sorted tensors to original indexing
            indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
            logits[indices_to_remove] = filter_value
        return logits
    
    def _top_p(self, logits, temperature, top_p):
        """ An utility function for top_p sampling """

            # Check for NaNs or Infs in logits
        # if torch.isnan(logits).any() or torch.isinf(logits).any():
        #     raise ValueError("Logits contain NaNs or Infs")
        
        probs = torch.softmax(logits / temperature, dim=-1)
        probs_sort, probs_idx = torch.sort(probs, dim=-1, descending=True)
        probs_sum = torch.cumsum(probs_sort, dim=-1)
        mask = probs_sum - probs_sort >= top_p
        probs_sort[mask] = 0.0

        # probs_sort.div_(probs_sort.sum(dim=-1, keepdim=True))
        # next_token_probs = torch.zeros(logits.shape).to(self.device).scatter_(-1, probs_idx, probs_sort) # probability of next token, ordered by vocab 
        
        # Normalize the probabilities
        probs_sort = probs_sort / probs_sort.sum(dim=-1, keepdim=True)

        # Check if probs_idx contains valid indices
        vocab_size = logits.size(-1)
        if probs_idx.max() >= vocab_size or probs_idx.min() < 0:
            raise ValueError("Indices in probs_idx are out of bounds")

        # Reconstruct the probabilities tensor according to the original vocabulary ordering
        next_token_probs = torch.zeros_like(logits).scatter_(-1, probs_idx, probs_sort)

        # Check for NaNs or Infs in next_token_probs
        if torch.isnan(next_token_probs).any() or torch.isinf(next_token_probs).any():
            raise ValueError("next_token_probs contain NaNs or Infs")

        return next_token_probs

    def _stopping_criteria(self, ids):
        # stop_words = ["word.", "word!", "word?", "word...", "word;"]
        # stop_words_ids = [tokenizer.encode(stop_word, return_tensors='pt', add_special_tokens=False)[0][-1].to(self.device) for stop_word in stop_words]
        
        if ids[0][-1] == self.watermark_tokenizer.eos_token_id:
            return True

        # if ids[0][-1] in stop_words_ids:
        #     if len(ids[0]) > self.min_new_tokens:
        #         return True
        return False
    
    def _stopping_criteria_batch(self, output_ids):
        # Define stop tokens
        stop_ids = [
            self.watermark_tokenizer.eos_token_id,
            self.watermark_tokenizer.encode('.', add_special_tokens=False)[-1],
            self.watermark_tokenizer.encode('!', add_special_tokens=False)[-1],
            self.watermark_tokenizer.encode('?', add_special_tokens=False)[-1],
        ]
        stopping_sequences = []
        min_length = self.min_new_tokens
        for seq in output_ids:
            if seq.size(0) < min_length:
                stopping_sequences.append(False)
            elif seq[-1].item() in stop_ids:
                stopping_sequences.append(True)
            else:
                stopping_sequences.append(False)
        return stopping_sequences

    def _get_zeta(self, probs):
        zeta = [i for i in range(len(probs[0])+1)]
        return zeta
    
    def _generate_U(self, key, x, size):
        # Convert the key and previous token id to a string, then encode it to bytes
        seed_input = f"{key}-{x}".encode('utf-8')
        # Use hashlib to create a hash of the input, and convert this hash to an integer seed
        seed = int(hashlib.sha256(seed_input).hexdigest(), 16) % 2**32
        # Set the seed for PyTorch's random number generator
        torch.manual_seed(seed)
        # Generate the vector of uniform random variables
        U = torch.rand(size).to(self.device)   # []
        return U.unsqueeze(0)

    def _generate_U_batch(self, key, x, size):
        # Convert the key and previous token id to a string, then encode it to bytes
        seed_input = f"{key}-{x}".encode('utf-8')
        # Use hashlib to create a hash of the input, and convert this hash to an integer seed
        seed = int(hashlib.sha256(seed_input).hexdigest(), 16) % 2**32
        # Set the seed for PyTorch's random number generator
        torch.manual_seed(seed)
        # Generate the vector of uniform random variables
        U = torch.rand(size).to(self.device)   # []
        return U
    
    def _gumble_sampling(self, probs, U):
        # # Convert the single probability to log scale
        # log_prob = torch.log(probs) 
        # # Generate Gumbel noise for this element
        # gumbel_noise = -torch.log(-torch.log(U))
        # # Compute the perturbed logit
        # perturbed_logit = log_prob + gumbel_noise
        perturbed_logit = torch.log(U)/probs
        # Since there's only one token, it is trivially selected
        next_id = torch.argmax(perturbed_logit)  # torch(next_id)
        return next_id.unsqueeze(0).unsqueeze(0)

    def _gumble_sampling_batch(self, probs, U):
        perturbed_logit = torch.log(U)/probs
        next_id = torch.argmax(perturbed_logit, dim=-1)  # torch(next_id)
        return next_id.unsqueeze(1)

    def _watermarking(self, probs, alpha, zeta, prev_id, key):
        # construction P_zeta
        probs_zeta = torch.minimum(probs, torch.tensor(alpha, device=self.device))
        zeta_tilde = torch.sum(torch.clamp(probs-alpha, min=0.0), dim=1, keepdim=True)
        probs_zeta = torch.cat((probs_zeta, zeta_tilde), dim=1)

        # sample from P_zeta
        U = self._generate_U(key=key, x=prev_id, size=len(probs_zeta[0]))
        zeta_i = self._gumble_sampling(probs_zeta, U)

        if zeta_i.item() != zeta[-1]:
            next_id = zeta_i
            # In the _watermarking function, after obtaining `next_id`
            vocab_size = len(probs[0])
            if next_id.item() >= vocab_size or next_id.item() < 0:
                raise ValueError(f"_watermarking generated an invalid token ID {next_id.item()}.")
            return next_id, zeta_i
        else:
            probs_tilde = torch.clamp(probs-alpha, min=0.0) / torch.sum(torch.clamp(probs-alpha, min=0.0))
            probs_tilde_i = torch.multinomial(probs_tilde, num_samples=1)
            # In the _watermarking function, after obtaining `next_id`
            vocab_size = len(probs[0])
            if probs_tilde_i.item() >= vocab_size or probs_tilde_i.item() < 0:
                raise ValueError(f"_watermarking generated an invalid token ID {probs_tilde_i.item()}.")
            return probs_tilde_i, zeta_i
    
    def _watermarking_batch(self, probs, U):
        # construction P_zeta
        probs_zeta = torch.minimum(probs, torch.tensor(self.alpha, device=self.device))
        zeta_tilde = torch.sum(torch.clamp(probs-self.alpha, min=0.0), dim=-1, keepdim=True)
        probs_zeta = torch.cat((probs_zeta, zeta_tilde), dim=-1)

        # sample from P_zeta
        zeta_i = self._gumble_sampling_batch(probs_zeta, U)

        # Determine the indices where zeta_i equals the last index
        special_index = probs_zeta.size(-1) - 1
        is_special = zeta_i == special_index
        is_special = is_special.squeeze(1)
        # Calculate adjusted probabilities for sampling
        adjusted_probs = torch.clamp(probs - self.alpha, min=0.0)
        adjusted_probs_sum = adjusted_probs.sum(dim=-1, keepdim=True)

        next_ids = torch.empty(zeta_i.size(), dtype=torch.long, device=self.device)
        next_ids[is_special] = torch.multinomial(adjusted_probs[is_special] / adjusted_probs_sum[is_special], num_samples=1)
        next_ids[~is_special] = zeta_i[~is_special]

        return next_ids
    
    # Un-watermarked text generation
    def generate_unwatermarked(self, prompt):
        input_ids = self.watermark_tokenizer.encode(prompt, return_tensors='pt').to(self.device)
        output_ids = torch.tensor([[]], dtype=torch.int64, device=self.device)

        attn = torch.ones_like(input_ids)
        past = None
        for i in range(self.max_new_tokens):
            with torch.no_grad():
                if past:
                    output = self.watermark_model(input_ids[:,-1:], attention_mask=attn, past_key_values=past)
                else:
                    output = self.watermark_model(input_ids)
            
            logits = output.logits[:,-1, :]
            self._postprocess_next_token_scores(logits, 1, 1, output_ids, repetition_penalty=self.repetition_penalty, no_repeat_ngram_size=self.no_repeat_ngram_size)   
            probs = self._top_p(logits, self.T, self.top_p)
            next_id = torch.multinomial(probs, num_samples=1)   # sampling

            input_ids = torch.cat((input_ids, next_id), dim=-1)   # update input_ids
            output_ids = torch.cat((output_ids, next_id), dim=-1)   # update output_ids

            past = output.past_key_values
            attn = torch.cat([attn, attn.new_ones((attn.shape[0], 1))], dim=-1)

            # stopping criteria
            stop = self._stopping_criteria(output_ids)
            if stop:
                output_text = self.watermark_tokenizer.decode(output_ids[0].tolist())
                return output_text
        
        output_text = self.watermark_tokenizer.decode(output_ids[0])
        return output_text

    # Watermarked text generation
    def generate_watermarked(self, prompt):
        input_ids = self.watermark_tokenizer.encode(prompt, return_tensors='pt').to(self.device)

        output_ids = torch.tensor([[]], dtype=torch.int64, device=self.device)
        # zeta_ids = torch.tensor([[]], dtype=torch.int64, device=self.device)
        attn = torch.ones_like(input_ids)
        past = None
        for t in range(self.max_new_tokens):
            with torch.no_grad():
                if past:
                    output = self.watermark_model(input_ids[:,-1:], attention_mask=attn, past_key_values=past)
                else:
                    output = self.watermark_model(input_ids)
            
            logits = output.logits[:,-1, :]
            self._postprocess_next_token_scores(logits, 1, 1, output_ids, repetition_penalty=self.repetition_penalty, no_repeat_ngram_size=self.no_repeat_ngram_size)
            probs = self._top_p(logits, self.T, self.top_p)
            start = self.start
            if t >= start:
                prev_id = '-'.join(output_ids[0][-start:].cpu().numpy().astype(str))
                next_id, zeta_i = self._watermarking(probs, self.alpha, self._get_zeta(probs), prev_id, key=self.key)   # watermarking
            else:
                next_id = torch.multinomial(probs, num_samples=1)
                zeta_i = torch.tensor([[0]]).to(self.device)

            input_ids = torch.cat((input_ids, next_id), dim=-1)   # update input_ids
            output_ids = torch.cat((output_ids, next_id), dim=-1)   # update output_ids
            # zeta_ids = torch.cat((zeta_ids, zeta_i), dim=-1)   # update zeta_ids

            past = output.past_key_values
            attn = torch.cat([attn, attn.new_ones((attn.shape[0], 1))], dim=-1)

            # stopping criteria
            stop = self._stopping_criteria(output_ids)
            if stop:
                output_text = self.watermark_tokenizer.decode(output_ids[0].tolist())
                return output_text
        
        output_text = self.watermark_tokenizer.decode(output_ids[0])
        return output_text

    # Un-watermarked text generation for batch
    def generate_unwatermarked_batch(self, prompt):
        input = self.watermark_tokenizer(
            prompt, 
            return_tensors='pt',
            padding='max_length', 
            truncation=True,
            max_length=32,
        ).to(self.device)

        input_ids = input['input_ids']
        attention_mask = input['attention_mask']
        batch_size = input_ids.size(0)

        inputed_texts = self.watermark_tokenizer.batch_decode(
            input_ids[:, :], skip_special_tokens=True
        )

        finished = torch.zeros(batch_size, dtype=torch.bool, device=self.device)
        output_ids = torch.empty(batch_size, 0, dtype=torch.long, device=self.device)
        past = None
        for t in range(self.max_new_tokens):
            with torch.no_grad():
                if past:
                    output = self.watermark_model(
                        input_ids[:,-1:], 
                        attention_mask=attention_mask, 
                        past_key_values=past,
                        use_cache=True
                    )
                else:
                    output = self.watermark_model(
                        input_ids,
                        attention_mask=attention_mask,
                        use_cache=True
                    )
            
            logits = output.logits[:,-1, :]
            self._postprocess_next_token_scores(
                lprobs=logits,
                batch_size=batch_size,
                num_beams=1,
                prev_output_tokens=output_ids[:, :],
                repetition_penalty=self.repetition_penalty,
                no_repeat_ngram_size=self.no_repeat_ngram_size,
            )
            probs = self._top_p(logits, self.T, self.top_p)
            next_id = torch.multinomial(probs, num_samples=1)

            input_ids = torch.cat((input_ids, next_id), dim=-1)   # update input_ids
            output_ids = torch.cat((output_ids, next_id), dim=-1)   # update output_ids

            past = output.past_key_values
            attention_mask = torch.cat(
                [attention_mask, torch.ones(batch_size, 1, device=self.device)], dim=1
            )   

            stop = self._stopping_criteria_batch(output_ids)
            finished |= torch.tensor(stop, device=self.device)
            if finished.all():
                break

            next_id[finished] = self.watermark_tokenizer.pad_token_id

        generated_texts = self.watermark_tokenizer.batch_decode(
            output_ids[:, :], skip_special_tokens=True
        )
        return generated_texts

    # Watermarked text generation for batch
    def generate_watermarked_batch(self, prompt):
        input = self.watermark_tokenizer(
            prompt, 
            return_tensors='pt',
            padding='max_length', 
            truncation=True,
            max_length=32,
        ).to(self.device)

        input_ids = input['input_ids']
        attention_mask = input['attention_mask']
        batch_size = input_ids.size(0)

        inputed_texts = self.watermark_tokenizer.batch_decode(
            input_ids[:, :], skip_special_tokens=True
        )

        finished = torch.zeros(batch_size, dtype=torch.bool, device=self.device)
        output_ids = torch.empty(batch_size, 0, dtype=torch.long, device=self.device)
        past = None
        for t in range(self.max_new_tokens):
            with torch.no_grad():
                if past:
                    output = self.watermark_model(
                        input_ids[:,-1:], 
                        attention_mask=attention_mask, 
                        past_key_values=past,
                        use_cache=True
                    )
                else:
                    output = self.watermark_model(
                        input_ids,
                        attention_mask=attention_mask,
                        use_cache=True
                    )
            
            logits = output.logits[:,-1, :]
            self._postprocess_next_token_scores(
                lprobs=logits,
                batch_size=batch_size,
                num_beams=1,
                prev_output_tokens=output_ids[:, :],
                repetition_penalty=self.repetition_penalty,
                no_repeat_ngram_size=self.no_repeat_ngram_size,
            )
            probs = self._top_p(logits, self.T, self.top_p)
            start = self.start
            if t >= start:
                # Prepare previous IDs for U generation
                prev_ids_list = ['-'.join(map(str, ids[-start:].tolist())) for ids in output_ids]
                U = torch.stack([
                    self._generate_U_batch(key=self.key, x=prev_id, size=probs.size(-1)+1)
                    for prev_id in prev_ids_list
                ]).to(self.device)

                next_id = self._watermarking_batch(probs, U)   # watermarking
            else:
                next_id = torch.multinomial(probs, num_samples=1)

            input_ids = torch.cat((input_ids, next_id), dim=-1)   # update input_ids
            output_ids = torch.cat((output_ids, next_id), dim=-1)   # update output_ids

            past = output.past_key_values
            attention_mask = torch.cat(
                [attention_mask, torch.ones(batch_size, 1, device=self.device)], dim=1
            )   

            stop = self._stopping_criteria_batch(output_ids)
            finished |= torch.tensor(stop, device=self.device)
            if finished.all():
                break

            next_id[finished] = self.watermark_tokenizer.pad_token_id

        generated_texts = self.watermark_tokenizer.batch_decode(
            output_ids[:, :], skip_special_tokens=True
        )
        return generated_texts

    # Watermark detection
    def detection(self, text):
        watermark_ids = self.auxilary_tokenizer.encode(text, 
                                return_tensors='pt', 
                                add_special_tokens=False).to(self.device)

        score = 0
        start = self.start

        with torch.no_grad():
            logits = self.auxilary_model(watermark_ids).logits

        for t in range(start, len(watermark_ids[0])):
            logtis_t = logits[:, t-1, :]
            probs = self._top_p(logtis_t, self.T, self.top_p)
            # construction P_zeta
            probs_zeta = torch.minimum(probs, torch.tensor(self.alpha, device=self.device))
            zeta_tilde = torch.sum(torch.clamp(probs-self.alpha, min=0.0), dim=1, keepdim=True)
            probs_zeta = torch.cat((probs_zeta, zeta_tilde), dim=1)
            # Sample from P_zeta
            prev_id = '-'.join(map(str, watermark_ids[0][t - start:t].cpu().numpy()))
            U = self._generate_U(key=self.key, x=prev_id, size=probs_zeta.shape[-1])
            zeta_i = self._gumble_sampling(probs_zeta, U)

            if zeta_i.item() != len(probs[0]):
                if zeta_i.item() == watermark_ids[0][t].item():
                    score += 1

        return score / (len(watermark_ids[0]) - start)
    
    # Watermark detection for batch
    def detection_batch(self, texts):
        # Tokenize the batch of texts
        inputs = self.auxilary_tokenizer(
            texts,
            return_tensors='pt',
            padding=True,
            truncation=True,
            max_length=512,
            add_special_tokens=False,
        ).to(self.device)

        ids = inputs['input_ids']               # Shape: (batch_size, seq_length)
        attention_mask = inputs['attention_mask']  # Shape: (batch_size, seq_length)
        batch_size, seq_length = ids.size()

        start = self.start

        with torch.no_grad():
            logits = self.auxilary_model(ids.long(), attention_mask=attention_mask).logits  # Shape: (batch_size, seq_length, vocab_size)

        # Initialize score and count tensors
        score = torch.zeros(batch_size, device=self.device)
        count = torch.zeros(batch_size, device=self.device)

        # Calculate the sequence lengths for each sample
        sequence_lengths = attention_mask.sum(dim=1)

        for t in range(start, seq_length):
            # Create a mask to select only the samples where t is less than sequence length
            valid_positions = sequence_lengths > t

            if valid_positions.any():
                logits_t = logits[valid_positions, t - 1, :]  # Select valid logits
                ids_t = ids[valid_positions]                  # Select valid ids

                probs = self._top_p(logits_t, self.T, self.top_p)
                # Construct P_zeta
                probs_zeta = torch.minimum(probs, torch.tensor(self.alpha, device=self.device))
                zeta_tilde = torch.sum(torch.clamp(probs - self.alpha, min=0.0), dim=1, keepdim=True)
                probs_zeta = torch.cat((probs_zeta, zeta_tilde), dim=1)

                # Prepare previous IDs for U generation
                prev_ids = ids_t[:, t - start:t].cpu().numpy()
                prev_ids_list = ['-'.join(map(str, prev_id)) for prev_id in prev_ids]

                U = torch.stack([
                    self._generate_U_batch(key=self.key, x=int(prev_id), size=probs_zeta.size(-1))
                    for prev_id in prev_ids_list
                ]).to(self.device)

                zeta_i = self._gumble_sampling_batch(probs_zeta, U)

                # Flatten zeta_i and true_ids
                zeta_i = zeta_i.view(-1)
                true_ids = ids_t[:, t].view(-1)

                # Update the score where zeta_i is not the special index and matches the true ID
                valid_idx = (zeta_i != probs.size(-1)) & (zeta_i == true_ids)
                score[valid_positions] += valid_idx.long()

                # Update the count for valid positions
                count[valid_positions] += 1

        final_score = (score / count).cpu().numpy()
        return final_score