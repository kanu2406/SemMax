import re
import torch
import torch.nn as nn
import numpy as np
import numpy.random as npr
from typing import List
from scipy.stats import norm
from sentence_transformers import SentenceTransformer
from semmax_utils.repro import set_seed

class ResidualBlock(nn.Module):
    
    def __init__(self, dim, hidden_dim, dropout=0.1):
        super(ResidualBlock, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.LayerNorm(dim)
        )

    def forward(self, x):
        return x + self.net(x)

_SENT_END_RE = re.compile(r'[.!?]["\')\]]?\s*$')
# Shared canonical splitter — ONE function used by both sides
def split_sentences(text: str) -> List[str]:
    return [s.strip() for s in re.split(r'(?<=[.!?])\s+', text) if s.strip()]



def first_complete_sentence(chunk: str):
    
    parts = split_sentences(chunk)
    if not parts:
        return None
    first = parts[0]
   
    if len(parts) >= 2 or _SENT_END_RE.search(first):
        return first
    return None

class RobustNetwork(nn.Module):
    
    def __init__(self, embed_dim=768, hidden_dim=2048, num_blocks=2, dropout=0.15):
        super(RobustNetwork, self).__init__()
        self.embed_dim = embed_dim
        self.blocks = nn.ModuleList([
            ResidualBlock(embed_dim, hidden_dim, dropout) for _ in range(num_blocks)
        ])

    def forward(self, x):
        for block in self.blocks:
            x = block(x)
        out = x / torch.norm(x, dim=-1, keepdim=True) * np.sqrt(self.embed_dim)
        return out


# 2. GENERATOR

class SemanticWmGeneratorNN:
    def __init__(self,
            model,
            tokenizer,
            seed: int = 29,
            weights_path: str = "robust_hasher_weights.pth",
            salt_key: int = 35317,
            num_seq: int = 8,
            embedder_name: str = 'BAAI/bge-base-en-v1.5',
            window_size: int = 5,
            device: str = "cuda" if torch.cuda.is_available() else "cpu"
        ):

        self.model = model
        self.tokenizer = tokenizer
        self.salt_key = salt_key
        self.seed = seed
        self.num_seq = num_seq
        self.window_size = window_size
        self.device = device

        print(f"Loading Soft-Hash Semantic Embedder: {embedder_name}...")
        self.embedder = SentenceTransformer(embedder_name).to(self.device)
        self.embed_dim = self.embedder.get_sentence_embedding_dimension()
        self.sentence_r_cache = {}

        print(f"Loading Learned Hasher from {weights_path}...")
        self.hash_net = RobustNetwork().to(self.device)
        
        # Failsafe in case weights aren't trained yet
        try:
            self.hash_net.load_state_dict(torch.load(weights_path, map_location=device))
        except FileNotFoundError:
            print(f"Warning: {weights_path} not found. Using untrained weights.")
        self.hash_net.eval()

    def get_multi_context_r_vector(self, prev_sentences: List[str], alpha: float = 0.6) -> torch.Tensor:
        """
        Calculates EMA over a strict sliding window to ensure the 
        watermark auto-heals during copy-paste cropping attacks.
        """
        clean_history = [s.strip() for s in prev_sentences if s.strip()]
        window_size = self.window_size

   
        active_window = clean_history[-window_size:]

        if not active_window:
            rng = npr.default_rng(self.seed * self.salt_key)
            r = rng.standard_normal(self.embed_dim).astype(np.float32)
            return torch.from_numpy(r).to(self.device)

        with torch.no_grad():
            for i, sentence in enumerate(active_window):
                if sentence not in self.sentence_r_cache:
                    v = self.embedder.encode(sentence, convert_to_tensor=True,normalize_embeddings=True,show_progress_bar=False).to(self.device)
                    self.sentence_r_cache[sentence] = self.hash_net(v.unsqueeze(0)).squeeze(0)
                
                r_current = self.sentence_r_cache[sentence]

                if i == 0:
                    combined_r = r_current
                else:
                    combined_r = (alpha * r_current) + ((1 - alpha) * combined_r)

        combined_r = combined_r / torch.norm(combined_r) * np.sqrt(self.embed_dim)
        return combined_r

    @torch.no_grad()
    def generate(
        self,
        prompts: list[str],
        max_gen_len: int,
        top_p: float = 0.95,
        do_sample: bool = True,
        num_beams: int = 1,
        temperature: float = 0.85,
        max_tokens_per_step: int = 60,
       
    ) -> List[str]:
        
        bsz = len(prompts)
        res = ["" for _ in range(bsz)]
        
        # We prime the history with the prompt so EMA has an anchor from step 1
        current_full_texts = list(prompts)
        sentence_history = [[] for p in prompts] 
        
        generated_tokens = [0 for _ in range(bsz)]
        max_iterations = (max_gen_len // 10) + 10
        iteration = 0

        gen = torch.Generator(device=self.device)
        gen.manual_seed(self.seed)

        while min(generated_tokens) < max_gen_len and iteration < max_iterations:
            iteration += 1
            
            set_seed(self.seed + iteration)

            inputs = self.tokenizer(current_full_texts, return_tensors='pt', padding=True, truncation=True).to(self.device)
            input_lengths = [len(inputs['input_ids'][i]) for i in range(bsz)]

            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_tokens_per_step,
                top_p=top_p,
                temperature=temperature,
                do_sample=do_sample,
                num_beams=num_beams,
                num_return_sequences=self.num_seq,
                pad_token_id=self.tokenizer.eos_token_id,
            )

            new_chunks = []
            for ii in range(len(outputs)):
                batch_idx = ii // self.num_seq
                offset = input_lengths[batch_idx]
                chunk_tokens = outputs[ii][offset:]
                chunk_text = self.tokenizer.decode(chunk_tokens, skip_special_tokens=True)

                # match = re.search(r'.*?[.!?](?:\s|$)', chunk_text)
                # first_sentence = match.group(0).strip() if match else chunk_text.strip()

                first_sentence = first_complete_sentence(chunk_text)
                if first_sentence is None:
                    parts = split_sentences(chunk_text)
                    first_sentence = parts[0] if parts else chunk_text.strip()
                new_chunks.append(first_sentence)

            for batch_idx in range(bsz):
                if generated_tokens[batch_idx] >= max_gen_len:
                    continue 

                # 1. Get EMA Context Vector
                r_tensor = self.get_multi_context_r_vector(sentence_history[batch_idx], alpha=0.6)

                start_idx = batch_idx * self.num_seq
                end_idx = start_idx + self.num_seq
                prompt_drafts = new_chunks[start_idx:end_idx]

                # 2. Get Pure Semantic Scores
                v_drafts = self.embedder.encode(prompt_drafts, normalize_embeddings=True,convert_to_tensor=True,show_progress_bar=False)
                semantic_scores = torch.matmul(v_drafts, r_tensor)

                recent_history = " ".join(sentence_history[batch_idx][-2:]).lower()
               
                recent_words = set([w for w in re.findall(r'\b\w+\b', recent_history) if len(w) > 3])
                
            
                final_scores = semantic_scores

                best_idx = torch.argmax(final_scores).item()

                best_chunk = prompt_drafts[best_idx]
                append_text = (" " + best_chunk) if len(res[batch_idx]) > 0 else best_chunk

                res[batch_idx] += append_text
                current_full_texts[batch_idx] += append_text
                sentence_history[batch_idx].append(best_chunk)

                added_len = len(self.tokenizer.encode(best_chunk, add_special_tokens=False))
                generated_tokens[batch_idx] += added_len

        return res


# 3. DETECTOR 
class SemanticWmDetectorNN:
    def __init__(self,
            tokenizer,
            seed: int = 29,
            weights_path: str = "robust_hasher_weights.pth",
            salt_key: int = 35317,
            embedder_name: str = 'BAAI/bge-base-en-v1.5',
            window_size: int = 5,
            device: str = "cuda" if torch.cuda.is_available() else "cpu",
        ):

        self.tokenizer = tokenizer
        self.seed = seed
        self.salt_key = salt_key
        self.window_size = window_size
        self.device = device

        print(f"Loading Soft-Hash Semantic Detector Embedder: {embedder_name}...")
        self.embedder = SentenceTransformer(embedder_name).to(self.device)
        self.embed_dim = self.embedder.get_sentence_embedding_dimension()
        self.sentence_r_cache = {}

        print(f"Loading Learned Hasher from {weights_path}...")
        self.hash_net = RobustNetwork().to(self.device)
        try:
            self.hash_net.load_state_dict(torch.load(weights_path, map_location=device))
        except FileNotFoundError:
            pass
        self.hash_net.eval()
    
    def get_multi_context_r_vector(self, prev_sentences: List[str], alpha: float = 0.6) -> torch.Tensor:
        """
        Calculates EMA over a strict sliding window to ensure the 
        watermark auto-heals during copy-paste cropping attacks.
        """
        clean_history = [s.strip() for s in prev_sentences if s.strip()]
        window_size = self.window_size

        # 1. Truncate the history to the sliding window BEFORE applying EMA
        active_window = clean_history[-window_size:]

        if not active_window:
            rng = npr.default_rng(self.seed * self.salt_key)
            r = rng.standard_normal(self.embed_dim).astype(np.float32)
            return torch.from_numpy(r).to(self.device)

        with torch.no_grad():
            for i, sentence in enumerate(active_window):
                if sentence not in self.sentence_r_cache:
                    v = self.embedder.encode(sentence, convert_to_tensor=True,normalize_embeddings=True,show_progress_bar=False).to(self.device)
                    self.sentence_r_cache[sentence] = self.hash_net(v.unsqueeze(0)).squeeze(0)
                
                r_current = self.sentence_r_cache[sentence]

                # EMA Accumulation Logic 
                if i == 0:
                    combined_r = r_current
                else:
                    combined_r = (alpha * r_current) + ((1 - alpha) * combined_r)

        # Normalize to standard normal variance scale
        combined_r = combined_r / torch.norm(combined_r) * np.sqrt(self.embed_dim)
        return combined_r

    def get_scores_by_t(self, texts: List[str]) -> List[np.array]:
        bsz = len(texts)
        score_lists = []

        with torch.no_grad():
            for ii in range(bsz):
                text = texts[ii]
                if not text.strip():
                    score_lists.append(np.array([]))
                    continue

                # sentences = [s.strip() for s in re.split(r'(?<=[.!?])\s+', text) if s.strip()]
                sentences = split_sentences(text)

                if len(sentences) == 0:
                    score_lists.append(np.array([]))
                    continue

                chunk_scores = []

                for i, current_sentence in enumerate(sentences):
                    prev_sentences = sentences[:i]
                    
                    # We pass alpha=0.6 here to ensure detector matches generator
                    r_tensor = self.get_multi_context_r_vector(prev_sentences, alpha=0.6)

                    v_current = self.embedder.encode(current_sentence, convert_to_tensor=True, normalize_embeddings=True,show_progress_bar=False)

                    s = torch.dot(v_current, r_tensor).item()
                    
                    v_norm_sq = torch.dot(v_current, v_current).item()

                    chunk_scores.append((s, v_norm_sq))

                score_lists.append(np.array(chunk_scores))

        return score_lists

    

    def get_pvalues(self, scores: np.array, eps: float = 1e-200) -> float:
        if len(scores) == 0: return 0.5

        s_vals = [val[0] for val in scores]
        v_norm_sq_vals = [val[1] for val in scores]

        sum_s = np.sum(s_vals)
        sum_v_norm_sq = np.sum(v_norm_sq_vals)

        if sum_v_norm_sq == 0: return 0.5

        Z = sum_s / np.sqrt(sum_v_norm_sq)
        pvalue = norm.sf(Z)

        return max(pvalue, eps)

    