from typing import Union

import numpy as np
import torch
from scipy.stats import norm

from ..base import BaseWatermark, BaseConfig
from utils.transformers_config import TransformersConfig
from visualize.data_for_visualization import DataForVisualization

from semmax_utils.sem_max_nn import SemanticWmGeneratorNN, SemanticWmDetectorNN, split_sentences


class SemMaxConfig(BaseConfig):
    """Config class for SemMax. Reads config/SemMax.json and sets parameters."""

    def initialize_parameters(self) -> None:
        """Initialize algorithm-specific parameters."""
        self.weights_path = self.config_dict['weights_path']
        self.seed = self.config_dict['seed']
        self.salt_key = self.config_dict['salt_key']
        self.num_seq = self.config_dict['num_seq']
        self.window_size = self.config_dict['window_size']
        self.embedder_name = self.config_dict['embedder_name']
        self.alpha = self.config_dict['alpha']

        # generation length / sampling
        self.max_gen_len = self.config_dict['max_gen_len']
        self.top_p = self.config_dict['top_p']
        self.temperature = self.config_dict['temperature']
        self.max_tokens_per_step = self.config_dict['max_tokens_per_step']

        # detection threshold on the document Z statistic
        self.z_threshold = self.config_dict['z_threshold']
        self.p_val_threshold = self.config_dict['p_val_threshold']

        # If True, generate_watermarked_text returns prompt+continuation (KGW/SIR
        # idiom) and you MUST run TruncatePromptTextEditor at detect time. If False,
        # it returns the continuation only and you must NOT truncate. See the big
        # note in generate_watermarked_text.
        self.emit_full_text = self.config_dict.get('emit_full_text', False)

    @property
    def algorithm_name(self) -> str:
        """Return the algorithm name."""
        return 'SemMax'


class SemMax(BaseWatermark):
    """Top-level class for the SemMax algorithm."""

    def __init__(self, algorithm_config: "str | SemMaxConfig",
                 transformers_config: TransformersConfig | None = None,
                 *args, **kwargs) -> None:
        """
        Parameters:
            algorithm_config (str | SemMaxConfig): path to config JSON, or a config object.
            transformers_config (TransformersConfig): transformers model config.
        """
        if isinstance(algorithm_config, str):
            self.config = SemMaxConfig(algorithm_config, transformers_config)
        elif isinstance(algorithm_config, SemMaxConfig):
            self.config = algorithm_config
        else:
            raise TypeError("algorithm_config must be either a path string or a SemMaxConfig instance")

        c = self.config

        self.generator = SemanticWmGeneratorNN(
            model=c.generation_model,
            tokenizer=c.generation_tokenizer,
            seed=c.seed,
            weights_path=c.weights_path,
            salt_key=c.salt_key,
            num_seq=c.num_seq,
            embedder_name=c.embedder_name,
            window_size=c.window_size,
            alpha=c.alpha,
            device=c.device,
        )
        self.detector = SemanticWmDetectorNN(
            tokenizer=c.generation_tokenizer,
            seed=c.seed,
            weights_path=c.weights_path,
            salt_key=c.salt_key,
            embedder_name=c.embedder_name,
            window_size=c.window_size,
            alpha=c.alpha,
            device=c.device,
        )

        # Generator and detector run the identical embedder + hash net. Alias them so
        # steady-state VRAM is halved and detection reuses the r-vector cache built
        # during generation. (Detector loads its own copies first, then drops them.)
        self.detector.embedder = self.generator.embedder
        self.detector.hash_net = self.generator.hash_net
        self.detector.sentence_r_cache = self.generator.sentence_r_cache

        gcfg = c.generation_model.generation_config
        for k in ("no_repeat_ngram_size", "repetition_penalty"):
            if k in c.gen_kwargs:
                setattr(gcfg, k, c.gen_kwargs[k])

    # -------------------------------------------------------------- generation

    def generate_watermarked_text(self, prompt: str, *args, **kwargs) -> str:
        """Generate watermarked text."""
        continuation = self.generator.generate(
            [prompt],
            max_gen_len=self.config.max_gen_len,
            top_p=self.config.top_p,
            temperature=self.config.temperature,
            max_tokens_per_step=self.config.max_tokens_per_step,
        )[0]

        
        if self.config.emit_full_text:
            return (prompt + " " + continuation) if continuation else prompt
        return continuation

    def generate_unwatermarked_text(self, prompt: str, *args, **kwargs) -> str:
        """Generate unwatermarked text (same prompt handling as watermarked side)."""
        c = self.config
        encoded_prompt = c.generation_tokenizer(prompt, return_tensors="pt",
                                                add_special_tokens=True).to(c.device)
        with torch.no_grad():
            encoded = c.generation_model.generate(**encoded_prompt, **c.gen_kwargs)
        if self.config.emit_full_text:
            return c.generation_tokenizer.batch_decode(encoded, skip_special_tokens=True)[0]
        prompt_len = encoded_prompt["input_ids"].shape[1]
        return c.generation_tokenizer.decode(encoded[0][prompt_len:], skip_special_tokens=True)

    # -------------------------------------------------------------- detection

    def _document_statistic(self, text: str):
        """Return (z, p_value). Higher z => more likely watermarked."""
        rows = np.asarray(self.detector.get_scores_by_t([text])[0])
        if rows.size == 0:
            return 0.0, 0.5

        d = self.detector
        # Prefer the calibrated document z if detector.calibrate(...) was run;
        # otherwise fall back to the self-normalized Z (matches get_pvalues).
        # if all(hasattr(d, a) for a in ("MU1", "MUR", "SD1", "SDR")):
        #     s = rows[:, 0]
        #     n = len(s)
        #     mu = np.full(n, d.MUR)
        #     mu[0] = d.MU1
        #     num = float((s - mu).sum())
        #     den = float(np.sqrt(d.SD1 ** 2 + (n - 1) * d.SDR ** 2))
        #     z = num / den if den > 0 else 0.0
        # else:
        sum_s = float(rows[:, 0].sum())
        sum_v = float(rows[:, 1].sum())
        z = sum_s / np.sqrt(sum_v) if sum_v > 0 else 0.0

        return float(z), float(max(norm.sf(z), 1e-200))

    def detect_watermark(self, text: str, return_dict: bool = True, *args, **kwargs) -> Union[dict, tuple]:
        """Detect watermark in the input text."""
        z_score, p_val = self._document_statistic(text)
        # is_watermarked = z_score > self.config.z_threshold
        is_watermarked = p_val < self.config.p_val_threshold


        if return_dict:
            return {"is_watermarked": bool(is_watermarked), "score": float(z_score), "p-value" : float(p_val)}
        else:
            return (bool(is_watermarked), float(z_score), float(p_val))

    

    