from typing import Union

import numpy as np
import torch

from ..base import BaseWatermark, BaseConfig
from utils.transformers_config import TransformersConfig
# from visualize.data_for_visualization import DataForVisualization

from watermark.watermax.wm import NewRobustWmSentenceGenerator, NewGaussianSentenceWm


def _to_float(v, default):
    """Map JSON-friendly 'inf'/'nan'/number -> float (JSON has no inf/nan)."""
    if v is None:
        return default
    if isinstance(v, str):
        s = v.strip().lower()
        if s in ("inf", "+inf", "infinity"):
            return np.inf
        if s in ("-inf", "-infinity"):
            return -np.inf
        if s == "nan":
            return np.nan
        return float(s)
    return float(v)


class RobustGaussConfig(BaseConfig):
    """Config class for the Gaussian n-gram watermark."""

    def initialize_parameters(self) -> None:
        # shared watermark keys (MUST match between generator and detector)
        self.seed = self.config_dict['seed']
        self.seeding = self.config_dict['seeding']
        self.salt_key = self.config_dict['salt_key']
        self.ngram = self.config_dict['ngram']
        self.num_seq = self.config_dict['num_seq']

        # generation
        self.max_gen_len = self.config_dict['max_gen_len']
        self.n_splits = self.config_dict['n_splits']
        self.beam_chunk_size = self.config_dict['beam_chunk_size']
        self.top_p = self.config_dict['top_p']
        self.temperature = self.config_dict['temperature']
        self.do_sample = self.config_dict['do_sample']
        self.num_beams = self.config_dict['num_beams']
        self.eos_value = _to_float(self.config_dict.get('eos_value', 'inf'), np.inf)

        # detection
        self.scoring_method = self.config_dict.get('scoring_method', 'v1')
        self.pvalue_threshold = self.config_dict.get('pvalue_threshold', 0.05)

        # The detector splits the text into fixed windows of `split_len` tokens.
        # The generator makes n_splits windows of (max_gen_len // n_splits) tokens,
        # so detection windows line up with generation windows at this value.
        self.split_len = self.config_dict.get('split_len', None)
        if self.split_len is None:
            self.split_len = max(1, self.max_gen_len // max(1, self.n_splits))

        # KGW/SIR idiom: return prompt+continuation and let the detection pipeline's
        # TruncatePromptTextEditor strip the prompt. Safe here because detection is
        # self-synchronizing per token (unlike the sentence-chain SemMax method).
        self.emit_full_text = self.config_dict.get('emit_full_text', True)

    @property
    def algorithm_name(self) -> str:
        return 'Watermax'


class RobustGauss(BaseWatermark):
    """Top-level class for the Gaussian n-gram watermark."""

    def __init__(self, algorithm_config: "str | RobustGaussConfig",
                 transformers_config: TransformersConfig | None = None,
                 *args, **kwargs) -> None:
        if isinstance(algorithm_config, str):
            self.config = RobustGaussConfig(algorithm_config, transformers_config)
        elif isinstance(algorithm_config, RobustGaussConfig):
            self.config = algorithm_config
        else:
            raise TypeError("algorithm_config must be either a path string or a RobustGaussConfig instance")

        c = self.config

        self.generator = NewRobustWmSentenceGenerator(
            model=c.generation_model,
            tokenizer=c.generation_tokenizer,
            seed=c.seed,
            seeding=c.seeding,
            salt_key=c.salt_key,
            ngram=c.ngram,
            num_seq=c.num_seq,
            eos_value=c.eos_value,
        )
        self.detector = NewGaussianSentenceWm(
            tokenizer=c.generation_tokenizer,
            split_len=c.split_len,
            ngram=c.ngram,
            seed=c.seed,
            seeding=c.seeding,
            salt_key=c.salt_key,
        )

        
        gcfg = c.generation_model.generation_config
        for k in ("no_repeat_ngram_size", "repetition_penalty"):
            if k in c.gen_kwargs:
                setattr(gcfg, k, c.gen_kwargs[k])

    # -------------------------------------------------------------- generation

    def generate_watermarked_text(self, prompt: str, *args, **kwargs) -> str:
        # generator.generate returns full text (prompt + continuation), decoded.
        res = self.generator.generate(
            [prompt],
            max_gen_len=self.config.max_gen_len,
            top_p=self.config.top_p,
            do_sample=self.config.do_sample,
            num_beams=self.config.num_beams,
            temperature=self.config.temperature,
            n_splits=self.config.n_splits,
            beam_chunk_size=self.config.beam_chunk_size,
        )
        text = res[0]
        if self.config.emit_full_text:
            return text
        
        return text[len(prompt):] if text.startswith(prompt) else text

    def generate_unwatermarked_text(self, prompt: str, *args, **kwargs) -> str:
        c = self.config
        encoded_prompt = c.generation_tokenizer(prompt, return_tensors="pt",
                                                add_special_tokens=True).to(c.device)
        with torch.no_grad():
            encoded = c.generation_model.generate(**encoded_prompt, **c.gen_kwargs)
        return c.generation_tokenizer.batch_decode(encoded, skip_special_tokens=True)[0]

    # -------------------------------------------------------------- detection

    def _pvalue(self, text: str) -> float:
        score_lists = self.detector.get_scores_by_t(
            [text], scoring_method=self.config.scoring_method)
        rts = score_lists[0] if score_lists else []
        if rts is None or len(rts) == 0:
            return 1.0
        return float(self.detector.get_pvalues(rts))

    def detect_watermark(self, text: str, return_dict: bool = True, *args, **kwargs) -> Union[dict, tuple]:
        pvalue = self._pvalue(text)
       
        score = float(-np.log10(max(pvalue, 1e-300)))
        is_watermarked = bool(pvalue < self.config.pvalue_threshold)
        if return_dict:
            return {"is_watermarked": is_watermarked, "score": score, "p_value": pvalue}
        return (is_watermarked, score)

    