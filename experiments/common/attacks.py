import os
import re
import gc
import csv
import json
import math
import argparse

import numpy as np
import torch
from transformers import (
    AutoModelForCausalLM, AutoTokenizer, AutoModelForSeq2SeqLM,
    BertTokenizer, BertForMaskedLM, T5Tokenizer, T5ForConditionalGeneration,
)

from watermark.auto_watermark import AutoWatermark
from utils.transformers_config import TransformersConfig
from evaluation.tools.text_editor import (
    WordDeletion, SynonymSubstitution, ContextAwareSynonymSubstitution,
    BackTranslationTextEditor, DipperParaphraser,
)

from experiments.common.io import strip_prompt

BERT_PATH  = "bert-large-uncased"

MT_FWD = "Helsinki-NLP/opus-mt-en-de"
MT_BWD = "Helsinki-NLP/opus-mt-de-en"

PARA_MODEL  = "humarin/chatgpt_paraphraser_on_T5_base"
PARA_PREFIX = "paraphrase: "

DIPPER_TOK_PATH   = "google/t5-v1_1-xxl"
DIPPER_MODEL_PATH = "kalpeshk2011/dipper-paraphraser-xxl"





class MarianTranslator:
    def __init__(self, model_name, device):
        from transformers import MarianMTModel, MarianTokenizer
        self.tok = MarianTokenizer.from_pretrained(model_name)
        self.model = MarianMTModel.from_pretrained(model_name).to(device)
        self.device = device

    def translate(self, text):
        if not text or not text.strip():
            return text
        batch = self.tok([text], return_tensors="pt", truncation=True,
                         max_length=512, padding=True).to(self.device)
        with torch.no_grad():
            gen = self.model.generate(**batch, max_length=512, num_beams=4)
        return self.tok.batch_decode(gen, skip_special_tokens=True)[0]


class SmallParaphraser:
    """Local sentence-wise paraphrase attack (small T5, no API)."""
    def __init__(self, model_name, device, max_new_tokens=64, num_beams=4):
        self.tok = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(device)
        self.device = device
        self.max_new_tokens = max_new_tokens
        self.num_beams = num_beams

    def _para(self, sentence):
        ids = self.tok(PARA_PREFIX + sentence, return_tensors="pt",
                       truncation=True, max_length=256).to(self.device)
        with torch.no_grad():
            out = self.model.generate(**ids, max_new_tokens=self.max_new_tokens,
                                      num_beams=self.num_beams, do_sample=False)
        return self.tok.decode(out[0], skip_special_tokens=True).strip()

    def edit(self, text, reference=None):
        sents = [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]
        if not sents:
            return text
        return " ".join(self._para(s) for s in sents)



class CropAttack:
    """Copy-paste cropping: drop the first `frac` fraction of sentences."""
    def __init__(self, frac=0.5):
        self.frac = frac

    def edit(self, text, reference=None):
        s = sent_split(text)
        if len(s) < 2:
            return text
        k = min(int(round(len(s) * self.frac)), len(s) - 1)
        return " ".join(s[k:])




def build_attacks(names, device):
    ed = {}
    for n in names:
        if n == "clean":
            ed[n] = None
        elif n == "Word-D":
            ed[n] = WordDeletion(ratio=0.3)
        elif n == "Word-S":
            ed[n] = SynonymSubstitution(ratio=0.5)
        elif n == "Word-S-Context":
            ed[n] = ContextAwareSynonymSubstitution(
                ratio=0.5, tokenizer=BertTokenizer.from_pretrained(BERT_PATH),
                model=BertForMaskedLM.from_pretrained(BERT_PATH).to(device))
        elif n.startswith("Crop-"):
            ed[n] = CropAttack(frac=float(n.split("-", 1)[1]))
        elif n == "Translation":
            f, b = MarianTranslator(MT_FWD, device), MarianTranslator(MT_BWD, device)
            from evaluation.tools.text_editor import BackTranslationTextEditor
            ed[n] = BackTranslationTextEditor(translate_to_intermediary=f.translate,
                                              translate_to_source=b.translate)
        elif n == "Paraphrase-Small":
            ed[n] = SmallParaphraser(PARA_MODEL, device)
        elif n == "Doc-P-Dipper":
            print("  loading Dipper XXL (this is large)...")
            ed[n] = DipperParaphraser(
                tokenizer=T5Tokenizer.from_pretrained(DIPPER_TOK_PATH),
                model=T5ForConditionalGeneration.from_pretrained(
                    DIPPER_MODEL_PATH, device_map="auto"),
                lex_diversity=60, order_diversity=0, sent_interval=1)
                # max_new_tokens=100, do_sample=True, top_p=0.75, top_k=None)
        else:
            raise ValueError(f"unknown attack {n}")
    return ed




























def apply_attack(text, prompt, editor):
    text = strip_prompt(text, prompt)
    if editor is None:
        return text
    try:
        return editor.edit(text, prompt)
    except TypeError:
        return editor.edit(text)




def run_attack(method, attack, editor, wm, recs, out_dir):
    path = os.path.join(out_dir, f"{method}__{attack}.jsonl")
    done, _ = load_outcome(path)
    todo = [r for r in recs if r["idx"] not in done]
    print(f"  [{attack}] {len(done)} done, {len(todo)} to attack+score")

    for k, r in enumerate(todo, 1):
        original = strip_prompt(r["watermarked_text"], r["prompt"])
        try:
            attacked = apply_attack(r["watermarked_text"], r["prompt"], editor)
            d, s, err = detect(wm, attacked)
        except Exception as e:
            # transient (e.g. translation) -> 'attack ...' prefix -> retried next run
            attacked, d, s, err = None, None, None, f"attack {type(e).__name__}: {e}"

        append_line(path, {"idx": r["idx"], "original_text": original,
                           "attacked_text": attacked, "detect": d, "score": s, "error": err})
        if k % 10 == 0 or k == len(todo):
            print(f"    {attack} {k}/{len(todo)} (idx {r['idx']}) score={s}")


