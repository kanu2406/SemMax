# SemMax

SemMax is a sentence-level watermark that embeds a continuous semantic signal into
generated text. At each step the base model produces several candidate sentences, and
SemMax emits the one whose embedding best aligns with a context-dependent reference
direction. Because the signal is a real-valued alignment rather than a discrete
signal, it degrades gradually rather than abruptly under text modifications.


## 1. Setup

```bash
git clone https://github.com/kanu2406/SemMax.git
pip install -r requirements.txt
```

**Checkpoints (required for generation/detection).** Pretrained SemMax components are required to run the full generation and detection pipeline. These include:

 - the fine-tuned sentence encoder, trained on C4 (semmax_utils/finetuned_embedder)
 -  the reference network (semmax_utils/robust_weights.pth).

If you use the provided checkpoints, you can skip the training steps and proceed directly to generation and evaluation.

> **Note:** every script is run from the repo root with `PYTHONPATH=.` and all outputs go under `results/`.

---

## 2. Reproduce the figures from saved results

To generate paper's plots without doing generations again, the small summary CSVs
can be used found under `results/`.

```bash
PYTHONPATH=. python experiments/ablations/ema_context/alpha_ablation_robustness.py --plot_only
PYTHONPATH=. python experiments/ablations/num_drafts/num_drafts_ablation_plot.py
PYTHONPATH=. python experiments/ablations/entropy/temp_plot.py
PYTHONPATH=. python experiments/score_dist/score_dist_analysis.py
```

To regenerate everything end-to-end instead, follow the following steps.

---

## 3. Full pipeline

Order: **A. train components → B. main C4 dataset comparison → C. ablations → D. score
distribution → E. cross-domain.** B–E depend on the checkpoints from A.

### A. Train SemMax components (C4 dataset)

1. **Fine-tune the sentence encoder** :
   ```bash
   python semmax_utils/finetune_embedder.py \
     --model_name_or_path sentence-transformers/all-mpnet-base-v2 \
     --dataset_path semmax_utils/data/original-c4-texts-8000-pegasus-bigram=False-threshold=0.0-split \
     --output_dir semmax_utils/finetuned_embedder \
     --learning_rate 4e-5 --warmup_steps 50 --max_seq_length 64 \
     --num_train_epochs 3 --logging_steps 10 \
     --evaluation_strategy epoch --save_strategy epoch \
     --remove_unused_columns False --delta 0.8 --do_train --overwrite_output_dir
   ```
2. **Train the reference network** (produces `semmax_utils/robust_weights.pth`):
   ```bash
   python semmax_utils/train_reference_net.py
   ```

### B. Main comparison on C4 dataset

6. **Generate** watermarked + unwatermarked text for all three methods
   (SemMax, Watermax, KSEMSTAMP). Output: `results/generations/`.
   ```bash
   PYTHONPATH=. python experiments/main_c4/generate.py
   ```
7. **Robustness** (detectability under attacks, results saved in `results/robustness/`):
   ```bash
   PYTHONPATH=. python experiments/main_c4/robust_analysis.py
   ```
8. **Quality** (relative PPL, ROUGE-L):
   ```bash
   PYTHONPATH=. python experiments/main_c4/quality_analysis.py
   ```

### C. Ablations

All ablations use the same 100 C4 prompts. 

9. **Alpha (EMA context) **
   ```bash
   PYTHONPATH=. python experiments/ablations/ema_alpha/alpha_ablation_generate.py     
   PYTHONPATH=. python experiments/ablations/ema_alpha/alpha_ablation_robustness.py   
   ```
10. **Number of drafts (`num_seq`) **
    ```bash
    PYTHONPATH=. python experiments/ablations/num_drafts/num_drafts_ablation_generate.py
    PYTHONPATH=. python experiments/ablations/num_drafts/num_drafts_ablation_plot.py
    ```
11. **Entropy / temperature **
    ```bash
    PYTHONPATH=. python experiments/ablations/entropy/temp_generate.py
    PYTHONPATH=. python experiments/ablations/entropy/temp_plot.py
    ```

### D. Score-distribution analysis 

Uses the C4 generations and robustness scores.

12. ```bash
    PYTHONPATH=. python experiments/score_dist/normality_check.py        # null normality (Q-Q, stats)
    PYTHONPATH=. python experiments/score_dist/score_dist_analysis.py    # distributions + separation
    ```

### E. Cross-domain generalization (BookSum)

We evaluate on BookSum in two settings. **Cross-domain** which reuses the C4-trained components and **in-domain** where retraining is required on BookSum dataset domain.


**E1. Cross-domain**:
```bash
PYTHONPATH=. python experiments/main_c4/generate.py \
  --dataset_path dataset/booksum_processed.jsonl \
  --output_path results/generations_booksum_cross_domain

PYTHONPATH=. python experiments/main_c4/robust_analysis.py \
  --generations_path results/generations_booksum_cross_domain \
  --output_path results/robustness_booksum_cross_domain

PYTHONPATH=. python experiments/main_c4/quality_analysis.py \
  --dir results/generations_booksum_cross_domain
```

**E2. In-domain** (retrain components on BookSum first):
```bash
# retrain the encoder on BookSum
python semmax_utils/finetune_embedder.py \
  --model_name_or_path sentence-transformers/all-mpnet-base-v2 \
  --dataset_path dataset/booksum-pegasus-bigram=False-split \
  --output_dir semmax_utils/finetuned_embedder_booksum \
  --learning_rate 4e-5 --warmup_steps 50 --max_seq_length 64 \
  --num_train_epochs 3 --logging_steps 10 \
  --evaluation_strategy epoch --save_strategy epoch \
  --remove_unused_columns False --delta 0.8 --do_train --overwrite_output_dir

# retrain reference network on BookSum
python semmax_utils/train_reference_net.py \
  --data ./dataset/booksum-pegasus-bigram=False-split \
  --save_path ./semmax_utils/robust_weights_booksum.pth \
  --embedder_name ./semmax_utils/finetuned_embedder_booksum

 
# generate + evaluate with the BookSum-trained components
PYTHONPATH=. python experiments/generalization_booksum/generate_booksum.py    # -> results/generations_booksum_in_domain
PYTHONPATH=. python experiments/generalization_booksum/robust_booksum.py      # -> results/robustness_booksum_in_domain
```

