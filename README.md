
1. Build subset for training embedders

```
python semmax_utils/build_subset.py semmax_utils/semstamp-data/original-c4-texts --n 8000

```

2. generating paraphrases to train

```
python semmax_utils/paraphrase_gen.py semmax_utils/semstamp-data/original-c4-texts-8000
```

3. Splitting the dataset to finetune embedder
```
python semmax_utils/train_val_test_split.py semmax_utils/semstamp-data/original-c4-texts-8000-pegasus-bigram=False-threshold=0.0
```

4. Training

```
python semmax_utils/finetune_embedder.py   --model_name_or_path sentence-transformers/all-mpnet-base-v2   --dataset_path semmax_utils/semstamp-data/original-c4-texts-8000-pegasus-bigram=False-threshold=0.0-split   --output_dir semmax_utils/finetuned_embedder   --learning_rate 4e-5   --warmup_steps 50   --max_seq_length 64   --num_train_epochs 3   --logging_steps 10   --evaluation_strategy epoch   --save_strategy epoch   --remove_unused_columns False   --delta 0.8   --do_train   --overwrite_output_dir

```

5. Train second embedder

```
python semmax_utils/contrastive_learning_2.py

```

6. Generation for comparing all methods (Results saved in generations)

```
python generate.py

```

7. Robustness Analysis (Results saved in generations)

```
python robust_analysis.py

```
