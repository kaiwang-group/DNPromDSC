# import os
# import sys
# import glob
# import random
# import itertools
# from collections import defaultdict, Counter
# import numpy as np
# import pandas as pd
# from scipy.stats import pearsonr
# import torch
# import torch.nn as nn
# from transformers import AutoTokenizer
# try:
#     import Levenshtein
#     HAS_LEVENSHTEIN = True
# except ImportError:
#     HAS_LEVENSHTEIN = False
#     print('未检测到 Levenshtein。建议安装：pip install Levenshtein')
# try:
#     from Bio import Align
#     HAS_BIOPYTHON = True
#     aligner = Align.PairwiseAligner()
#     aligner.mode = 'local'
#     aligner.match_score = 2.0
#     aligner.mismatch_score = -3.0
#     aligner.open_gap_score = -5.0
#     aligner.extend_gap_score = -2.0
# except ImportError:
#     HAS_BIOPYTHON = False
#     print('未检测到 biopython。Bio Alignment 将跳过。')
# RANDOM_SEED = 42
# NUM_LABELS = 5
# NUM_SETS_TO_EVALUATE = 10
# SEQ_LEN = 80
# SAMPLE_SIZE = 2000
# LEV_PAIRS = 1000
# BIO_PAIRS = 500
# NATURAL_MASTER_PATH = '/data/stu1/wrc3_pycharm_project/WRC_mask_diffusion/wrc_masked_dataset/wrc_class5_63468.txt'
# GENERATED_ROOT_DIR = '/data/stu1/wrc3_pycharm_project/WRC_mask_diffusion/wrc_gen_all/optimized/optimizer_ablation_3modes_full80_snapshots_10sets/full_optimizer/round_080'
# OUTPUT_ROOT_DIR = os.path.join(GENERATED_ROOT_DIR, 'results_evaluation_dnabert2_regression')
# PREDICTOR_MODEL_TAG = 'DNABERT2-Reg'
# PREDICTOR_EXP_NAME = '5fold_regression_ensemble'
# DNABERT2_BACKBONE_PATH = '/data/stu1/wrc3_pycharm_project/PromoDGDE_main/Predictor/wrc_class5_DNABERT2/DNABERT_2'
# DNABERT2_TOKENIZER_PATH = '/data/stu1/wrc3_pycharm_project/PromoDGDE_main/Predictor/fuxian_dnabert2_mlp/DNABERT_2'
# if DNABERT2_BACKBONE_PATH not in sys.path:
#     sys.path.append(DNABERT2_BACKBONE_PATH)
# try:
#     from bert_layers import BertModel
# except ImportError:
#     print(f'Error: cannot import bert_layers.py from {DNABERT2_BACKBONE_PATH}. Please check the local DNABERT2 directory.')
#     raise
# PREDICTOR_RESULTS_ROOT = '/data/stu1/wrc3_pycharm_project/PromoDGDE_main/Predictor/wrc_class5_DNABERT2/results_regression_5fold_seed42_ep150'
# PREDICTOR_CHECKPOINTS = [os.path.join(PREDICTOR_RESULTS_ROOT, f'fold{i}', 'best_model.pth') for i in range(1, 6)]
# STRENGTH_BIN_FILE = '/data/stu1/wrc3_pycharm_project/PromoDGDE_main/Data/SC/wrc_regression_63468_5fold/full_dataset_with_bins.csv'
# PREDICTOR_DEVICE_STR = 'cuda:1'
# PREDICT_BATCH_SIZE = 256
# PREDICT_MAX_LEN = 100
# PREDICT_KMER = 6
# PREDICT_DO_KMER = False
# MLP_HIDDEN_SIZE = 256
# DROPOUT_RATE = 0.2
# USE_LAST4 = True
# POOL_TYPE = 'attn'
# OUTPUT_REPORT_NAME = 'wrc_evaluation_report_dnabert2_regression.txt'
# OUTPUT_CSV_NAME = 'wrc_evaluation_data_dnabert2_regression.csv'
# OUTPUT_SUMMARY_CSV_NAME = 'wrc_evaluation_summary_dnabert2_regression.csv'
#
# def seed_everything(seed=42):
#     random.seed(seed)
#     np.random.seed(seed)
#     torch.manual_seed(seed)
#     if torch.cuda.is_available():
#         torch.cuda.manual_seed(seed)
#         torch.cuda.manual_seed_all(seed)
#
# class Logger:
#
#     def __init__(self, filepath):
#         os.makedirs(os.path.dirname(filepath), exist_ok=True)
#         self.file = open(filepath, 'w', encoding='utf-8')
#
#     def print(self, msg=''):
#         print(msg)
#         self.file.write(str(msg) + '\n')
#         self.file.flush()
#
#     def close(self):
#         self.file.close()
#
# def clean_acgt_seq(seq, max_len=80):
#     seq = str(seq).upper()
#     seq = seq.replace('U', 'T')
#     seq = seq.replace(' ', '')
#     seq = ''.join((ch for ch in seq if ch in 'ACGT'))
#     if len(seq) > max_len:
#         seq = seq[:max_len]
#     return seq
#
# def load_and_split_natural_data(path):
#     data_dict = defaultdict(list)
#     if not os.path.exists(path):
#         print(f'天然序列文件不存在：{path}')
#         return data_dict
#     with open(path, 'r') as f:
#         for line in f:
#             parts = line.strip().split()
#             if len(parts) < 2:
#                 continue
#             seq = clean_acgt_seq(parts[0], max_len=SEQ_LEN)
#             try:
#                 label = int(parts[1])
#             except Exception:
#                 continue
#             if label in range(NUM_LABELS) and len(seq) > 0:
#                 data_dict[label].append(seq)
#     return data_dict
#
# def get_set_dir(root_dir, set_idx):
#     return os.path.join(root_dir, f'set_{set_idx:02d}')
#
# def find_generated_file_for_label(set_dir, label):
#     exact_paths = [os.path.join(set_dir, f'zhou_unet_label_{label}.csv'), os.path.join(set_dir, f'promoter_Final_V15_97_{label}.csv')]
#     for path in exact_paths:
#         if os.path.isfile(path):
#             return path
#     patterns = [os.path.join(set_dir, f'*label_{label}*.csv'), os.path.join(set_dir, f'*label{label}*.csv'), os.path.join(set_dir, f'*_label{label}_*.csv'), os.path.join(set_dir, f'*_{label}.csv')]
#     skip_keywords = ['evaluation', 'report', 'summary', 'generated_all', 'all_labels', 'optimized_detail', 'optimization_summary', 'optimization_log', 'prediction', 'metrics']
#     for pattern in patterns:
#         files = sorted(glob.glob(pattern))
#         candidates = []
#         for f in files:
#             if not os.path.isfile(f):
#                 continue
#             base = os.path.basename(f).lower()
#             if any((keyword in base for keyword in skip_keywords)):
#                 continue
#             candidates.append(f)
#         if candidates:
#             return candidates[0]
#     return None
#
# def load_generated_data(path):
#     if path is None or not os.path.exists(path):
#         return []
#     try:
#         df = pd.read_csv(path)
#         if df.empty:
#             return []
#         if 'sequence' in df.columns:
#             col = 'sequence'
#         elif 'seq' in df.columns:
#             col = 'seq'
#         else:
#             col = df.columns[0]
#         seqs = []
#         for s in df[col].tolist():
#             seq = clean_acgt_seq(s, max_len=SEQ_LEN)
#             if len(seq) > 0:
#                 seqs.append(seq)
#         return seqs
#     except Exception as e:
#         print(f'读取生成序列失败：{path} | {e}')
#         return []
#
# def get_gc_content(seqs):
#     if not seqs:
#         return 0.0
#     gcs = []
#     for s in seqs:
#         if len(s) == 0:
#             continue
#         gc = (s.count('G') + s.count('C')) / len(s)
#         gcs.append(gc)
#     if not gcs:
#         return 0.0
#     return float(np.mean(gcs))
#
# def get_diversity(seqs):
#     if not seqs:
#         return 0.0
#     return len(set(seqs)) / len(seqs)
#
# def check_novelty(gen_seqs, train_set):
#     if not gen_seqs:
#         return 0.0
#     hits = sum((1 for s in gen_seqs if s in train_set))
#     return 1.0 - hits / len(gen_seqs)
#
# def calc_metrics_sampling(seqs_a, seqs_b=None, mode='intra', lev_pairs=1000, bio_pairs=500):
#     lev_dists = []
#     bio_scores = []
#     if not seqs_a:
#         return (-1, -1)
#     if mode == 'inter' and (not seqs_b):
#         return (-1, -1)
#     if HAS_LEVENSHTEIN:
#         pairs = min(lev_pairs, 20000)
#         for _ in range(pairs):
#             if mode == 'intra':
#                 if len(seqs_a) < 2:
#                     break
#                 s1, s2 = random.sample(seqs_a, 2)
#             else:
#                 s1 = random.choice(seqs_a)
#                 s2 = random.choice(seqs_b)
#             lev_dists.append(Levenshtein.distance(s1, s2))
#     if HAS_BIOPYTHON:
#         pairs = min(bio_pairs, 20000)
#         for _ in range(pairs):
#             if mode == 'intra':
#                 if len(seqs_a) < 2:
#                     break
#                 s1, s2 = random.sample(seqs_a, 2)
#             else:
#                 s1 = random.choice(seqs_a)
#                 s2 = random.choice(seqs_b)
#             raw_score = aligner.score(s1, s2)
#             max_possible = min(len(s1), len(s2)) * 2.0
#             identity = raw_score / max_possible if max_possible > 0 else 0.0
#             bio_scores.append(identity * 100.0)
#     mean_lev = float(np.mean(lev_dists)) if lev_dists else -1
#     mean_bio = float(np.mean(bio_scores)) if bio_scores else -1
#     return (mean_lev, mean_bio)
#
# def get_kmer_frequency(sequences, k=6):
#     kmers = [''.join(p) for p in itertools.product('ACGT', repeat=k)]
#     kmer_to_idx = {kmer: i for i, kmer in enumerate(kmers)}
#     counts = np.zeros(len(kmers), dtype=np.float64)
#     total = 0
#     for seq in sequences:
#         if len(seq) < k:
#             continue
#         for i in range(len(seq) - k + 1):
#             sub = seq[i:i + k]
#             idx = kmer_to_idx.get(sub, None)
#             if idx is not None:
#                 counts[idx] += 1
#                 total += 1
#     if total == 0:
#         return np.zeros(len(kmers), dtype=np.float64)
#     return counts / total
#
# def safe_pearson(x, y):
#     try:
#         pcc, _ = pearsonr(x, y)
#         return float(pcc)
#     except Exception:
#         return np.nan
#
# class AttnPool1d(nn.Module):
#
#     def __init__(self, hidden_size):
#         super().__init__()
#         self.attn = nn.Linear(hidden_size, 1)
#
#     def forward(self, x):
#         scores = self.attn(x).squeeze(-1)
#         weights = torch.softmax(scores, dim=-1)
#         pooled = torch.sum(x * weights.unsqueeze(-1), dim=1)
#         return pooled
#
# class DNABERT2_Hybrid_Regression(nn.Module):
#
#     def __init__(self, mlp_hidden_size=256, dropout_rate=0.2, use_last4=True, pool_type='attn'):
#         super().__init__()
#         self.bert = BertModel.from_pretrained(DNABERT2_BACKBONE_PATH)
#         self.use_last4 = use_last4
#         self.pool_type = pool_type
#         if hasattr(self.bert, 'config') and hasattr(self.bert.config, 'output_hidden_states'):
#             self.bert.config.output_hidden_states = use_last4
#         if hasattr(self.bert, 'config') and hasattr(self.bert.config, 'hidden_size'):
#             self.bert_hidden = self.bert.config.hidden_size
#         else:
#             self.bert_hidden = 768
#         if self.pool_type == 'attn':
#             self.attn_pool = AttnPool1d(self.bert_hidden)
#         else:
#             self.attn_pool = None
#         self.regressor = nn.Sequential(nn.LayerNorm(self.bert_hidden), nn.Linear(self.bert_hidden, mlp_hidden_size), nn.GELU(), nn.Dropout(dropout_rate), nn.Linear(mlp_hidden_size, 1))
#
#     def _get_token_reps(self, outputs):
#         hidden_states = None
#         last_hidden_state = None
#         if hasattr(outputs, 'last_hidden_state'):
#             last_hidden_state = outputs.last_hidden_state
#         elif isinstance(outputs, (list, tuple)) and len(outputs) > 0:
#             last_hidden_state = outputs[0]
#         if hasattr(outputs, 'hidden_states') and outputs.hidden_states is not None:
#             hidden_states = outputs.hidden_states
#         if self.use_last4 and hidden_states is not None and (len(hidden_states) >= 4):
#             token_reps = torch.stack(hidden_states[-4:], dim=0).mean(dim=0)
#         else:
#             if last_hidden_state is None:
#                 raise RuntimeError('Unable to parse DNABERT2 output.')
#             token_reps = last_hidden_state
#         return token_reps
#
#     def _pool(self, token_reps, attention_mask=None):
#         if self.pool_type == 'cls':
#             pooled = token_reps[:, 0, :]
#         else:
#             pooled = self.attn_pool(token_reps)
#         return pooled
#
#     def forward(self, input_ids, attention_mask=None):
#         outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=self.use_last4)
#         token_reps = self._get_token_reps(outputs)
#         pooled = self._pool(token_reps, attention_mask=attention_mask)
#         output = self.regressor(pooled)
#         return output
#
# def seq_to_kmer_text(seq, k=6):
#     seq = clean_acgt_seq(seq, max_len=SEQ_LEN)
#     if len(seq) < k:
#         return seq
#     return ' '.join((seq[i:i + k] for i in range(len(seq) - k + 1)))
#
# def safe_torch_load(path, device):
#     try:
#         return torch.load(path, map_location=device, weights_only=True)
#     except TypeError:
#         return torch.load(path, map_location=device)
#
# def strip_module_prefix(state_dict):
#     if not isinstance(state_dict, dict):
#         return state_dict
#     if any((k.startswith('module.') for k in state_dict.keys())):
#         state_dict = {k[len('module.'):] if k.startswith('module.') else k: v for k, v in state_dict.items()}
#     if any((k.startswith('_orig_mod.') for k in state_dict.keys())):
#         state_dict = {k[len('_orig_mod.'):] if k.startswith('_orig_mod.') else k: v for k, v in state_dict.items()}
#     return state_dict
#
# def load_strength_thresholds(csv_path, logger=None):
#
#     def log(msg):
#         if logger is None:
#             print(msg)
#         else:
#             logger.print(msg)
#     if not os.path.exists(csv_path):
#         raise FileNotFoundError(f'找不到区间文件：{csv_path}')
#     df = pd.read_csv(csv_path)
#     if 'strength' not in df.columns:
#         raise ValueError('文件中缺少 strength 列。')
#     if 'bin' not in df.columns:
#         raise ValueError('文件中缺少 bin 列。')
#     df['strength'] = pd.to_numeric(df['strength'], errors='coerce')
#     df['bin'] = pd.to_numeric(df['bin'], errors='coerce')
#     df = df.dropna(subset=['strength', 'bin'])
#     df['bin'] = df['bin'].astype(int)
#     found_bins = sorted(df['bin'].unique().tolist())
#     expected_bins = list(range(NUM_LABELS))
#     if found_bins != expected_bins:
#         raise RuntimeError(f'Bin 异常：expected={expected_bins}, found={found_bins}')
#     bin_ranges = {}
#     log('\n' + '=' * 100)
#     log('Regression strength intervals')
#     log('=' * 100)
#     for label in range(NUM_LABELS):
#         values = df.loc[df['bin'] == label, 'strength'].values
#         if len(values) == 0:
#             raise RuntimeError(f'Bin {label} 为空。')
#         min_v = float(np.min(values))
#         max_v = float(np.max(values))
#         bin_ranges[label] = (min_v, max_v)
#         log(f'Class {label}: N={len(values)} | Strength=[{min_v:.6f}, {max_v:.6f}]')
#     thresholds = []
#     for label in range(NUM_LABELS - 1):
#         left_max = bin_ranges[label][1]
#         right_min = bin_ranges[label + 1][0]
#         threshold = (left_max + right_min) / 2.0
#         thresholds.append(threshold)
#     thresholds = np.asarray(thresholds, dtype=np.float64)
#     log('\nStrength thresholds:')
#     for idx, value in enumerate(thresholds, start=1):
#         log(f't{idx} = {value:.6f}')
#     log('\nMapping:')
#     log(f'Class 0: x < {thresholds[0]:.6f}')
#     for label in range(1, NUM_LABELS - 1):
#         log(f'Class {label}: {thresholds[label - 1]:.6f} <= x < {thresholds[label]:.6f}')
#     log(f'Class 4: x >= {thresholds[-1]:.6f}')
#     log('=' * 100)
#     return (thresholds, bin_ranges)
#
# def strength_to_label(strengths, thresholds):
#     strengths = np.asarray(strengths, dtype=np.float64)
#     labels = np.digitize(strengths, bins=thresholds, right=False)
#     return labels.astype(np.int64)
#
# def load_ensemble_models(device, logger=None):
#
#     def log(msg):
#         if logger is None:
#             print(msg)
#         else:
#             logger.print(msg)
#     models = []
#     log('\nLoading DNABERT2 5-fold regression ensemble...')
#     for fold_idx, path in enumerate(PREDICTOR_CHECKPOINTS, start=1):
#         if not os.path.exists(path):
#             log(f'Fold {fold_idx}: MISSING | {path}')
#             continue
#         model = DNABERT2_Hybrid_Regression(mlp_hidden_size=MLP_HIDDEN_SIZE, dropout_rate=DROPOUT_RATE, use_last4=USE_LAST4, pool_type=POOL_TYPE).to(device)
#         try:
#             ckpt = safe_torch_load(path, device)
#             if isinstance(ckpt, dict) and 'model_state_dict' in ckpt:
#                 state_dict = ckpt['model_state_dict']
#             elif isinstance(ckpt, dict) and 'state_dict' in ckpt:
#                 state_dict = ckpt['state_dict']
#             else:
#                 state_dict = ckpt
#             state_dict = strip_module_prefix(state_dict)
#             model.load_state_dict(state_dict, strict=True)
#             model.eval()
#             models.append(model)
#             log(f'Fold {fold_idx}: OK | {path}')
#         except Exception as e:
#             log(f'Fold {fold_idx}: LOAD FAILED')
#             log(f'Error: {e}')
#     log(f'Successfully loaded {len(models)}/5 models.')
#     return models
#
# @torch.no_grad()
# def predict_ensemble_strength(models, tokenizer, sequences, device, batch_size=256):
#     if not models:
#         return np.asarray([], dtype=np.float32)
#     all_strengths = []
#     for start in range(0, len(sequences), batch_size):
#         batch = sequences[start:start + batch_size]
#         texts = []
#         for seq in batch:
#             seq = clean_acgt_seq(seq, max_len=SEQ_LEN)
#             if len(seq) == 0:
#                 continue
#             if PREDICT_DO_KMER:
#                 text = seq_to_kmer_text(seq, k=PREDICT_KMER)
#             else:
#                 text = seq
#             texts.append(text)
#         if not texts:
#             continue
#         encoding = tokenizer(texts, add_special_tokens=True, max_length=PREDICT_MAX_LEN, padding='max_length', truncation=True, return_attention_mask=True, return_tensors='pt')
#         input_ids = encoding['input_ids'].to(device, non_blocking=True)
#         attention_mask = encoding['attention_mask'].to(device, non_blocking=True)
#         fold_preds = []
#         for model in models:
#             pred = model(input_ids, attention_mask)
#             if pred.ndim != 2 or pred.shape[1] != 1:
#                 raise RuntimeError(f'Regression output should be (B,1), got {tuple(pred.shape)}')
#             fold_preds.append(pred.squeeze(1))
#         mean_strength = torch.stack(fold_preds, dim=0).mean(dim=0)
#         all_strengths.extend(mean_strength.detach().cpu().numpy().tolist())
#     return np.asarray(all_strengths, dtype=np.float32)
#
# def evaluate_label_control(models, tokenizer, gen_seqs, target_label, device, thresholds):
#     strengths = predict_ensemble_strength(models, tokenizer, gen_seqs, device, batch_size=PREDICT_BATCH_SIZE)
#     if len(strengths) == 0:
#         return None
#     pred_labels = strength_to_label(strengths, thresholds)
#     dist = dict(sorted(Counter(pred_labels).items()))
#     acc = float((pred_labels == target_label).mean())
#     mean_pred = float(pred_labels.mean())
#     mae = float(np.abs(pred_labels - target_label).mean())
#     return {'n_pred': int(len(pred_labels)), 'dist': dist, 'acc': acc, 'mean_pred': mean_pred, 'mae': mae}
#
# def evaluate_one_label(label, gen_path, nat_seqs, ensemble_models, tokenizer, device, thresholds, logger):
#     gen_seqs = load_generated_data(gen_path)
#     if not nat_seqs:
#         logger.print(f'Class_{label}: 天然序列为空。')
#         return None
#     if not gen_seqs:
#         logger.print(f'Class_{label}: 生成序列为空。')
#         return None
#     train_set_ref = set(nat_seqs)
#     curr_nat = random.sample(nat_seqs, SAMPLE_SIZE) if len(nat_seqs) > SAMPLE_SIZE else nat_seqs
#     curr_gen = gen_seqs[:SAMPLE_SIZE]
#     nat_gc = get_gc_content(nat_seqs)
#     gen_gc = get_gc_content(curr_gen)
#     gc_diff = abs(nat_gc - gen_gc)
#     diversity = get_diversity(curr_gen)
#     novelty = check_novelty(curr_gen, train_set_ref)
#     lev_nat, bio_nat = calc_metrics_sampling(curr_nat, mode='intra', lev_pairs=LEV_PAIRS, bio_pairs=BIO_PAIRS)
#     lev_gen, bio_gen = calc_metrics_sampling(curr_gen, mode='intra', lev_pairs=LEV_PAIRS, bio_pairs=BIO_PAIRS)
#     lev_inter, bio_inter = calc_metrics_sampling(curr_gen, curr_nat, mode='inter', lev_pairs=LEV_PAIRS, bio_pairs=BIO_PAIRS)
#     logger.print(f'\nProcessing Class_{label} ...')
#     logger.print(f'   [File] {gen_path}')
#     logger.print(f'   [General] GC={gen_gc:.4f} (Nat={nat_gc:.4f}) | GC_diff={gc_diff:.4f} | Div={diversity:.4f} | Nov={novelty:.4f}')
#     logger.print(f'   [EditDist] Intra-Nat={lev_nat:.1f} | Intra-Gen={lev_gen:.1f} | Inter={lev_inter:.1f}')
#     logger.print(f'   [BioAlign] Intra-Nat={bio_nat:.1f}% | Intra-Gen={bio_gen:.1f}% | Inter={bio_inter:.1f}%')
#     kmer_pcc_dict = {}
#     for k in [4, 5, 6]:
#         freq_nat = get_kmer_frequency(curr_nat, k)
#         freq_gen = get_kmer_frequency(curr_gen, k)
#         pcc = safe_pearson(freq_nat, freq_gen)
#         kmer_pcc_dict[k] = pcc
#         logger.print(f'   [K-mer={k}] PCC={pcc:.4f}')
#     pred_metrics = None
#     if ensemble_models:
#         pred_metrics = evaluate_label_control(ensemble_models, tokenizer, curr_gen, label, device, thresholds)
#         if pred_metrics is not None:
#             logger.print(f"   [Predict] Target={label} | N={pred_metrics['n_pred']}")
#             logger.print(f"   [Predict] Dist={pred_metrics['dist']}")
#             logger.print(f"   [Predict] Acc={pred_metrics['acc']:.4f}")
#             logger.print(f"   [Predict] MeanPred={pred_metrics['mean_pred']:.4f} | MAE={pred_metrics['mae']:.4f}")
#     row = {'Class_ID': label, 'File': gen_path, 'N_Generated_Loaded': len(gen_seqs), 'N_Evaluated': len(curr_gen), 'GC_Nat': nat_gc, 'GC_Gen': gen_gc, 'GC_Diff': gc_diff, 'Diversity': diversity, 'Novelty': novelty, 'Lev_Intra_Nat': lev_nat, 'Lev_Intra_Gen': lev_gen, 'Lev_Inter': lev_inter, 'Bio_Intra_Nat': bio_nat, 'Bio_Intra_Gen': bio_gen, 'Bio_Inter': bio_inter, 'K4_PCC': kmer_pcc_dict.get(4, np.nan), 'K5_PCC': kmer_pcc_dict.get(5, np.nan), 'K6_PCC': kmer_pcc_dict.get(6, np.nan), 'Predictor': f'{PREDICTOR_MODEL_TAG}/{PREDICTOR_EXP_NAME}'}
#     if pred_metrics is not None:
#         row.update({'Pred_N': pred_metrics['n_pred'], 'Pred_Acc': pred_metrics['acc'], 'Pred_MeanPred': pred_metrics['mean_pred'], 'Pred_MAE': pred_metrics['mae'], 'Pred_Dist': str(pred_metrics['dist'])})
#     return row
#
# def evaluate_one_set(set_idx, set_dir, nat_data_dict, ensemble_models, tokenizer, device, thresholds):
#     set_output_dir = os.path.join(OUTPUT_ROOT_DIR, f'set_{set_idx:02d}')
#     os.makedirs(set_output_dir, exist_ok=True)
#     output_report_path = os.path.join(set_output_dir, OUTPUT_REPORT_NAME)
#     output_csv_path = os.path.join(set_output_dir, OUTPUT_CSV_NAME)
#     output_summary_csv_path = os.path.join(set_output_dir, OUTPUT_SUMMARY_CSV_NAME)
#     logger = Logger(output_report_path)
#     logger.print('=' * 100)
#     logger.print(f'Evaluation Set {set_idx:02d}')
#     logger.print('=' * 100)
#     logger.print(f'Source set dir: {set_dir}')
#     logger.print(f'Output dir: {set_output_dir}')
#     logger.print(f'Predictor: {PREDICTOR_MODEL_TAG}/{PREDICTOR_EXP_NAME}')
#     logger.print(f'SAMPLE_SIZE={SAMPLE_SIZE}')
#     rows = []
#     for label in range(NUM_LABELS):
#         gen_path = find_generated_file_for_label(set_dir, label)
#         if gen_path is None:
#             logger.print(f'\nClass_{label}: 未找到生成 CSV。')
#             continue
#         nat_seqs = nat_data_dict.get(label, [])
#         row = evaluate_one_label(label=label, gen_path=gen_path, nat_seqs=nat_seqs, ensemble_models=ensemble_models, tokenizer=tokenizer, device=device, thresholds=thresholds, logger=logger)
#         if row is not None:
#             row['Set_ID'] = set_idx
#             rows.append(row)
#     if not rows:
#         logger.print('\n本套没有有效结果。')
#         logger.close()
#         return (None, None)
#     class_df = pd.DataFrame(rows)
#     class_df.to_csv(output_csv_path, index=False)
#     metrics = ['GC_Diff', 'Diversity', 'Novelty', 'Lev_Inter', 'Bio_Inter', 'K4_PCC', 'K5_PCC', 'K6_PCC', 'Pred_Acc', 'Pred_MAE', 'Pred_MeanPred']
#     summary = {'Set_ID': set_idx}
#     for metric in metrics:
#         if metric not in class_df.columns:
#             continue
#         values = pd.to_numeric(class_df[metric], errors='coerce').dropna()
#         if len(values) > 0:
#             summary[metric] = float(values.mean())
#     if 'Pred_MeanPred' in class_df.columns:
#         temp = class_df[['Class_ID', 'Pred_MeanPred']].dropna()
#         if len(temp) >= 2:
#             corr = np.corrcoef(temp['Class_ID'].values.astype(float), temp['Pred_MeanPred'].values.astype(float))[0, 1]
#             summary['Corr_Target_MeanPred'] = float(corr)
#     logger.print('\n' + '=' * 100)
#     logger.print(f'Set {set_idx:02d} Summary')
#     logger.print('=' * 100)
#     if 'Pred_Acc' in summary:
#         logger.print(f"Avg Pred Acc : {summary['Pred_Acc']:.4f}")
#     if 'Pred_MAE' in summary:
#         logger.print(f"Avg Pred MAE : {summary['Pred_MAE']:.4f}")
#     logger.print(f"Avg GC Diff  : {summary.get('GC_Diff', np.nan):.4f}")
#     logger.print(f"Avg 4-mer PCC: {summary.get('K4_PCC', np.nan):.4f}")
#     logger.print(f"Avg 5-mer PCC: {summary.get('K5_PCC', np.nan):.4f}")
#     logger.print(f"Avg 6-mer PCC: {summary.get('K6_PCC', np.nan):.4f}")
#     if 'Corr_Target_MeanPred' in summary:
#         logger.print(f"Corr target-mean_pred: {summary['Corr_Target_MeanPred']:.4f}")
#     pd.DataFrame([{'Metric': key, 'Value': value} for key, value in summary.items()]).to_csv(output_summary_csv_path, index=False)
#     logger.print('\nSaved:')
#     logger.print(output_report_path)
#     logger.print(output_csv_path)
#     logger.print(output_summary_csv_path)
#     logger.close()
#     return (class_df, summary)
#
# def make_mean_std_table(df, group_col=None):
#     exclude_columns = {'Set_ID', 'Class_ID', 'File', 'Predictor', 'Pred_Dist'}
#     numeric_cols = []
#     for col in df.columns:
#         if col in exclude_columns:
#             continue
#         converted = pd.to_numeric(df[col], errors='coerce')
#         if converted.notna().any():
#             numeric_cols.append(col)
#     rows = []
#     if group_col is None:
#         groups = [('Overall', df)]
#     else:
#         groups = list(df.groupby(group_col))
#     for group_name, group_df in groups:
#         row = {}
#         if group_col is not None:
#             row[group_col] = group_name
#         for metric in numeric_cols:
#             values = pd.to_numeric(group_df[metric], errors='coerce').dropna().values
#             if len(values) == 0:
#                 continue
#             row[metric + '_Mean'] = float(np.mean(values))
#             row[metric + '_Std'] = float(np.std(values, ddof=1)) if len(values) > 1 else 0.0
#             row[metric + '_Var'] = float(np.var(values, ddof=1)) if len(values) > 1 else 0.0
#             row[metric + '_N'] = int(len(values))
#         rows.append(row)
#     return pd.DataFrame(rows)
#
# def write_pretty_summary_report(set_summary_df, class_mean_std_df):
#     report_path = os.path.join(OUTPUT_ROOT_DIR, 'wrc_10sets_pretty_summary.txt')
#
#     def get_stats(metric):
#         if metric not in set_summary_df.columns:
#             return (np.nan, np.nan, np.nan, 0)
#         values = pd.to_numeric(set_summary_df[metric], errors='coerce').dropna().values.astype(float)
#         if len(values) == 0:
#             return (np.nan, np.nan, np.nan, 0)
#         mean_v = float(np.mean(values))
#         std_v = float(np.std(values, ddof=1)) if len(values) > 1 else 0.0
#         var_v = float(np.var(values, ddof=1)) if len(values) > 1 else 0.0
#         return (mean_v, std_v, var_v, len(values))
#     width = 120
#     with open(report_path, 'w', encoding='utf-8') as f:
#         f.write('=' * width + '\n')
#         f.write('10-SET GENERATED PROMOTER EVALUATION SUMMARY\n')
#         f.write('Predictor: DNABERT2 5-fold Regression Ensemble\n')
#         f.write('Label decision: mean predicted strength -> predefined strength interval\n')
#         f.write('=' * width + '\n\n')
#         f.write('=' * width + '\n')
#         f.write('[Metric 1: Basic Sequence Quality]\n')
#         f.write('=' * width + '\n\n')
#         metric_info = [('GC_Diff', 'GC Diff', '<0.05 is good', lambda x: x < 0.05), ('Diversity', 'Diversity', '>0.99 is good', lambda x: x > 0.99), ('Novelty', 'Novelty', 'higher is better', lambda x: x > 0.95)]
#         for metric, name, note, condition in metric_info:
#             mean_v, std_v, var_v, n = get_stats(metric)
#             if np.isnan(mean_v):
#                 continue
#             icon = '✅' if condition(mean_v) else '⚠️'
#             f.write(f"   • {name:<22}: {mean_v:.4f} ± {std_v:.4f} (Var={var_v:.6f}, N={n}){' ' * 8}{icon}  {note}\n")
#         f.write('\n\n')
#         f.write('=' * width + '\n')
#         f.write('[Metric 2: Global Similarity Gen vs Nat]\n')
#         f.write('=' * width + '\n\n')
#         mean_v, std_v, var_v, n = get_stats('Lev_Inter')
#         if not np.isnan(mean_v):
#             f.write(f"   • {'Edit Distance':<22}: {mean_v:.2f} ± {std_v:.2f} (Var={var_v:.6f}, N={n}){' ' * 8}✅  lower means closer to natural\n")
#         mean_v, std_v, var_v, n = get_stats('Bio_Inter')
#         if not np.isnan(mean_v):
#             f.write(f"   • {'Bio Alignment':<22}: {mean_v:.4f}% ± {std_v:.4f}% (Var={var_v:.6f}, N={n}){' ' * 8}✅  higher means closer to natural\n")
#         f.write('\n\n')
#         f.write('=' * width + '\n')
#         f.write('[Metric 3: Local Motif Similarity Avg PCC]\n')
#         f.write('=' * width + '\n\n')
#         for metric, name in [('K4_PCC', '4-mer PCC'), ('K5_PCC', '5-mer PCC'), ('K6_PCC', '6-mer PCC')]:
#             mean_v, std_v, var_v, n = get_stats(metric)
#             if np.isnan(mean_v):
#                 continue
#             icon = '✅' if mean_v > 0.9 else '⚠️'
#             f.write(f"   • {name:<22}: {mean_v:.4f} ± {std_v:.4f} (Var={var_v:.6f}, N={n}){' ' * 8}{icon}  target >0.90\n")
#         f.write('\n\n')
#         f.write('=' * width + '\n')
#         f.write('[Metric 4: Label Controllability by DNABERT2 5-fold Regression Ensemble]\n')
#         f.write('=' * width + '\n\n')
#         mean_v, std_v, var_v, n = get_stats('Pred_Acc')
#         if not np.isnan(mean_v):
#             f.write(f"   • {'Avg Acc':<22}: {mean_v:.4f} ± {std_v:.4f} (Var={var_v:.6f}, N={n}){' ' * 8}✅  higher is better\n")
#         mean_v, std_v, var_v, n = get_stats('Pred_MAE')
#         if not np.isnan(mean_v):
#             f.write(f"   • {'Avg MAE':<22}: {mean_v:.4f} ± {std_v:.4f} (Var={var_v:.6f}, N={n}){' ' * 8}✅  lower is better\n")
#         mean_v, std_v, var_v, n = get_stats('Corr_Target_MeanPred')
#         if not np.isnan(mean_v):
#             f.write(f"   • {'Corr target-mean_pred':<22}: {mean_v:.4f} ± {std_v:.4f} (Var={var_v:.6f}, N={n}){' ' * 8}✅  higher is better\n")
#         f.write('\n\n')
#         f.write('=' * width + '\n')
#         f.write('[Metric 5: Class-wise Label Controllability Across 10 Sets]\n')
#         f.write('=' * width + '\n\n')
#         if class_mean_std_df is not None and len(class_mean_std_df) > 0 and ('Pred_Acc_Mean' in class_mean_std_df.columns):
#             temp = class_mean_std_df.sort_values('Class_ID')
#             for _, row in temp.iterrows():
#                 label = int(row['Class_ID'])
#                 acc_mean = float(row['Pred_Acc_Mean'])
#                 acc_std = float(row.get('Pred_Acc_Std', 0.0))
#                 mae_mean = float(row.get('Pred_MAE_Mean', np.nan))
#                 mae_std = float(row.get('Pred_MAE_Std', np.nan))
#                 f.write(f'   • Class {label}: Acc={acc_mean:.4f} ± {acc_std:.4f}')
#                 if not np.isnan(mae_mean):
#                     f.write(f' | MAE={mae_mean:.4f} ± {mae_std:.4f}')
#                 f.write('\n')
#         f.write('\n')
#         f.write('=' * width + '\n')
#         f.write('Accuracy values are reported on a 0-1 scale.\n')
#         f.write('=' * width + '\n')
#     return report_path
#
# def write_final_report(set_summary_df, class_mean_std_df):
#     report_path = os.path.join(OUTPUT_ROOT_DIR, 'wrc_10sets_final_report_dnabert2_regression.txt')
#     with open(report_path, 'w', encoding='utf-8') as f:
#         f.write('=' * 110 + '\n')
#         f.write('10-SET GENERATED PROMOTER EVALUATION\n')
#         f.write('DNABERT2 5-fold regression ensemble\n')
#         f.write('=' * 110 + '\n\n')
#         f.write(f'Generated root : {GENERATED_ROOT_DIR}\n')
#         f.write(f'Output root    : {OUTPUT_ROOT_DIR}\n')
#         f.write(f'Predictor root : {PREDICTOR_RESULTS_ROOT}\n')
#         f.write(f'Random seed    : {RANDOM_SEED}\n')
#         f.write(f'Valid sets     : {len(set_summary_df)}\n\n')
#         columns = ['Set_ID', 'Pred_Acc', 'Pred_MAE', 'GC_Diff', 'K4_PCC', 'K5_PCC', 'K6_PCC', 'Corr_Target_MeanPred']
#         existing = [c for c in columns if c in set_summary_df.columns]
#         f.write(set_summary_df[existing].to_string(index=False, float_format=lambda x: f'{x:.4f}'))
#         f.write('\n\n')
#         if class_mean_std_df is not None and len(class_mean_std_df) > 0:
#             f.write('=' * 110 + '\n')
#             f.write('CLASS-LEVEL MEAN ± STD\n')
#             f.write('=' * 110 + '\n')
#             show_cols = ['Class_ID', 'Pred_Acc_Mean', 'Pred_Acc_Std', 'Pred_MAE_Mean', 'Pred_MAE_Std', 'GC_Diff_Mean', 'GC_Diff_Std', 'K4_PCC_Mean', 'K4_PCC_Std', 'K5_PCC_Mean', 'K5_PCC_Std', 'K6_PCC_Mean', 'K6_PCC_Std']
#             existing = [c for c in show_cols if c in class_mean_std_df.columns]
#             f.write(class_mean_std_df[existing].to_string(index=False, float_format=lambda x: f'{x:.4f}'))
#             f.write('\n')
#     return report_path
#
# def evaluate_10sets():
#     os.makedirs(OUTPUT_ROOT_DIR, exist_ok=True)
#     print('=' * 100)
#     print('10-SET PROMOTER EVALUATION')
#     print('=' * 100)
#     print(f'Generated root : {GENERATED_ROOT_DIR}')
#     print(f'Output root    : {OUTPUT_ROOT_DIR}')
#     print(f'Predictor root : {PREDICTOR_RESULTS_ROOT}')
#     print(f'Strength bins  : {STRENGTH_BIN_FILE}')
#     print(f'Seed           : {RANDOM_SEED}')
#     print('=' * 100)
#     if not os.path.isdir(GENERATED_ROOT_DIR):
#         print('ERROR: generated root does not exist:')
#         print(GENERATED_ROOT_DIR)
#         return
#     if not os.path.isdir(PREDICTOR_RESULTS_ROOT):
#         print('ERROR: predictor root does not exist:')
#         print(PREDICTOR_RESULTS_ROOT)
#         return
#     if not os.path.exists(STRENGTH_BIN_FILE):
#         print('ERROR: strength bin file does not exist:')
#         print(STRENGTH_BIN_FILE)
#         return
#     nat_data_dict = load_and_split_natural_data(NATURAL_MASTER_PATH)
#     if not nat_data_dict:
#         print('ERROR: natural dataset failed to load.')
#         return
#     print('\nNatural reference:')
#     for label in range(NUM_LABELS):
#         print(f'Class {label}: {len(nat_data_dict[label])}')
#     if 'cuda' in PREDICTOR_DEVICE_STR and torch.cuda.is_available():
#         device = torch.device(PREDICTOR_DEVICE_STR)
#     else:
#         device = torch.device('cpu')
#     print(f'\nPredictor device: {device}')
#     thresholds, bin_ranges = load_strength_thresholds(STRENGTH_BIN_FILE)
#     print(f'\nLoading tokenizer from: {DNABERT2_TOKENIZER_PATH}')
#     tokenizer = AutoTokenizer.from_pretrained(DNABERT2_TOKENIZER_PATH, trust_remote_code=True)
#     ensemble_models = load_ensemble_models(device)
#     if not ensemble_models:
#         print('ERROR: no regression model loaded successfully.')
#         return
#     if len(ensemble_models) < 5:
#         print(f'\nWARNING: only {len(ensemble_models)}/5 models loaded.')
#     all_class_dfs = []
#     all_set_summaries = []
#     for set_idx in range(1, NUM_SETS_TO_EVALUATE + 1):
#         set_dir = get_set_dir(GENERATED_ROOT_DIR, set_idx)
#         print('\n' + '=' * 100)
#         print(f'SET {set_idx:02d}/{NUM_SETS_TO_EVALUATE}')
#         print(f'Source directory: {set_dir}')
#         print('=' * 100)
#         if not os.path.isdir(set_dir):
#             print(f'WARNING: {set_dir} 不存在，跳过。')
#             continue
#         for label in range(NUM_LABELS):
#             file_path = find_generated_file_for_label(set_dir, label)
#             if file_path is None:
#                 print(f'Class {label}: MISSING')
#             else:
#                 print(f'Class {label}: {os.path.basename(file_path)}')
#         class_df, summary = evaluate_one_set(set_idx=set_idx, set_dir=set_dir, nat_data_dict=nat_data_dict, ensemble_models=ensemble_models, tokenizer=tokenizer, device=device, thresholds=thresholds)
#         if class_df is None:
#             continue
#         all_class_dfs.append(class_df)
#         all_set_summaries.append(summary)
#     if not all_set_summaries:
#         print('\nERROR: 没有任何一套得到有效结果。')
#         return
#     class_raw_df = pd.concat(all_class_dfs, axis=0, ignore_index=True)
#     set_summary_df = pd.DataFrame(all_set_summaries)
#     set_raw_path = os.path.join(OUTPUT_ROOT_DIR, 'wrc_10sets_set_level_summary_raw.csv')
#     set_summary_df.to_csv(set_raw_path, index=False)
#     set_mean_std_df = make_mean_std_table(set_summary_df, group_col=None)
#     set_mean_std_path = os.path.join(OUTPUT_ROOT_DIR, 'wrc_10sets_set_level_mean_std.csv')
#     set_mean_std_df.to_csv(set_mean_std_path, index=False)
#     class_raw_path = os.path.join(OUTPUT_ROOT_DIR, 'wrc_10sets_class_level_raw.csv')
#     class_raw_df.to_csv(class_raw_path, index=False)
#     class_mean_std_df = make_mean_std_table(class_raw_df, group_col='Class_ID')
#     class_mean_std_path = os.path.join(OUTPUT_ROOT_DIR, 'wrc_10sets_class_level_mean_std.csv')
#     class_mean_std_df.to_csv(class_mean_std_path, index=False)
#     print('\n' + '=' * 100)
#     print('10-SET FINAL RESULTS')
#     print('=' * 100)
#     print(f'Valid sets: {len(set_summary_df)}')
#     important_metrics = ['Pred_Acc', 'Pred_MAE', 'GC_Diff', 'Diversity', 'Novelty', 'Lev_Inter', 'Bio_Inter', 'K4_PCC', 'K5_PCC', 'K6_PCC', 'Corr_Target_MeanPred']
#     if len(set_mean_std_df) > 0:
#         summary_row = set_mean_std_df.iloc[0]
#         print('\nOverall mean ± std:')
#         for metric in important_metrics:
#             mean_col = metric + '_Mean'
#             std_col = metric + '_Std'
#             if mean_col not in summary_row.index:
#                 continue
#             mean_v = summary_row[mean_col]
#             std_v = summary_row[std_col] if std_col in summary_row.index else np.nan
#             print(f'{metric:<28}: {mean_v:.4f} ± {std_v:.4f}')
#     print('\nClass-level Pred_Acc (10-set mean ± std):')
#     if 'Pred_Acc_Mean' in class_mean_std_df.columns:
#         for _, row in class_mean_std_df.sort_values('Class_ID').iterrows():
#             print(f"Class {int(row['Class_ID'])}: {row['Pred_Acc_Mean']:.4f} ± {row['Pred_Acc_Std']:.4f}")
#     pretty_report_path = write_pretty_summary_report(set_summary_df=set_summary_df, class_mean_std_df=class_mean_std_df)
#     final_report_path = write_final_report(set_summary_df=set_summary_df, class_mean_std_df=class_mean_std_df)
#     print('\n' + '=' * 100)
#     print('ALL RESULTS SAVED TO')
#     print('=' * 100)
#     print(OUTPUT_ROOT_DIR)
#     print('\nFiles:')
#     print(f'Set raw summary   : {set_raw_path}')
#     print(f'Set mean/std      : {set_mean_std_path}')
#     print(f'Class raw results : {class_raw_path}')
#     print(f'Class mean/std    : {class_mean_std_path}')
#     print(f'Final report      : {final_report_path}')
#     print(f'Pretty report     : {pretty_report_path}')
#     print('=' * 100)
#
# def main():
#     seed_everything(RANDOM_SEED)
#     evaluate_10sets()
# if __name__ == '__main__':
#     main()




import os
from pathlib import Path
import sys
import glob
import random
import itertools
from collections import defaultdict, Counter
import numpy as np
import pandas as pd
from scipy.stats import pearsonr
import torch
import torch.nn as nn
from transformers import AutoTokenizer
try:
    import Levenshtein
    HAS_LEVENSHTEIN = True
except ImportError:
    HAS_LEVENSHTEIN = False
    print('Levenshtein was not found. Install it with: pip install Levenshtein')
try:
    from Bio import Align
    HAS_BIOPYTHON = True
    aligner = Align.PairwiseAligner()
    aligner.mode = 'local'
    aligner.match_score = 2.0
    aligner.mismatch_score = -3.0
    aligner.open_gap_score = -5.0
    aligner.extend_gap_score = -2.0
except ImportError:
    HAS_BIOPYTHON = False
    print('Biopython was not found. Bio alignment will be skipped.')
RANDOM_SEED = 42
NUM_LABELS = 5
NUM_SETS_TO_EVALUATE = 10
SEQ_LEN = 80
SAMPLE_SIZE = 2000
LEV_PAIRS = 1000
BIO_PAIRS = 500
PROJECT_ROOT = Path(__file__).resolve().parent.parent
NATURAL_MASTER_PATH = str(PROJECT_ROOT / "dataset" / "wrc_class5_63468.txt")
GENERATED_ROOT_DIR = str(PROJECT_ROOT / "Diffusion_to_optimized" /"results_diff"/ "results_optimized_10sets")
OUTPUT_ROOT_DIR = os.path.join(GENERATED_ROOT_DIR, 'results_evaluation_dnabert2_regression')
PREDICTOR_MODEL_TAG = 'DNABERT2-Reg'
PREDICTOR_EXP_NAME = '5fold_regression_ensemble'
DNABERT2_BACKBONE_PATH = str(PROJECT_ROOT / "prediction" / "DNABERT_2")
DNABERT2_TOKENIZER_PATH = str(PROJECT_ROOT / "prediction" / "DNABERT_2")
if DNABERT2_BACKBONE_PATH not in sys.path:
    sys.path.append(DNABERT2_BACKBONE_PATH)
try:
    from bert_layers import BertModel
except ImportError:
    print(f'Error: cannot import bert_layers.py from {DNABERT2_BACKBONE_PATH}. Please check the local DNABERT2 directory.')
    raise
PREDICTOR_RESULTS_ROOT = str(PROJECT_ROOT / "prediction" / "checkpoint_dnabert2")
PREDICTOR_CHECKPOINTS = [os.path.join(PREDICTOR_RESULTS_ROOT, f'fold{i}', 'best_model.pth') for i in range(1, 6)]
STRENGTH_BIN_FILE = str(PROJECT_ROOT / "dataset" / "wrc_regression_63468_5fold" / "full_dataset_with_bins.csv")
PREDICTOR_DEVICE_STR = 'cuda:1'
PREDICT_BATCH_SIZE = 256
PREDICT_MAX_LEN = 100
PREDICT_KMER = 6
PREDICT_DO_KMER = False
MLP_HIDDEN_SIZE = 256
DROPOUT_RATE = 0.2
USE_LAST4 = True
POOL_TYPE = 'attn'
OUTPUT_REPORT_NAME = 'wrc_evaluation_report_dnabert2_regression.txt'
OUTPUT_CSV_NAME = 'wrc_evaluation_data_dnabert2_regression.csv'
OUTPUT_SUMMARY_CSV_NAME = 'wrc_evaluation_summary_dnabert2_regression.csv'

def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

class Logger:

    def __init__(self, filepath):
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        self.file = open(filepath, 'w', encoding='utf-8')

    def print(self, msg=''):
        print(msg)
        self.file.write(str(msg) + '\n')
        self.file.flush()

    def close(self):
        self.file.close()

def clean_acgt_seq(seq, max_len=80):
    seq = str(seq).upper()
    seq = seq.replace('U', 'T')
    seq = seq.replace(' ', '')
    seq = ''.join((ch for ch in seq if ch in 'ACGT'))
    if len(seq) > max_len:
        seq = seq[:max_len]
    return seq

def load_and_split_natural_data(path):
    data_dict = defaultdict(list)
    if not os.path.exists(path):
        print(f'Natural sequence file not found: {path}')
        return data_dict
    with open(path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 2:
                continue
            seq = clean_acgt_seq(parts[0], max_len=SEQ_LEN)
            try:
                label = int(parts[1])
            except Exception:
                continue
            if label in range(NUM_LABELS) and len(seq) > 0:
                data_dict[label].append(seq)
    return data_dict

def get_set_dir(root_dir, set_idx):
    return os.path.join(root_dir, f'set_{set_idx:02d}')

def find_generated_file_for_label(set_dir, label):
    exact_paths = [os.path.join(set_dir, f'zhou_unet_label_{label}.csv'), os.path.join(set_dir, f'promoter_Final_V15_97_{label}.csv')]
    for path in exact_paths:
        if os.path.isfile(path):
            return path
    patterns = [os.path.join(set_dir, f'*label_{label}*.csv'), os.path.join(set_dir, f'*label{label}*.csv'), os.path.join(set_dir, f'*_label{label}_*.csv'), os.path.join(set_dir, f'*_{label}.csv')]
    skip_keywords = ['evaluation', 'report', 'summary', 'generated_all', 'all_labels', 'optimized_detail', 'optimization_summary', 'optimization_log', 'prediction', 'metrics']
    for pattern in patterns:
        files = sorted(glob.glob(pattern))
        candidates = []
        for f in files:
            if not os.path.isfile(f):
                continue
            base = os.path.basename(f).lower()
            if any((keyword in base for keyword in skip_keywords)):
                continue
            candidates.append(f)
        if candidates:
            return candidates[0]
    return None

def load_generated_data(path):
    if path is None or not os.path.exists(path):
        return []
    try:
        df = pd.read_csv(path)
        if df.empty:
            return []
        if 'sequence' in df.columns:
            col = 'sequence'
        elif 'seq' in df.columns:
            col = 'seq'
        else:
            col = df.columns[0]
        seqs = []
        for s in df[col].tolist():
            seq = clean_acgt_seq(s, max_len=SEQ_LEN)
            if len(seq) > 0:
                seqs.append(seq)
        return seqs
    except Exception as e:
        print(f'Failed to read generated sequences: {path} | {e}')
        return []

def get_gc_content(seqs):
    if not seqs:
        return 0.0
    gcs = []
    for s in seqs:
        if len(s) == 0:
            continue
        gc = (s.count('G') + s.count('C')) / len(s)
        gcs.append(gc)
    if not gcs:
        return 0.0
    return float(np.mean(gcs))

def get_diversity(seqs):
    if not seqs:
        return 0.0
    return len(set(seqs)) / len(seqs)

def check_novelty(gen_seqs, train_set):
    if not gen_seqs:
        return 0.0
    hits = sum((1 for s in gen_seqs if s in train_set))
    return 1.0 - hits / len(gen_seqs)

def calc_metrics_sampling(seqs_a, seqs_b=None, mode='intra', lev_pairs=1000, bio_pairs=500):
    lev_dists = []
    bio_scores = []
    if not seqs_a:
        return (-1, -1)
    if mode == 'inter' and (not seqs_b):
        return (-1, -1)
    if HAS_LEVENSHTEIN:
        pairs = min(lev_pairs, 20000)
        for _ in range(pairs):
            if mode == 'intra':
                if len(seqs_a) < 2:
                    break
                s1, s2 = random.sample(seqs_a, 2)
            else:
                s1 = random.choice(seqs_a)
                s2 = random.choice(seqs_b)
            lev_dists.append(Levenshtein.distance(s1, s2))
    if HAS_BIOPYTHON:
        pairs = min(bio_pairs, 20000)
        for _ in range(pairs):
            if mode == 'intra':
                if len(seqs_a) < 2:
                    break
                s1, s2 = random.sample(seqs_a, 2)
            else:
                s1 = random.choice(seqs_a)
                s2 = random.choice(seqs_b)
            raw_score = aligner.score(s1, s2)
            max_possible = min(len(s1), len(s2)) * 2.0
            identity = raw_score / max_possible if max_possible > 0 else 0.0
            bio_scores.append(identity * 100.0)
    mean_lev = float(np.mean(lev_dists)) if lev_dists else -1
    mean_bio = float(np.mean(bio_scores)) if bio_scores else -1
    return (mean_lev, mean_bio)

def get_kmer_frequency(sequences, k=6):
    kmers = [''.join(p) for p in itertools.product('ACGT', repeat=k)]
    kmer_to_idx = {kmer: i for i, kmer in enumerate(kmers)}
    counts = np.zeros(len(kmers), dtype=np.float64)
    total = 0
    for seq in sequences:
        if len(seq) < k:
            continue
        for i in range(len(seq) - k + 1):
            sub = seq[i:i + k]
            idx = kmer_to_idx.get(sub, None)
            if idx is not None:
                counts[idx] += 1
                total += 1
    if total == 0:
        return np.zeros(len(kmers), dtype=np.float64)
    return counts / total

def safe_pearson(x, y):
    try:
        pcc, _ = pearsonr(x, y)
        return float(pcc)
    except Exception:
        return np.nan

class AttnPool1d(nn.Module):

    def __init__(self, hidden_size):
        super().__init__()
        self.attn = nn.Linear(hidden_size, 1)

    def forward(self, x):
        scores = self.attn(x).squeeze(-1)
        weights = torch.softmax(scores, dim=-1)
        pooled = torch.sum(x * weights.unsqueeze(-1), dim=1)
        return pooled

class DNABERT2_Hybrid_Regression(nn.Module):

    def __init__(self, mlp_hidden_size=256, dropout_rate=0.2, use_last4=True, pool_type='attn'):
        super().__init__()
        self.bert = BertModel.from_pretrained(DNABERT2_BACKBONE_PATH)
        self.use_last4 = use_last4
        self.pool_type = pool_type
        if hasattr(self.bert, 'config') and hasattr(self.bert.config, 'output_hidden_states'):
            self.bert.config.output_hidden_states = use_last4
        if hasattr(self.bert, 'config') and hasattr(self.bert.config, 'hidden_size'):
            self.bert_hidden = self.bert.config.hidden_size
        else:
            self.bert_hidden = 768
        if self.pool_type == 'attn':
            self.attn_pool = AttnPool1d(self.bert_hidden)
        else:
            self.attn_pool = None
        self.regressor = nn.Sequential(nn.LayerNorm(self.bert_hidden), nn.Linear(self.bert_hidden, mlp_hidden_size), nn.GELU(), nn.Dropout(dropout_rate), nn.Linear(mlp_hidden_size, 1))

    def _get_token_reps(self, outputs):
        hidden_states = None
        last_hidden_state = None
        if hasattr(outputs, 'last_hidden_state'):
            last_hidden_state = outputs.last_hidden_state
        elif isinstance(outputs, (list, tuple)) and len(outputs) > 0:
            last_hidden_state = outputs[0]
        if hasattr(outputs, 'hidden_states') and outputs.hidden_states is not None:
            hidden_states = outputs.hidden_states
        if self.use_last4 and hidden_states is not None and (len(hidden_states) >= 4):
            token_reps = torch.stack(hidden_states[-4:], dim=0).mean(dim=0)
        else:
            if last_hidden_state is None:
                raise RuntimeError('Unable to parse DNABERT2 output.')
            token_reps = last_hidden_state
        return token_reps

    def _pool(self, token_reps, attention_mask=None):
        if self.pool_type == 'cls':
            pooled = token_reps[:, 0, :]
        else:
            pooled = self.attn_pool(token_reps)
        return pooled

    def forward(self, input_ids, attention_mask=None):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=self.use_last4)
        token_reps = self._get_token_reps(outputs)
        pooled = self._pool(token_reps, attention_mask=attention_mask)
        output = self.regressor(pooled)
        return output

def seq_to_kmer_text(seq, k=6):
    seq = clean_acgt_seq(seq, max_len=SEQ_LEN)
    if len(seq) < k:
        return seq
    return ' '.join((seq[i:i + k] for i in range(len(seq) - k + 1)))

def safe_torch_load(path, device):
    try:
        return torch.load(path, map_location=device, weights_only=True)
    except TypeError:
        return torch.load(path, map_location=device)

def strip_module_prefix(state_dict):
    if not isinstance(state_dict, dict):
        return state_dict
    if any((k.startswith('module.') for k in state_dict.keys())):
        state_dict = {k[len('module.'):] if k.startswith('module.') else k: v for k, v in state_dict.items()}
    if any((k.startswith('_orig_mod.') for k in state_dict.keys())):
        state_dict = {k[len('_orig_mod.'):] if k.startswith('_orig_mod.') else k: v for k, v in state_dict.items()}
    return state_dict

def load_strength_thresholds(csv_path, logger=None):

    def log(msg):
        if logger is None:
            print(msg)
        else:
            logger.print(msg)
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f'Strength bin file not found: {csv_path}')
    df = pd.read_csv(csv_path)
    if 'strength' not in df.columns:
        raise ValueError('The file has no strength column.')
    if 'bin' not in df.columns:
        raise ValueError('The file has no bin column.')
    df['strength'] = pd.to_numeric(df['strength'], errors='coerce')
    df['bin'] = pd.to_numeric(df['bin'], errors='coerce')
    df = df.dropna(subset=['strength', 'bin'])
    df['bin'] = df['bin'].astype(int)
    found_bins = sorted(df['bin'].unique().tolist())
    expected_bins = list(range(NUM_LABELS))
    if found_bins != expected_bins:
        raise RuntimeError(f'Unexpected bins: expected={expected_bins}, found={found_bins}')
    bin_ranges = {}
    log('\n' + '=' * 100)
    log('Regression strength intervals')
    log('=' * 100)
    for label in range(NUM_LABELS):
        values = df.loc[df['bin'] == label, 'strength'].values
        if len(values) == 0:
            raise RuntimeError(f'Bin {label} is empty.')
        min_v = float(np.min(values))
        max_v = float(np.max(values))
        bin_ranges[label] = (min_v, max_v)
        log(f'Class {label}: N={len(values)} | Strength=[{min_v:.6f}, {max_v:.6f}]')
    thresholds = []
    for label in range(NUM_LABELS - 1):
        left_max = bin_ranges[label][1]
        right_min = bin_ranges[label + 1][0]
        threshold = (left_max + right_min) / 2.0
        thresholds.append(threshold)
    thresholds = np.asarray(thresholds, dtype=np.float64)
    log('\nStrength thresholds:')
    for idx, value in enumerate(thresholds, start=1):
        log(f't{idx} = {value:.6f}')
    log('\nMapping:')
    log(f'Class 0: x < {thresholds[0]:.6f}')
    for label in range(1, NUM_LABELS - 1):
        log(f'Class {label}: {thresholds[label - 1]:.6f} <= x < {thresholds[label]:.6f}')
    log(f'Class 4: x >= {thresholds[-1]:.6f}')
    log('=' * 100)
    return (thresholds, bin_ranges)

def strength_to_label(strengths, thresholds):
    strengths = np.asarray(strengths, dtype=np.float64)
    labels = np.digitize(strengths, bins=thresholds, right=False)
    return labels.astype(np.int64)

def load_ensemble_models(device, logger=None):

    def log(msg):
        if logger is None:
            print(msg)
        else:
            logger.print(msg)
    models = []
    log('\nLoading DNABERT2 5-fold regression ensemble...')
    for fold_idx, path in enumerate(PREDICTOR_CHECKPOINTS, start=1):
        if not os.path.exists(path):
            log(f'Fold {fold_idx}: MISSING | {path}')
            continue
        model = DNABERT2_Hybrid_Regression(mlp_hidden_size=MLP_HIDDEN_SIZE, dropout_rate=DROPOUT_RATE, use_last4=USE_LAST4, pool_type=POOL_TYPE).to(device)
        try:
            ckpt = safe_torch_load(path, device)
            if isinstance(ckpt, dict) and 'model_state_dict' in ckpt:
                state_dict = ckpt['model_state_dict']
            elif isinstance(ckpt, dict) and 'state_dict' in ckpt:
                state_dict = ckpt['state_dict']
            else:
                state_dict = ckpt
            state_dict = strip_module_prefix(state_dict)
            model.load_state_dict(state_dict, strict=True)
            model.eval()
            models.append(model)
            log(f'Fold {fold_idx}: OK | {path}')
        except Exception as e:
            log(f'Fold {fold_idx}: LOAD FAILED')
            log(f'Error: {e}')
    log(f'Successfully loaded {len(models)}/5 models.')
    return models

@torch.no_grad()
def predict_ensemble_strength(models, tokenizer, sequences, device, batch_size=256):
    if not models:
        return np.asarray([], dtype=np.float32)
    all_strengths = []
    for start in range(0, len(sequences), batch_size):
        batch = sequences[start:start + batch_size]
        texts = []
        for seq in batch:
            seq = clean_acgt_seq(seq, max_len=SEQ_LEN)
            if len(seq) == 0:
                continue
            if PREDICT_DO_KMER:
                text = seq_to_kmer_text(seq, k=PREDICT_KMER)
            else:
                text = seq
            texts.append(text)
        if not texts:
            continue
        encoding = tokenizer(texts, add_special_tokens=True, max_length=PREDICT_MAX_LEN, padding='max_length', truncation=True, return_attention_mask=True, return_tensors='pt')
        input_ids = encoding['input_ids'].to(device, non_blocking=True)
        attention_mask = encoding['attention_mask'].to(device, non_blocking=True)
        fold_preds = []
        for model in models:
            pred = model(input_ids, attention_mask)
            if pred.ndim != 2 or pred.shape[1] != 1:
                raise RuntimeError(f'Regression output should be (B,1), got {tuple(pred.shape)}')
            fold_preds.append(pred.squeeze(1))
        mean_strength = torch.stack(fold_preds, dim=0).mean(dim=0)
        all_strengths.extend(mean_strength.detach().cpu().numpy().tolist())
    return np.asarray(all_strengths, dtype=np.float32)

def evaluate_label_control(models, tokenizer, gen_seqs, target_label, device, thresholds):
    strengths = predict_ensemble_strength(models, tokenizer, gen_seqs, device, batch_size=PREDICT_BATCH_SIZE)
    if len(strengths) == 0:
        return None
    pred_labels = strength_to_label(strengths, thresholds)
    dist = dict(sorted(Counter(pred_labels).items()))
    acc = float((pred_labels == target_label).mean())
    mean_pred = float(pred_labels.mean())
    mae = float(np.abs(pred_labels - target_label).mean())
    return {'n_pred': int(len(pred_labels)), 'dist': dist, 'acc': acc, 'mean_pred': mean_pred, 'mae': mae}

def evaluate_one_label(label, gen_path, nat_seqs, ensemble_models, tokenizer, device, thresholds, logger):
    gen_seqs = load_generated_data(gen_path)
    if not nat_seqs:
        logger.print(f'Class_{label}: natural sequences are empty.')
        return None
    if not gen_seqs:
        logger.print(f'Class_{label}: generated sequences are empty.')
        return None
    train_set_ref = set(nat_seqs)
    curr_nat = random.sample(nat_seqs, SAMPLE_SIZE) if len(nat_seqs) > SAMPLE_SIZE else nat_seqs
    curr_gen = gen_seqs[:SAMPLE_SIZE]
    nat_gc = get_gc_content(nat_seqs)
    gen_gc = get_gc_content(curr_gen)
    gc_diff = abs(nat_gc - gen_gc)
    diversity = get_diversity(curr_gen)
    novelty = check_novelty(curr_gen, train_set_ref)
    lev_nat, bio_nat = calc_metrics_sampling(curr_nat, mode='intra', lev_pairs=LEV_PAIRS, bio_pairs=BIO_PAIRS)
    lev_gen, bio_gen = calc_metrics_sampling(curr_gen, mode='intra', lev_pairs=LEV_PAIRS, bio_pairs=BIO_PAIRS)
    lev_inter, bio_inter = calc_metrics_sampling(curr_gen, curr_nat, mode='inter', lev_pairs=LEV_PAIRS, bio_pairs=BIO_PAIRS)
    logger.print(f'\nProcessing Class_{label} ...')
    logger.print(f'   [File] {gen_path}')
    logger.print(f'   [General] GC={gen_gc:.4f} (Nat={nat_gc:.4f}) | GC_diff={gc_diff:.4f} | Div={diversity:.4f} | Nov={novelty:.4f}')
    logger.print(f'   [EditDist] Intra-Nat={lev_nat:.1f} | Intra-Gen={lev_gen:.1f} | Inter={lev_inter:.1f}')
    logger.print(f'   [BioAlign] Intra-Nat={bio_nat:.1f}% | Intra-Gen={bio_gen:.1f}% | Inter={bio_inter:.1f}%')
    kmer_pcc_dict = {}
    for k in [4, 5, 6]:
        freq_nat = get_kmer_frequency(curr_nat, k)
        freq_gen = get_kmer_frequency(curr_gen, k)
        pcc = safe_pearson(freq_nat, freq_gen)
        kmer_pcc_dict[k] = pcc
        logger.print(f'   [K-mer={k}] PCC={pcc:.4f}')
    pred_metrics = None
    if ensemble_models:
        pred_metrics = evaluate_label_control(ensemble_models, tokenizer, curr_gen, label, device, thresholds)
        if pred_metrics is not None:
            logger.print(f"   [Predict] Target={label} | N={pred_metrics['n_pred']}")
            logger.print(f"   [Predict] Dist={pred_metrics['dist']}")
            logger.print(f"   [Predict] Acc={pred_metrics['acc']:.4f}")
            logger.print(f"   [Predict] MeanPred={pred_metrics['mean_pred']:.4f} | MAE={pred_metrics['mae']:.4f}")
    row = {'Class_ID': label, 'File': gen_path, 'N_Generated_Loaded': len(gen_seqs), 'N_Evaluated': len(curr_gen), 'GC_Nat': nat_gc, 'GC_Gen': gen_gc, 'GC_Diff': gc_diff, 'Diversity': diversity, 'Novelty': novelty, 'Lev_Intra_Nat': lev_nat, 'Lev_Intra_Gen': lev_gen, 'Lev_Inter': lev_inter, 'Bio_Intra_Nat': bio_nat, 'Bio_Intra_Gen': bio_gen, 'Bio_Inter': bio_inter, 'K4_PCC': kmer_pcc_dict.get(4, np.nan), 'K5_PCC': kmer_pcc_dict.get(5, np.nan), 'K6_PCC': kmer_pcc_dict.get(6, np.nan), 'Predictor': f'{PREDICTOR_MODEL_TAG}/{PREDICTOR_EXP_NAME}'}
    if pred_metrics is not None:
        row.update({'Pred_N': pred_metrics['n_pred'], 'Pred_Acc': pred_metrics['acc'], 'Pred_MeanPred': pred_metrics['mean_pred'], 'Pred_MAE': pred_metrics['mae'], 'Pred_Dist': str(pred_metrics['dist'])})
    return row

def evaluate_one_set(set_idx, set_dir, nat_data_dict, ensemble_models, tokenizer, device, thresholds):
    set_output_dir = os.path.join(OUTPUT_ROOT_DIR, f'set_{set_idx:02d}')
    os.makedirs(set_output_dir, exist_ok=True)
    output_report_path = os.path.join(set_output_dir, OUTPUT_REPORT_NAME)
    output_csv_path = os.path.join(set_output_dir, OUTPUT_CSV_NAME)
    output_summary_csv_path = os.path.join(set_output_dir, OUTPUT_SUMMARY_CSV_NAME)
    logger = Logger(output_report_path)
    logger.print('=' * 100)
    logger.print(f'Evaluation Set {set_idx:02d}')
    logger.print('=' * 100)
    logger.print(f'Source set dir: {set_dir}')
    logger.print(f'Output dir: {set_output_dir}')
    logger.print(f'Predictor: {PREDICTOR_MODEL_TAG}/{PREDICTOR_EXP_NAME}')
    logger.print(f'SAMPLE_SIZE={SAMPLE_SIZE}')
    rows = []
    for label in range(NUM_LABELS):
        gen_path = find_generated_file_for_label(set_dir, label)
        if gen_path is None:
            logger.print(f'\nClass_{label}: generated CSV not found.')
            continue
        nat_seqs = nat_data_dict.get(label, [])
        row = evaluate_one_label(label=label, gen_path=gen_path, nat_seqs=nat_seqs, ensemble_models=ensemble_models, tokenizer=tokenizer, device=device, thresholds=thresholds, logger=logger)
        if row is not None:
            row['Set_ID'] = set_idx
            rows.append(row)
    if not rows:
        logger.print('\nNo valid results for this set.')
        logger.close()
        return (None, None)
    class_df = pd.DataFrame(rows)
    class_df.to_csv(output_csv_path, index=False)
    metrics = ['GC_Diff', 'Diversity', 'Novelty', 'Lev_Inter', 'Bio_Inter', 'K4_PCC', 'K5_PCC', 'K6_PCC', 'Pred_Acc', 'Pred_MAE', 'Pred_MeanPred']
    summary = {'Set_ID': set_idx}
    for metric in metrics:
        if metric not in class_df.columns:
            continue
        values = pd.to_numeric(class_df[metric], errors='coerce').dropna()
        if len(values) > 0:
            summary[metric] = float(values.mean())
    if 'Pred_MeanPred' in class_df.columns:
        temp = class_df[['Class_ID', 'Pred_MeanPred']].dropna()
        if len(temp) >= 2:
            corr = np.corrcoef(temp['Class_ID'].values.astype(float), temp['Pred_MeanPred'].values.astype(float))[0, 1]
            summary['Corr_Target_MeanPred'] = float(corr)
    logger.print('\n' + '=' * 100)
    logger.print(f'Set {set_idx:02d} Summary')
    logger.print('=' * 100)
    if 'Pred_Acc' in summary:
        logger.print(f"Avg Pred Acc : {summary['Pred_Acc']:.4f}")
    if 'Pred_MAE' in summary:
        logger.print(f"Avg Pred MAE : {summary['Pred_MAE']:.4f}")
    logger.print(f"Avg GC Diff  : {summary.get('GC_Diff', np.nan):.4f}")
    logger.print(f"Avg 4-mer PCC: {summary.get('K4_PCC', np.nan):.4f}")
    logger.print(f"Avg 5-mer PCC: {summary.get('K5_PCC', np.nan):.4f}")
    logger.print(f"Avg 6-mer PCC: {summary.get('K6_PCC', np.nan):.4f}")
    if 'Corr_Target_MeanPred' in summary:
        logger.print(f"Corr target-mean_pred: {summary['Corr_Target_MeanPred']:.4f}")
    pd.DataFrame([{'Metric': key, 'Value': value} for key, value in summary.items()]).to_csv(output_summary_csv_path, index=False)
    logger.print('\nSaved:')
    logger.print(output_report_path)
    logger.print(output_csv_path)
    logger.print(output_summary_csv_path)
    logger.close()
    return (class_df, summary)

def make_mean_std_table(df, group_col=None):
    exclude_columns = {'Set_ID', 'Class_ID', 'File', 'Predictor', 'Pred_Dist'}
    numeric_cols = []
    for col in df.columns:
        if col in exclude_columns:
            continue
        converted = pd.to_numeric(df[col], errors='coerce')
        if converted.notna().any():
            numeric_cols.append(col)
    rows = []
    if group_col is None:
        groups = [('Overall', df)]
    else:
        groups = list(df.groupby(group_col))
    for group_name, group_df in groups:
        row = {}
        if group_col is not None:
            row[group_col] = group_name
        for metric in numeric_cols:
            values = pd.to_numeric(group_df[metric], errors='coerce').dropna().values
            if len(values) == 0:
                continue
            row[metric + '_Mean'] = float(np.mean(values))
            row[metric + '_Std'] = float(np.std(values, ddof=1)) if len(values) > 1 else 0.0
            row[metric + '_Var'] = float(np.var(values, ddof=1)) if len(values) > 1 else 0.0
            row[metric + '_N'] = int(len(values))
        rows.append(row)
    return pd.DataFrame(rows)

def write_pretty_summary_report(set_summary_df, class_mean_std_df):
    report_path = os.path.join(OUTPUT_ROOT_DIR, 'wrc_10sets_pretty_summary.txt')

    def get_stats(metric):
        if metric not in set_summary_df.columns:
            return (np.nan, np.nan, np.nan, 0)
        values = pd.to_numeric(set_summary_df[metric], errors='coerce').dropna().values.astype(float)
        if len(values) == 0:
            return (np.nan, np.nan, np.nan, 0)
        mean_v = float(np.mean(values))
        std_v = float(np.std(values, ddof=1)) if len(values) > 1 else 0.0
        var_v = float(np.var(values, ddof=1)) if len(values) > 1 else 0.0
        return (mean_v, std_v, var_v, len(values))
    width = 120
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write('=' * width + '\n')
        f.write('10-SET GENERATED PROMOTER EVALUATION SUMMARY\n')
        f.write('Predictor: DNABERT2 5-fold Regression Ensemble\n')
        f.write('Label decision: mean predicted strength -> predefined strength interval\n')
        f.write('=' * width + '\n\n')
        f.write('=' * width + '\n')
        f.write('[Metric 1: Basic Sequence Quality]\n')
        f.write('=' * width + '\n\n')
        metric_info = [('GC_Diff', 'GC Diff', '<0.05 is good', lambda x: x < 0.05), ('Diversity', 'Diversity', '>0.99 is good', lambda x: x > 0.99), ('Novelty', 'Novelty', 'higher is better', lambda x: x > 0.95)]
        for metric, name, note, condition in metric_info:
            mean_v, std_v, var_v, n = get_stats(metric)
            if np.isnan(mean_v):
                continue
            icon = '✅' if condition(mean_v) else '⚠️'
            f.write(f"   • {name:<22}: {mean_v:.4f} ± {std_v:.4f} (Var={var_v:.6f}, N={n}){' ' * 8}{icon}  {note}\n")
        f.write('\n\n')
        f.write('=' * width + '\n')
        f.write('[Metric 2: Global Similarity Gen vs Nat]\n')
        f.write('=' * width + '\n\n')
        mean_v, std_v, var_v, n = get_stats('Lev_Inter')
        if not np.isnan(mean_v):
            f.write(f"   • {'Edit Distance':<22}: {mean_v:.2f} ± {std_v:.2f} (Var={var_v:.6f}, N={n}){' ' * 8}✅  lower means closer to natural\n")
        mean_v, std_v, var_v, n = get_stats('Bio_Inter')
        if not np.isnan(mean_v):
            f.write(f"   • {'Bio Alignment':<22}: {mean_v:.4f}% ± {std_v:.4f}% (Var={var_v:.6f}, N={n}){' ' * 8}✅  higher means closer to natural\n")
        f.write('\n\n')
        f.write('=' * width + '\n')
        f.write('[Metric 3: Local Motif Similarity Avg PCC]\n')
        f.write('=' * width + '\n\n')
        for metric, name in [('K4_PCC', '4-mer PCC'), ('K5_PCC', '5-mer PCC'), ('K6_PCC', '6-mer PCC')]:
            mean_v, std_v, var_v, n = get_stats(metric)
            if np.isnan(mean_v):
                continue
            icon = '✅' if mean_v > 0.9 else '⚠️'
            f.write(f"   • {name:<22}: {mean_v:.4f} ± {std_v:.4f} (Var={var_v:.6f}, N={n}){' ' * 8}{icon}  target >0.90\n")
        f.write('\n\n')
        f.write('=' * width + '\n')
        f.write('[Metric 4: Label Controllability by DNABERT2 5-fold Regression Ensemble]\n')
        f.write('=' * width + '\n\n')
        mean_v, std_v, var_v, n = get_stats('Pred_Acc')
        if not np.isnan(mean_v):
            f.write(f"   • {'Avg Acc':<22}: {mean_v:.4f} ± {std_v:.4f} (Var={var_v:.6f}, N={n}){' ' * 8}✅  higher is better\n")
        mean_v, std_v, var_v, n = get_stats('Pred_MAE')
        if not np.isnan(mean_v):
            f.write(f"   • {'Avg MAE':<22}: {mean_v:.4f} ± {std_v:.4f} (Var={var_v:.6f}, N={n}){' ' * 8}✅  lower is better\n")
        mean_v, std_v, var_v, n = get_stats('Corr_Target_MeanPred')
        if not np.isnan(mean_v):
            f.write(f"   • {'Corr target-mean_pred':<22}: {mean_v:.4f} ± {std_v:.4f} (Var={var_v:.6f}, N={n}){' ' * 8}✅  higher is better\n")
        f.write('\n\n')
        f.write('=' * width + '\n')
        f.write('[Metric 5: Class-wise Label Controllability Across 10 Sets]\n')
        f.write('=' * width + '\n\n')
        if class_mean_std_df is not None and len(class_mean_std_df) > 0 and ('Pred_Acc_Mean' in class_mean_std_df.columns):
            temp = class_mean_std_df.sort_values('Class_ID')
            for _, row in temp.iterrows():
                label = int(row['Class_ID'])
                acc_mean = float(row['Pred_Acc_Mean'])
                acc_std = float(row.get('Pred_Acc_Std', 0.0))
                mae_mean = float(row.get('Pred_MAE_Mean', np.nan))
                mae_std = float(row.get('Pred_MAE_Std', np.nan))
                f.write(f'   • Class {label}: Acc={acc_mean:.4f} ± {acc_std:.4f}')
                if not np.isnan(mae_mean):
                    f.write(f' | MAE={mae_mean:.4f} ± {mae_std:.4f}')
                f.write('\n')
        f.write('\n')
        f.write('=' * width + '\n')
        f.write('Accuracy values are reported on a 0-1 scale.\n')
        f.write('=' * width + '\n')
    return report_path

def write_final_report(set_summary_df, class_mean_std_df):
    report_path = os.path.join(OUTPUT_ROOT_DIR, 'wrc_10sets_final_report_dnabert2_regression.txt')
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write('=' * 110 + '\n')
        f.write('10-SET GENERATED PROMOTER EVALUATION\n')
        f.write('DNABERT2 5-fold regression ensemble\n')
        f.write('=' * 110 + '\n\n')
        f.write(f'Generated root : {GENERATED_ROOT_DIR}\n')
        f.write(f'Output root    : {OUTPUT_ROOT_DIR}\n')
        f.write(f'Predictor root : {PREDICTOR_RESULTS_ROOT}\n')
        f.write(f'Random seed    : {RANDOM_SEED}\n')
        f.write(f'Valid sets     : {len(set_summary_df)}\n\n')
        columns = ['Set_ID', 'Pred_Acc', 'Pred_MAE', 'GC_Diff', 'K4_PCC', 'K5_PCC', 'K6_PCC', 'Corr_Target_MeanPred']
        existing = [c for c in columns if c in set_summary_df.columns]
        f.write(set_summary_df[existing].to_string(index=False, float_format=lambda x: f'{x:.4f}'))
        f.write('\n\n')
        if class_mean_std_df is not None and len(class_mean_std_df) > 0:
            f.write('=' * 110 + '\n')
            f.write('CLASS-LEVEL MEAN ± STD\n')
            f.write('=' * 110 + '\n')
            show_cols = ['Class_ID', 'Pred_Acc_Mean', 'Pred_Acc_Std', 'Pred_MAE_Mean', 'Pred_MAE_Std', 'GC_Diff_Mean', 'GC_Diff_Std', 'K4_PCC_Mean', 'K4_PCC_Std', 'K5_PCC_Mean', 'K5_PCC_Std', 'K6_PCC_Mean', 'K6_PCC_Std']
            existing = [c for c in show_cols if c in class_mean_std_df.columns]
            f.write(class_mean_std_df[existing].to_string(index=False, float_format=lambda x: f'{x:.4f}'))
            f.write('\n')
    return report_path

def evaluate_10sets():
    os.makedirs(OUTPUT_ROOT_DIR, exist_ok=True)
    print('=' * 100)
    print('10-SET PROMOTER EVALUATION')
    print('=' * 100)
    print(f'Generated root : {GENERATED_ROOT_DIR}')
    print(f'Output root    : {OUTPUT_ROOT_DIR}')
    print(f'Predictor root : {PREDICTOR_RESULTS_ROOT}')
    print(f'Strength bins  : {STRENGTH_BIN_FILE}')
    print(f'Seed           : {RANDOM_SEED}')
    print('=' * 100)
    if not os.path.isdir(GENERATED_ROOT_DIR):
        print('ERROR: generated root does not exist:')
        print(GENERATED_ROOT_DIR)
        return
    if not os.path.isdir(PREDICTOR_RESULTS_ROOT):
        print('ERROR: predictor root does not exist:')
        print(PREDICTOR_RESULTS_ROOT)
        return
    if not os.path.exists(STRENGTH_BIN_FILE):
        print('ERROR: strength bin file does not exist:')
        print(STRENGTH_BIN_FILE)
        return
    nat_data_dict = load_and_split_natural_data(NATURAL_MASTER_PATH)
    if not nat_data_dict:
        print('ERROR: natural dataset failed to load.')
        return
    print('\nNatural reference:')
    for label in range(NUM_LABELS):
        print(f'Class {label}: {len(nat_data_dict[label])}')
    if 'cuda' in PREDICTOR_DEVICE_STR and torch.cuda.is_available():
        device = torch.device(PREDICTOR_DEVICE_STR)
    else:
        device = torch.device('cpu')
    print(f'\nPredictor device: {device}')
    thresholds, bin_ranges = load_strength_thresholds(STRENGTH_BIN_FILE)
    print(f'\nLoading tokenizer from: {DNABERT2_TOKENIZER_PATH}')
    tokenizer = AutoTokenizer.from_pretrained(DNABERT2_TOKENIZER_PATH, trust_remote_code=True)
    ensemble_models = load_ensemble_models(device)
    if not ensemble_models:
        print('ERROR: no regression model loaded successfully.')
        return
    if len(ensemble_models) < 5:
        print(f'\nWARNING: only {len(ensemble_models)}/5 models loaded.')
    all_class_dfs = []
    all_set_summaries = []
    for set_idx in range(1, NUM_SETS_TO_EVALUATE + 1):
        set_dir = get_set_dir(GENERATED_ROOT_DIR, set_idx)
        print('\n' + '=' * 100)
        print(f'SET {set_idx:02d}/{NUM_SETS_TO_EVALUATE}')
        print(f'Source directory: {set_dir}')
        print('=' * 100)
        if not os.path.isdir(set_dir):
            print(f'WARNING: {set_dir} does not exist; skipping.')
            continue
        for label in range(NUM_LABELS):
            file_path = find_generated_file_for_label(set_dir, label)
            if file_path is None:
                print(f'Class {label}: MISSING')
            else:
                print(f'Class {label}: {os.path.basename(file_path)}')
        class_df, summary = evaluate_one_set(set_idx=set_idx, set_dir=set_dir, nat_data_dict=nat_data_dict, ensemble_models=ensemble_models, tokenizer=tokenizer, device=device, thresholds=thresholds)
        if class_df is None:
            continue
        all_class_dfs.append(class_df)
        all_set_summaries.append(summary)
    if not all_set_summaries:
        print('\nERROR: no sets produced valid results.')
        return
    class_raw_df = pd.concat(all_class_dfs, axis=0, ignore_index=True)
    set_summary_df = pd.DataFrame(all_set_summaries)
    set_raw_path = os.path.join(OUTPUT_ROOT_DIR, 'wrc_10sets_set_level_summary_raw.csv')
    set_summary_df.to_csv(set_raw_path, index=False)
    set_mean_std_df = make_mean_std_table(set_summary_df, group_col=None)
    set_mean_std_path = os.path.join(OUTPUT_ROOT_DIR, 'wrc_10sets_set_level_mean_std.csv')
    set_mean_std_df.to_csv(set_mean_std_path, index=False)
    class_raw_path = os.path.join(OUTPUT_ROOT_DIR, 'wrc_10sets_class_level_raw.csv')
    class_raw_df.to_csv(class_raw_path, index=False)
    class_mean_std_df = make_mean_std_table(class_raw_df, group_col='Class_ID')
    class_mean_std_path = os.path.join(OUTPUT_ROOT_DIR, 'wrc_10sets_class_level_mean_std.csv')
    class_mean_std_df.to_csv(class_mean_std_path, index=False)
    print('\n' + '=' * 100)
    print('10-SET FINAL RESULTS')
    print('=' * 100)
    print(f'Valid sets: {len(set_summary_df)}')
    important_metrics = ['Pred_Acc', 'Pred_MAE', 'GC_Diff', 'Diversity', 'Novelty', 'Lev_Inter', 'Bio_Inter', 'K4_PCC', 'K5_PCC', 'K6_PCC', 'Corr_Target_MeanPred']
    if len(set_mean_std_df) > 0:
        summary_row = set_mean_std_df.iloc[0]
        print('\nOverall mean ± std:')
        for metric in important_metrics:
            mean_col = metric + '_Mean'
            std_col = metric + '_Std'
            if mean_col not in summary_row.index:
                continue
            mean_v = summary_row[mean_col]
            std_v = summary_row[std_col] if std_col in summary_row.index else np.nan
            print(f'{metric:<28}: {mean_v:.4f} ± {std_v:.4f}')
    print('\nClass-level Pred_Acc (10-set mean ± std):')
    if 'Pred_Acc_Mean' in class_mean_std_df.columns:
        for _, row in class_mean_std_df.sort_values('Class_ID').iterrows():
            print(f"Class {int(row['Class_ID'])}: {row['Pred_Acc_Mean']:.4f} ± {row['Pred_Acc_Std']:.4f}")
    pretty_report_path = write_pretty_summary_report(set_summary_df=set_summary_df, class_mean_std_df=class_mean_std_df)
    final_report_path = write_final_report(set_summary_df=set_summary_df, class_mean_std_df=class_mean_std_df)
    print('\n' + '=' * 100)
    print('ALL RESULTS SAVED TO')
    print('=' * 100)
    print(OUTPUT_ROOT_DIR)
    print('\nFiles:')
    print(f'Set raw summary   : {set_raw_path}')
    print(f'Set mean/std      : {set_mean_std_path}')
    print(f'Class raw results : {class_raw_path}')
    print(f'Class mean/std    : {class_mean_std_path}')
    print(f'Final report      : {final_report_path}')
    print(f'Pretty report     : {pretty_report_path}')
    print('=' * 100)

def main():
    seed_everything(RANDOM_SEED)
    evaluate_10sets()
if __name__ == '__main__':
    main()
