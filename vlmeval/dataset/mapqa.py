import os
import os.path as osp
from collections import defaultdict

import numpy as np
import pandas as pd
from tqdm import tqdm

from vlmeval.smp import LMUDataRoot, d2df, dump, get_intermediate_file_path, load
from .image_base import ImageBaseDataset


def normalize_answer(answer):
    """Normalize answer for comparison."""
    if isinstance(answer, list):
        return sorted([str(a).strip().lower() for a in answer])
    return str(answer).strip().lower()


def exact_match_score(pred, gt):
    """Calculate exact match score.
    
    For list answers: F1 score based on set overlap.
    For single answers: 1 if exact match, 0 otherwise.
    """
    pred_norm = normalize_answer(pred)
    gt_norm = normalize_answer(gt)
    
    if isinstance(gt_norm, list):
        # For list answers, compute F1
        pred_set = set(pred_norm) if isinstance(pred_norm, list) else {pred_norm}
        gt_set = set(gt_norm)
        
        if len(gt_set) == 0:
            return 1.0 if len(pred_set) == 0 else 0.0
        
        tp = len(pred_set & gt_set)
        fp = len(pred_set - gt_set)
        fn = len(gt_set - pred_set)
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        
        if precision + recall == 0:
            return 0.0
        f1 = 2 * precision * recall / (precision + recall)
        return f1
    else:
        # For single answers, exact match
        return 1.0 if pred_norm == gt_norm else 0.0


class MapQA(ImageBaseDataset):
    TYPE = 'VQA'

    # We use local TSV files (no download URL for custom sampled data)
    DATASET_URL = {
        'MapQA': '',
    }
    DATASET_MD5 = {}

    @classmethod
    def supported_datasets(cls):
        return ['MapQA']

    def load_data(self, dataset):
        data_path = osp.join(LMUDataRoot(), f'{dataset}.tsv')
        assert osp.exists(data_path), f'Data file not found: {data_path}'
        return load(data_path)

    def __init__(self, dataset='MapQA', skip_noimg=True):
        super().__init__(dataset, skip_noimg)
        # When TSV only has image_path (no base64 image column), meta_only stays True.
        # Set to False so dump_image resolves full paths.
        self.meta_only = False
        # Override img_root to use our prepared images directory
        alt_img_root = osp.join(os.environ.get('USERPROFILE', ''), 'AppData', 'Roaming',
                                'TRAE SOLO CN', 'ModularData', 'ai-agent',
                                'work-mode-projects', '6a58f4312119acff537a0a82',
                                'MapQA_eval', 'images')
        if osp.isdir(alt_img_root):
            self.img_root = alt_img_root

    def build_prompt(self, line):
        if isinstance(line, int):
            line = self.data.iloc[line]

        if self.meta_only:
            tgt_path = [line['image_path']] if isinstance(line['image_path'], str) else line['image_path']
        else:
            tgt_path = self.dump_image(line)

        question = line['question']

        msgs = []
        if isinstance(tgt_path, list):
            msgs.extend([dict(type='image', value=p) for p in tgt_path])
        else:
            msgs = [dict(type='image', value=tgt_path)]
        msgs.append(dict(type='text', value=question))
        msgs.append(dict(type='text', value='\nAnswer the question concisely. If the answer is a list of states, list them separated by commas.'))
        return msgs

    def evaluate(self, eval_file, **judge_kwargs):
        data = load(eval_file)
        assert 'answer' in data and 'prediction' in data

        scores = []
        question_types = []
        subsets = []

        for i in range(len(data)):
            pred = str(data.iloc[i]['prediction'])
            gt = data.iloc[i]['answer']
            
            # Parse gt - it might be a string representation of a list
            if isinstance(gt, str):
                gt = gt.strip()
                if gt.startswith('[') and gt.endswith(']'):
                    try:
                        import ast
                        gt = ast.literal_eval(gt)
                    except:
                        pass
            
            score = exact_match_score(pred, gt)
            scores.append(score)
            
            if 'question_type' in data.columns:
                question_types.append(data.iloc[i]['question_type'])
            else:
                question_types.append('unknown')
            
            if 'subset' in data.columns:
                subsets.append(data.iloc[i]['subset'])
            else:
                subsets.append('unknown')

        data['score'] = scores
        detailed_result_file = get_intermediate_file_path(eval_file, '_detailed_results')
        dump(data, detailed_result_file)

        # Overall score
        ret = dict()
        ret['Overall'] = np.mean(scores) * 100

        # By question type
        type_scores = defaultdict(list)
        for qt, sc in zip(question_types, scores):
            type_scores[qt].append(sc)
        for qt, sc_list in type_scores.items():
            ret[f'Type|{qt}'] = np.mean(sc_list) * 100

        # By subset
        subset_scores = defaultdict(list)
        for sub, sc in zip(subsets, scores):
            subset_scores[sub].append(sc)
        for sub, sc_list in subset_scores.items():
            ret[f'Subset|{sub}'] = np.mean(sc_list) * 100

        ret = d2df(ret)
        ret = ret.round(2)

        result_file = get_intermediate_file_path(eval_file, '_acc')
        dump(ret, result_file)
        return ret
