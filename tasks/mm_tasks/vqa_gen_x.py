# Copyright 2022 The OFA-Sys Team. 
# All rights reserved.
# This source code is licensed under the Apache 2.0 license 
# found in the LICENSE file in the root directory.

from dataclasses import dataclass, field
import json
import logging
import os
import math
import pickle
from functools import partial
from typing import Optional
from argparse import Namespace
from data.file_dataset import FileDataset

import torch
from fairseq import metrics
from fairseq.tasks import register_task

from data.mm_data.vqa_gen_x_dataset import VqaGenXDataset
from models import search
from data import data_utils
from tasks.ofa_task import OFAConfig, OFATask
from utils.trie import Trie
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)


def get_symbols_to_strip_from_output(generator):
    if hasattr(generator, "symbols_to_strip_from_output"):
        return generator.symbols_to_strip_from_output
    else:
        return {generator.bos, generator.eos}


def decode_fn(x, tgt_dict, bpe, generator, tokenizer=None):
    x = tgt_dict.string(x.int().cpu(), extra_symbols_to_ignore=get_symbols_to_strip_from_output(generator))
    if bpe is not None:
        x = bpe.decode(x)
    if tokenizer is not None:
        x = tokenizer.decode(x)
    return x


@dataclass
class VqaGenXConfig(OFAConfig):
    max_object_length: int = field(
        default=30, metadata={"help": "the maximum object sequence length"}
    )    
    ans2label_dict: Optional[str] = field(
        default='{"no": 0, "yes":1}',
        metadata={"help": 'answer to label dict'},
    )
    ans2label_file: Optional[str] = field(
        default=None,
        metadata={"help": "path to load ans2label file"},
    )

    add_object: bool = field(
        default=False,
        metadata={"help": "add object to encoder"},
    )
    valid_batch_size: int = field(
        default=20,
        metadata={"help": "valid batch size per step"},
    )
    prompt_type: Optional[str] = field(
        default=None,
        metadata={"help": "prompt_type"},
    )
    uses_ema: Optional[bool] = field(
        default=False,
        metadata={"help": "whether to use ema"},
    )
    val_inference_type: Optional[str] = field(
        default='allcand',
        metadata={"help": "inference type in validation (allcand or beamsearch), default to allcand"},
    )    
    eval_args: Optional[str] = field(
        default='{"beam":5,"unnormalized":true,"temperature":1.0}',
        metadata={
            "help": 'generation args as JSON string for inference, only activated when --val-inference-type=beamsearch'
        },
    )    


@register_task("vqa_gen_x", dataclass=VqaGenXConfig)
class VqaGenXTask(OFATask):
    def __init__(self, cfg: VqaGenXConfig, src_dict, tgt_dict):
        super().__init__(cfg, src_dict, tgt_dict)

        self.ans2label_dict = None
        if self.cfg.ans2label_file is not None:
            self.ans2label_dict = pickle.load(open(self.cfg.ans2label_file, "rb"))
        else:
            self.ans2label_dict = json.loads(self.cfg.ans2label_dict)

        self.uses_ema = self.cfg.uses_ema

        assert self.cfg.val_inference_type in ["allcand", "beamsearch", "beamsearch+similarity"], \
            "Unknown inference type encountered: {}, should be allcand or beamsearch.".format(self.cfg.val_inference_type)

    def load_dataset(self, split, epoch=1, combine=False, **kwargs):
        paths = self.cfg.data.split(',')
        assert len(paths) > 0

        if split == 'train':
            table_path = paths[(epoch - 1) % (len(paths) - 1)]
        else:
            table_path = paths[-1]
        dataset = FileDataset(table_path, self.cfg.selected_cols)

        self.datasets[split] = VqaGenXDataset(
            split,
            dataset,
            self.bpe,
            self.src_dict,
            self.tgt_dict,
            max_src_length=self.cfg.max_src_length,
            max_object_length=self.cfg.max_object_length,
            max_tgt_length=self.cfg.max_tgt_length,
            patch_image_size=self.cfg.patch_image_size,
            add_object=self.cfg.add_object,
            imagenet_default_mean_and_std=self.cfg.imagenet_default_mean_and_std,
            prompt_type=self.cfg.prompt_type
        )

    def build_model(self, cfg):
        model = super().build_model(cfg)
        if self.cfg.val_inference_type == "beamsearch":
            gen_args = json.loads(self.cfg.eval_args)
            self.generator = self.build_generator(
                [model], Namespace(**gen_args)
            )
        elif self.cfg.val_inference_type == "beamsearch+similarity":
            gen_args = json.loads(self.cfg.eval_args)
            gen_args["match_source_len"] = False
            gen_args["beam_size"] = 1
            self.generator = self.build_generator(
                [model], Namespace(**gen_args)
            )
            self.similarity_model = SentenceTransformer("stsb-mpnet-base-v2")
        else:
            raise NotImplementedError("Error: Unknown inference type encountered.")

        return model

    def build_generator(
        self, models, args, seq_gen_cls=None, extra_gen_cls_kwargs=None, prefix_allowed_tokens_fn=None,
    ):
        seq_generator = super().build_generator(models, args, seq_gen_cls, extra_gen_cls_kwargs, prefix_allowed_tokens_fn)

        return seq_generator

    def valid_step(self, sample, model, criterion, **extra_kwargs):
        loss, sample_size, logging_output = super().valid_step(sample, model, criterion)

        if self.uses_ema:
            assert 'ema_model' in extra_kwargs and extra_kwargs['ema_model'] is not None
        if self.uses_ema:
            eval_model = extra_kwargs['ema_model']
        else:
            eval_model = model

        eval_model.eval()
        with torch.no_grad():
            if self.cfg.val_inference_type in ["beamsearch", "beamsearch+similarity"]:
                raw_hyps = self.inference_step(self.generator, [eval_model], sample, prefix_tokens=sample["prefix_tokens"])
                hyps = []
                expls = []
                for i, sample_id in enumerate(sample["id"].tolist()):
                    detok_hypo_str = decode_fn(raw_hyps[i][0]["tokens"], self.tgt_dict, self.bpe, self.generator).strip()
                    print(detok_hypo_str)
                    if "because" in detok_hypo_str:
                        expls.append(f"because {detok_hypo_str.split('because')[1]}")
                        hyps.append(detok_hypo_str.split('because')[0])
                    else:
                        hyps.append(detok_hypo_str)
                        expls.append("")
            else:
                raise NotImplementedError("Error: Unknown inference type encountered.")

        scores = [ref_dict.get(hyp, 0) for ref_dict, hyp in zip(sample['ref_dict'], hyps)]
        target_expls = [decode_fn(x[x.ne(1)], self.tgt_dict, self.bpe, self.generator).strip() for x in sample["explanations"]]
        expl_scores = [self.compute_similarity(expl, target) for expl, target in zip(expls, target_expls)]
        logging_output["_vqa_score_sum"] = sum(scores)
        logging_output["_vqa_cnt"] = len(scores)
        logging_output["_expl_score_sum"] = sum(expl_scores)
        logging_output["_expl_cnt"] = len(expl_scores)
        logging_output["_total_sum_scores"] = sum([scores[i] * expl_scores[i] for i in range(len(scores))])
        logging_output["_total_cnt"] = len(scores)

        return loss, sample_size, logging_output

    def reduce_metrics(self, logging_outputs, criterion):
        super().reduce_metrics(logging_outputs, criterion)

        def sum_logs(key):
            import torch
            result = sum(log.get(key, 0) for log in logging_outputs)
            if torch.is_tensor(result):
                result = result.cpu()
            return result

        def compute_score(meters, sum_key, cnt_key):
            score = meters[sum_key].sum / meters[cnt_key].sum
            score = score if isinstance(score, float) else score.item()
            return round(score, 4)

        if sum_logs("_vqa_cnt") > 0:
            metrics.log_scalar("_vqa_score_sum", sum_logs("_vqa_score_sum"))
            metrics.log_scalar("_vqa_cnt", sum_logs("_vqa_cnt"))
            metrics.log_derived("vqa_score", partial(compute_score, sum_key="_vqa_score_sum", cnt_key="_vqa_cnt"))

        if sum_logs("_expl_cnt") > 0:
            metrics.log_scalar("_expl_score_sum", sum_logs("_expl_score_sum"))
            metrics.log_scalar("_expl_cnt", sum_logs("_expl_cnt"))
            metrics.log_derived("expl_score", partial(compute_score, sum_key="_expl_score_sum", cnt_key="_expl_cnt"))

        if sum_logs("_total_cnt") > 0:
            metrics.log_scalar("_total_sum_scores", sum_logs("_total_sum_scores"))
            metrics.log_scalar("_total_cnt", sum_logs("_total_cnt"))
            metrics.log_derived("total_score", partial(compute_score, sum_key="_total_sum_scores", cnt_key="_total_cnt"))

    def compute_similarity(self, target, expl):
        embeds = torch.from_numpy(self.similarity_model.encode([target, expl]))
        return torch.clip(torch.nn.functional.cosine_similarity(embeds[0], embeds[1], dim=0), min=0, max=1).item()
