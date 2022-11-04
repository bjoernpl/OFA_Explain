# Copyright 2022 The OFA-Sys Team. 
# All rights reserved.
# This source code is licensed under the Apache 2.0 license 
# found in the LICENSE file in the root directory.

from dataclasses import dataclass, field

import torch
from omegaconf import II

from fairseq import metrics, utils
from fairseq.criterions import FairseqCriterion, register_criterion
from fairseq.dataclass import FairseqDataclass


@dataclass
class CrossEntropyExplRegularizedCriterionConfig(FairseqDataclass):
    report_accuracy: bool = field(
        default=False,
        metadata={"help": "report accuracy metric"},
    )
    sentence_avg: bool = II("optimization.sentence_avg")
    cosine_loss_scale: float = field(
        default=1.0,
        metadata={"help": "scale for cosine loss"},
    )


@register_criterion(
    "cross_entropy_expl_reg", dataclass=CrossEntropyExplRegularizedCriterionConfig
)
class CrossEntropyExplRegularizedCriterion(FairseqCriterion):
    def __init__(
        self,
        task,
        sentence_avg,
        report_accuracy=False,
        cosine_loss_scale=1.0
    ):
        super().__init__(task)
        self.loss = torch.nn.CrossEntropyLoss(label_smoothing=0.1)
        self.sentence_avg = sentence_avg
        self.report_accuracy = report_accuracy
        self.cosine_loss_scale = cosine_loss_scale

    def forward(self, model, sample, update_num=0, reduce=True):
        """Compute the loss for the given sample.

        Returns a tuple with three elements:
        1) the loss
        2) the sample size, which is used as the denominator for the gradient
        3) logging outputs to display while training
        """
        net_output = model(**sample["net_input"])
        attn = net_output[1]["attn"][0]
        target = sample["target"]
        # set attention to 0 for padding tokens
        attention = attn * target.ne(self.padding_idx).unsqueeze(-1)
        # get indices of 'because' token (142)
        indices = torch.tensor([a[1] for a in torch.where(target.eq(142))], device=target.device).unsqueeze(1)
        bs = target.size(0)
        ans_indices = [
            torch.range(0, indices[i].item()-1, device=target.device, dtype=torch.long)
            for i in range(bs)
        ]
        expl_indices = [
            torch.range(indices[i].item()+1, target.shape[1]-1, device=target.device, dtype=torch.long)
            for i in range(bs)
        ]
        ans_att_sum = torch.zeros((bs, attention.shape[-1]), device=target.device)
        expl_att_sum = torch.zeros((bs, attention.shape[-1]), device=target.device)
        for i in range(bs):
            ans_att_sum[i] = attention[i, ans_indices[i]].sum(0)
            expl_att_sum[i] = attention[i, expl_indices[i]].sum(0)
        cos_sim = torch.cosine_similarity(ans_att_sum, expl_att_sum, dim=1)
        cos_loss = ((1 - cos_sim) / 2).mean()
        cos_loss = cos_loss * self.cosine_loss_scale

        loss, nll_loss = self.compute_loss(model, net_output, sample, update_num, reduce=reduce)
        loss += cos_loss
        sample_size = (
            sample["target"].size(0) if self.sentence_avg else sample["ntokens"]
        )
        logging_output = {
            "loss": loss.data,
            "nll_loss": nll_loss.data,
            "cos_loss": cos_loss.data,
            "ntokens": sample["ntokens"],
            "nsentences": sample["nsentences"],
            "sample_size": sample_size,
        }
        if self.report_accuracy:
            n_correct, total = self.compute_accuracy(model, net_output, sample)
            logging_output["n_correct"] = utils.item(n_correct.data)
            logging_output["total"] = utils.item(total.data)
        return loss, sample_size, logging_output

    def get_lprobs_and_target(self, model, net_output, sample):
        lprobs = model.get_normalized_probs((net_output[0], ), log_probs=True)
        target = model.get_targets(sample, net_output)
        return lprobs.view(-1, lprobs.size(-1)), target.view(-1)

    def compute_loss(self, model, net_output, sample, update_num, reduce=True):
        lprobs, target = self.get_lprobs_and_target(model, net_output, sample)
        lprobs = lprobs[target != self.padding_idx]
        target = target[target != self.padding_idx]
        nll_loss = -lprobs.gather(dim=-1, index=target.unsqueeze(-1))
        loss = self.loss(lprobs, target)
        return loss, nll_loss.sum()

    def compute_accuracy(self, model, net_output, sample):
        lprobs, target = self.get_lprobs_and_target(model, net_output, sample)
        mask = target.ne(self.padding_idx)
        n_correct = torch.sum(
            lprobs.argmax(1).masked_select(mask).eq(target.masked_select(mask))
        )
        total = torch.sum(mask)
        return n_correct, total

    @classmethod
    def reduce_metrics(cls, logging_outputs) -> None:
        """Aggregate logging outputs from data parallel training."""
        loss_sum = sum(log.get("loss", 0) for log in logging_outputs)
        nll_loss_sum = sum(log.get("nll_loss", 0) for log in logging_outputs)
        cos_loss_sum = sum(log.get("cos_loss", 0) for log in logging_outputs)

        ntokens = sum(log.get("ntokens", 0) for log in logging_outputs)
        nsentences = sum(log.get("nsentences", 0) for log in logging_outputs)
        sample_size = sum(log.get("sample_size", 0) for log in logging_outputs)

        metrics.log_scalar(
            "loss", loss_sum / sample_size, sample_size, round=4
        )
        metrics.log_scalar(
            "cos_loss", cos_loss_sum / sample_size, sample_size, round=4
        )
        metrics.log_scalar(
            "nll_loss", nll_loss_sum / sample_size, sample_size, round=4
        )
        metrics.log_scalar(
            "ntokens", ntokens, 1, round=3
        )
        metrics.log_scalar(
            "nsentences", nsentences, 1, round=3
        )
        metrics.log_scalar(
            "sample_size", sample_size, 1, round=3
        )

        total = utils.item(sum(log.get("total", 0) for log in logging_outputs))
        if total > 0:
            metrics.log_scalar("total", total)
            n_correct = utils.item(
                sum(log.get("n_correct", 0) for log in logging_outputs)
            )
            metrics.log_scalar("n_correct", n_correct)
            metrics.log_derived(
                "accuracy",
                lambda meters: round(
                    meters["n_correct"].sum * 100.0 / meters["total"].sum, 3
                )
                if meters["total"].sum > 0
                else float("nan"),
            )

    @staticmethod
    def logging_outputs_can_be_summed() -> bool:
        """
        Whether the logging outputs returned by `forward` can be summed
        across workers prior to calling `reduce_metrics`. Setting this
        to True will improves distributed training speed.
        """
        return True
