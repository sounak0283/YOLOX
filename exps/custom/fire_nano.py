#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""YOLOX-nano fine-tune for fire/smoke detection.

Per PLAN.md Section 5.3: 2 classes {0: fire, 1: smoke}, same architecture/wrapper
as the shared person detector (detect/yolox_onnx.py). Trained from the COCO-pretrained
yolox_nano.pth checkpoint on D-Fire (see training/scripts/dfire_to_coco.py for the
COCO-format conversion this exp's data_dir/train_ann/val_ann point at).

Epoch count and batch size are reduced from PLAN.md's 100-150 epoch recipe to fit a
single 4GB-VRAM GPU (GTX 1650) within a practical session, per docs/MODEL_TRAINING.md.
Revisit epoch count once a full training budget (rented GPU time) is available.
"""
import os

from yolox.exp import Exp as MyExp

# Disable in-training COCO-mAP evaluation entirely. yolox.evaluators.coco_evaluator
# unconditionally imports yolox.layers.COCOeval_opt, which JIT-compiles a C++
# extension via torch.utils.cpp_extension.load() - that needs a working MSVC
# toolchain, which this machine does not have (Ninja alone is not enough; there is
# no cl.exe on PATH). Left as-is, this crashes mid-training every time
# Trainer.evaluate_and_save_model() runs (trainer.py's before_epoch() also forces
# eval_interval=1 for the last `no_aug_epochs`, regardless of the exp's own setting,
# so this can't be avoided by config alone). trainer.py wraps the whole training
# loop in try/except/finally, so the crash was previously silent: it logged
# "Training of experiment is done" and exited 0 after only 5 of 40 epochs.
# Not a real loss of signal - PLAN.md section 5.4 explicitly says not to tune this
# model on COCO mAP; the real evaluation is false-alarms/hour on site video via the
# K-of-N gate, run after export, not a held-out mAP score during training.
import yolox.core.trainer as _trainer_mod  # noqa: E402


def _skip_eval(self):
    from loguru import logger as _logger
    _logger.info(
        "evaluate_and_save_model: skipped (no C++ build toolchain on this machine "
        "for yolox.layers.COCOeval_opt) - saving current weights as last_epoch/best "
        "unconditionally instead of gating on AP."
    )
    self.best_ap = 0.0
    self.save_ckpt("last_epoch", update_best_ckpt=True, ap=0.0)
    if self.save_history_ckpt:
        self.save_ckpt(f"epoch_{self.epoch + 1}", ap=0.0)


_trainer_mod.Trainer.evaluate_and_save_model = _skip_eval


class Exp(MyExp):
    def __init__(self):
        super(Exp, self).__init__()
        self.depth = 0.33
        self.width = 0.25
        self.num_classes = 2

        # Train at 416 as PLAN.md specifies "input 640 for training, export at 416" —
        # but 640 training on 4GB VRAM risks OOM with mosaic augmentation; train and
        # export at 416 directly, consistent with the shared person-detector wrapper.
        self.input_size = (416, 416)
        self.test_size = (416, 416)
        self.random_size = (10, 20)
        self.mosaic_scale = (0.5, 1.5)
        self.mosaic_prob = 0.5
        self.enable_mixup = False

        self.data_dir = "A:/fsbd_training/dfire_coco"
        self.train_ann = "train.json"
        self.val_ann = "val.json"

        self.max_epoch = 40
        self.no_aug_epochs = 6
        self.warmup_epochs = 3
        # Periodic in-training COCO-mAP eval is disabled: it needs a JIT-compiled C++
        # extension (yolox.layers.fast_coco_eval_api) that requires a working MSVC
        # toolchain on Windows, which isn't available here and crashed the training
        # process outright (caught by trainer.py's try/except/finally, which then
        # logs a misleading "Training of experiment is done" even though only 5 of 40
        # epochs had run). Not a real loss: PLAN.md section 5.4 explicitly says not to
        # tune this model on mAP - the real evaluation is false-alarms/hour on site
        # video via the K-of-N gate, done after export, not a held-out mAP score
        # during training. eval_interval > max_epoch means it never triggers.
        self.eval_interval = self.max_epoch + 1
        self.print_interval = 20
        self.data_num_workers = 2

        self.exp_name = os.path.split(os.path.realpath(__file__))[1].split(".")[0]

    def get_model(self, sublinear=False):
        import torch.nn as nn

        def init_yolo(M):
            for m in M.modules():
                if isinstance(m, nn.BatchNorm2d):
                    m.eps = 1e-3
                    m.momentum = 0.03

        if "model" not in self.__dict__:
            from yolox.models import YOLOX, YOLOPAFPN, YOLOXHead
            in_channels = [256, 512, 1024]
            backbone = YOLOPAFPN(
                self.depth, self.width, in_channels=in_channels,
                act=self.act, depthwise=True,
            )
            head = YOLOXHead(
                self.num_classes, self.width, in_channels=in_channels,
                act=self.act, depthwise=True,
            )
            self.model = YOLOX(backbone, head)

        self.model.apply(init_yolo)
        self.model.head.initialize_biases(1e-2)
        return self.model
