#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""YOLOX-nano PPE object detector - 6 classes (helmet/vest/gloves, each as a
present/absent pair). Mirrors exps/custom/fire_nano.py's structure exactly,
including the two fixes that run already required on this machine (see
fire_nano.py's own comments for the full incident writeup):

  1. In-training COCO-mAP eval is disabled (needs an MSVC toolchain this machine
     doesn't have) - use ppe_validate_onnx.py's per-class/per-domain precision-recall
     harness after training instead, which is the metric that actually matters here
     (loss convergence says nothing about the CCTV domain gap this model is exposed
     to for gloves - see ppe_taxonomy.py's module docstring).
  2. `tools/train.py --resume` must be invoked WITHOUT `-c` (see fire_nano.py).

glasses_present/glasses_absent are DELIBERATELY EXCLUDED (not just uncovered like
shoes) - the 2026-09-16 data review found 65-72% near-duplication in both classes
via DCT perceptual hash (see training/scripts/_artifact/ppe_review.html#glasses_present)
and dropped them from this training run. `training/scripts/ppe_to_coco.py` performs
the exclusion at conversion time; category order here is
['helmet_present', 'helmet_absent', 'vest_present', 'vest_absent', 'gloves_present',
'gloves_absent'] - ppe_taxonomy.UNIFIED_CLASSES with glasses removed, order otherwise
unchanged.

data_dir below is `ppe_to_coco.py`'s output (25,657 images / 73,755 boxes across
train/val/test) - conversion run 2026-09-16, see that script's own docstring for the
glasses-exclusion and image-dropping logic (images with only glasses boxes are
dropped entirely, nothing left to annotate).
"""
import os

from yolox.exp import Exp as MyExp

# Same in-training-eval disable as fire_nano.py - see that file's comments for why.
import yolox.core.trainer as _trainer_mod  # noqa: E402


def _skip_eval(self):
    from loguru import logger as _logger
    _logger.info(
        "evaluate_and_save_model: skipped (no C++ build toolchain on this machine "
        "for yolox.layers.COCOeval_opt) - saving current weights as last_epoch/best "
        "unconditionally. Use training/scripts/ppe_validate_onnx.py after training "
        "for the metric that actually matters here: per-class, per-domain "
        "precision/recall, not COCO mAP."
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
        # len(ppe_taxonomy.UNIFIED_CLASSES) minus glasses_present/glasses_absent
        # (excluded, see module docstring) - hardcoded rather than imported to keep
        # this exp file importable standalone by YOLOX's own tooling (which adds
        # only the YOLOX repo root to sys.path, not training/scripts/); keep in sync
        # by hand if ppe_to_coco.py's KEPT_CLASSES changes.
        self.num_classes = 6

        # Same input size as yolox_person and firesmoke - this is what makes a
        # trained model drop into the existing YoloxOnnx wrapper with zero code
        # changes (backend/src/perimeter/detect/yolox_onnx.py assumes a fixed input
        # size read from the ONNX graph itself, so any square size works, but
        # matching the others keeps the cadence/latency budget in PLAN.md section 4
        # comparable across all three detectors sharing one CPU budget).
        self.input_size = (416, 416)
        self.test_size = (416, 416)
        self.random_size = (10, 20)
        self.mosaic_scale = (0.5, 1.5)
        self.mosaic_prob = 0.5
        self.enable_mixup = False

        self.data_dir = "A:/fsbd_training/ppe_coco"
        self.train_ann = "train.json"
        self.val_ann = "val.json"

        self.max_epoch = 40
        self.no_aug_epochs = 6
        self.warmup_epochs = 3
        self.eval_interval = self.max_epoch + 1  # never triggers - see _skip_eval above
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
