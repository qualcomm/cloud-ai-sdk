# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear

from __future__ import annotations


def build_export_cfg(args):
    from detectron2.config import get_cfg
    from detectron2.projects.point_rend import add_pointrend_config

    cfg = get_cfg()
    add_pointrend_config(cfg)
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list([
        "MODEL.WEIGHTS", args.weights,
        "MODEL.DEVICE", "cpu",
        "INPUT.MIN_SIZE_TEST", str(args.input_height),
        "INPUT.MAX_SIZE_TEST", str(args.input_width),
        "MODEL.RPN.PRE_NMS_TOPK_TEST", str(args.proposal_count),
        "MODEL.RPN.POST_NMS_TOPK_TEST", str(args.proposal_count),
        "MODEL.ROI_BOX_HEAD.POOLER_SAMPLING_RATIO", "2",
        "MODEL.ROI_MASK_HEAD.POOLER_SAMPLING_RATIO", "2",
        "MODEL.ROI_KEYPOINT_HEAD.POOLER_SAMPLING_RATIO", "2",
    ])
    cfg.freeze()
    return cfg
