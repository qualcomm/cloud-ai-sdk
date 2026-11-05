####################################################################################################
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear
####################################################################################################

from qaic_models.vision.ultralytics.common.classify_demo import run_ultralytics_classify_demo

from .model import DEFAULT_VARIANT, SUPPORTED_VARIANTS, YOLO26Classify


def main() -> None:
    run_ultralytics_classify_demo(
        YOLO26Classify,
        description="Run YOLO26 classify on a single image.",
        default_variant=DEFAULT_VARIANT,
        supported_variants=SUPPORTED_VARIANTS,
    )


if __name__ == "__main__":
    main()
