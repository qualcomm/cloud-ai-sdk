# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear

from torch_qaic.custom_ops import QAicBuildExt, HexagonKernelExtension
from setuptools import setup

setup(
    ext_modules=[HexagonKernelExtension(sources=["scale_kernel.cpp"])],
    cmdclass={"build_ext": QAicBuildExt},
)
