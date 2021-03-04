# Copyright (c) 2020 UATC, LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import subprocess
import sys


def compile_requirements(requirements, lockfile):
    """
    Run piptools compile over a requirement file to generate an output lockfile.
    We use `--allow-unsafe` as we want to include wheel, pip, and setuptools in our output.
    """
    subprocess.check_call(
        [
            sys.executable,
            "-m",
            "piptools",
            "compile",
            "--no-header",
            "--no-annotate",
            "--allow-unsafe",
            "--no-emit-index-url",
            "--no-emit-trusted-host",
            "-q",
            "-o",
            lockfile,
            requirements,
        ],
    )
