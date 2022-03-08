# Copyright (c) 2020 The Neuropod Authors
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

import inspect
import re

class Function(object):
    """
    Represents a function
    """

    def __init__(self, name, shortdoc="", params=[]):
        super(Function, self).__init__()
        self.name = name
        self.shortdoc = shortdoc
        self.params = params


class Parameter(object):
    """
    Represents a parameter to a function
    """

    def __init__(self, name):
        super(Parameter, self).__init__()
        self.name = name
        self.docs_arr = []
        self.has_default = False
        self.default = None

    def add_doc_line(self, line):
        """
        Add a line of documentation
        """
        self.docs_arr.append(line)

    def get_clean_doc(self):
        """
        Get a clean version of the documentation for this parameter
        """
        # Strips common leading whitespace
        return inspect.cleandoc("\n".join(self.docs_arr))

    def set_default(self, value):
        """
        Set the default value for this parameter
        """
        self.has_default = True
        self.default = value


def parse_docstring(f):
    """
    Parse the docstring of `f` and return a list of parameters
    """
    doc = inspect.cleandoc(f.__doc__)

    lines = doc.splitlines()
    params = []
    currentparam = None
    shortdoc = ""

    def flush_param_if_necessary():
        if currentparam:
            params.append(currentparam)

    # Loop through all the lines of the docstring
    for i, line in enumerate(lines):
        match = re.match(r':param\s+(\w+):\s+(.*)', line)
        if match:
            # This is a new param
            flush_param_if_necessary()

            currentparam = Parameter(name=match.group(1))
            currentparam.add_doc_line(match.group(2))
        elif i == 0:
            # If this is the first line and it's not a new parameter,
            # this is the shortdoc
            shortdoc = line
        elif currentparam is not None:
            # This is part of the documentation of the current parameter
            currentparam.add_doc_line(line)

    # Make sure we get the final parameter
    flush_param_if_necessary()

    # Check for default values
    params_required = []
    params_defaults = []
    for param in params:
        if param.name in f.neuropod_default_args:
            param.set_default(f.neuropod_default_args[param.name])
            params_defaults.append(param)
        else:
            params_required.append(param)

    return Function(name=f.__name__, shortdoc=shortdoc, params=params_required + params_defaults)

def write_doc(filename, f):
    """
    Write the documentation for a Function `f` to `filename`
    """
    with open(filename, "w") as docfile:

        # Write the title and shortdoc
        docfile.write("## " + f.name + "\n")
        docfile.write(f.shortdoc + "\n")

        # Write out the signature
        docfile.write("```\n")
        docfile.write(f.name + "(\n")
        for param in f.params:
            if param.has_default:
                docfile.write("    " + param.name + " = " + str(param.default) + ",\n")
            else:
                docfile.write("    " + param.name + ",\n")
        docfile.write(")\n")
        docfile.write("```\n")

        # Write out the list of params with their docs
        docfile.write("### Params:\n")
        for param in f.params:
            docfile.write("#### " + param.name + "\n")
            if param.has_default:
                docfile.write("*default: `" + str(param.default) + "`*\n\n")

            docfile.write(param.get_clean_doc() + "\n\n")

def write_doc_for_packager(packager, filename):
    """
    Writes the documentation for a packager to a file
    """
    f = parse_docstring(packager)
    write_doc(filename, f)

if __name__ == '__main__':
    import argparse
    import os

    from neuropod.packagers import create_tensorflow_neuropod, \
                                    create_pytorch_neuropod, \
                                    create_keras_neuropod, \
                                    create_torchscript_neuropod

    parser = argparse.ArgumentParser(description='Generate markdown documentation for the Neuropod packagers')
    parser.add_argument('out_dir', help='The output directory to write the docs to')
    args = parser.parse_args()

    packager_mapping = {
        "tensorflow.md": create_tensorflow_neuropod,
        "pytorch.md": create_pytorch_neuropod,
        "keras.md": create_keras_neuropod,
        "torchscript.md": create_torchscript_neuropod,
    }

    for filename, packager in packager_mapping.items():
        write_doc_for_packager(packager, os.path.join(args.out_dir, filename))
