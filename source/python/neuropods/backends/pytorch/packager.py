#
# Uber, Inc. (c) 2018
#

from neuropods.backends.python.packager import create_python_neuropod

# Since we can package a pytorch model by just treating it as python code,
# create an alias
# TODO: do we want to create a separate function for pytorch that just wraps
# `create_python_neuropod`?
# TODO: do we want to write different docstrings for python and pytorch?
create_pytorch_neuropod = create_python_neuropod
