#
# Uber, Inc. (c) 2019
#

# Create aliases for packagers so they can be used as
#
#     `from neuropods.packagers import create_*_neuropod`
#
# instead of
#
#     `from neuropods.backends.*.packager import create_*_neuropod`
#
# In order to make sure we don't load all the backends unnecessarily, we need to lazy-load
# them when the packaging function is called
def create_alias(packager_name):
    packager_module = "neuropods.backends." + packager_name + ".packager"
    packaging_function = "create_" + packager_name + "_neuropod"

    # A wrapper that gets the actual packaging function and runs it
    def wrap(*args, **kwargs):
        module = __import__(packager_module, fromlist=[packaging_function])
        return getattr(module, packaging_function)(*args, **kwargs)

    # Create the alias
    globals()[packaging_function] = wrap


# Create aliases for all our backends
create_alias("keras")
create_alias("python")
create_alias("pytorch")
create_alias("tensorflow")
create_alias("torchscript")
