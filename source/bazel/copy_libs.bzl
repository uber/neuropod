def _copy_libs_impl(ctx):
    # Get a list of the input files
    in_files = ctx.files.libs

    # Declare the output files
    out_files = [ctx.actions.declare_file(f.basename) for f in in_files]

    ctx.actions.run_shell(
        # Input files visible to the action.
        inputs = in_files,

        # Output files that must be created by the action.
        outputs = out_files,

        # A progress message to display during the build
        progress_message = "Copying libs for target %s" % ctx.attr.name,

        # Copy all the input files to the output directory
        command = "cp -a %s %s" %
                  (
                      " ".join([f.path for f in in_files]),
                      out_files[0].dirname,
                  ),
    )

    # Tell bazel to make these files available at runtime
    runfiles = ctx.runfiles(
        files = out_files,
        collect_default = True,
    )

    return [DefaultInfo(runfiles = runfiles)]

copy_libs = rule(
    implementation = _copy_libs_impl,
    attrs = {
        "libs": attr.label(
            mandatory = True,
            allow_files = True,
            doc = "A filegroup containing the libs to copy",
        ),
    },
    doc = """
Copies all the files in a filegroup to the rule's directory and makes them available
at runtime.
""",
)
