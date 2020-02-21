#!/bin/bash
set -e

# Make all symlinks inside the root relative, and copy everything outside the root
fix_symlinks() {
    echo "Fixing symlinks..."
    find "$1" -type l -print0 |
        while IFS= read -r -d $'\0' file; do
            # Get the canonical path for the file
            link=$(readlink -f "${file}")

            # Remove the link
            rm "${file}"

            if [[ "${link}" = $1* ]]
            then
                # It's a subdirectory of the root so create a relative symlink
                ln -sr "${link}" "${file}"
            else
                # Not a subdirectory so copy
                cp -r "${link}" "${file}"
            fi

            echo "${link}" "${file}"
        done
}

fix_symlinks "$1"