#!/bin/bash

# Default parameters
file_extension="rst" # default file extension
root_dir="./" # default root directory

# Process command-line arguments
while getopts "e:d:" opt; do
  case ${opt} in
    e ) file_extension=$OPTARG ;;
    d ) root_dir=$OPTARG ;;
    \? ) echo "Usage: cmd [-e file_extension] [-d root_dir]"
         exit 1 ;;
  esac
done

# Name of the combined document, assuming the extension is the same as the input files
combined_doc="all_docs.${file_extension}"

# Start with a clean combined document file
echo "" > "${root_dir}/${combined_doc}"

# Loop over all files with the specified extension in all subdirectories, excluding the combined_doc
find "${root_dir}" -name "*.${file_extension}" -type f | grep -v "${combined_doc}" | while read -r file
do
    # Get the file name without the path and the extension
    filename=$(basename -- "$file")
    filename="${filename%.*}"

    # Add the file name as a section header
    # For .md files, we'll use # for the section title
    if [ "${file_extension}" = "md" ]; then
        echo "# ${filename}" >> "${root_dir}/${combined_doc}"
    else
        echo "${filename}" | awk '{print toupper($0)}' >> "${root_dir}/${combined_doc}"
        printf '=%.0s' $(seq 1 $(echo -n "${filename}" | wc -c)) >> "${root_dir}/${combined_doc}"
    fi
    echo "" >> "${root_dir}/${combined_doc}"

    # Append the contents of the file to the combined document
    cat "$file" >> "${root_dir}/${combined_doc}"

    # Add a new line after each file to separate sections
    echo "" >> "${root_dir}/${combined_doc}"
done
