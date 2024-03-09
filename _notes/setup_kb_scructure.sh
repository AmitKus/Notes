#!/bin/bash

# Directory path for the knowledgebase
DIR_PATH=${1:-$(pwd)}


# Check if directory path was provided
if [ -z "$DIR_PATH" ]; then
  echo "Usage: $0 <directory_path>"
  exit 1
fi

# Create base directory if it doesn't exist
mkdir -p "$DIR_PATH"

# Navigate to the directory
cd "$DIR_PATH" || exit

# Define the structure of directories and files
declare -a structure=(
  "_01_Introduction/README.md"
  "_02_HowToUseThisKnowledgebase/README.md"
  "_03_Models/README.md"
  "_03_Models/_01_Inference.md"
  "_03_Models/_02_Performance.md"
  "_03_Models/_03_Training.md"
  "_03_Models/_04_Visions.md"
  "_04_Tools/README.md"
  "_05_Workflows/README.md"
  "_05_Workflows/_01_Agents.md"
  "_05_Workflows/_02_RAG.md"
  "_06_ConferencesAndPapers/README.md"
  "_07_AdditionalResources/README.md"
  "_07_AdditionalResources/_01_Notes.md"
  "_07_AdditionalResources/_02_Pics.md"
  "_08_Guides/README.md"
  "_08_Guides/_01_FindingWhatYouNeed.md"
  "_08_Guides/_02_UsingTheToolsAndModels.md"
  "_09_Contributing/README.md"
  "_10_CaseStudies/README.md"
  "_11_Tutorials/README.md"
  "_12_Glossary/README.md"
  "_13_FAQ/README.md"
  "_14_CommunityContributions/README.md"
)

# Create directories and files
for item in "${structure[@]}"; do
  file_path="$DIR_PATH/$item"
  dir_path=$(dirname "$file_path")

  # Create directory if it doesn't exist
  mkdir -p "$dir_path"

  # Create file if it doesn't exist
  if [ ! -f "$file_path" ]; then
    touch "$file_path"
    echo "# $(basename "$item" .md)" > "$file_path"
    echo "Content coming soon." >> "$file_path"
  fi
done

# Verify creation
echo "Verifying the structure..."

for item in "${structure[@]}"; do
  if [ -f "$DIR_PATH/$item" ]; then
    echo "Created: $DIR_PATH/$item"
  else
    echo "Missing: $DIR_PATH/$item"
  fi
done

echo "Structure verification completed."
