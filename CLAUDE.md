# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a Jekyll-based digital garden/knowledge base focused on artificial intelligence topics. It's structured as a static site generator with markdown content organized in the `_notes` directory.

## Commands

### Setup and Development

**Quick Start (Ruby 3.0.x compatible):**
```bash
# Install gems directly (bypass bundler compatibility issues)
gem install jekyll webrick nokogiri jekyll-last-modified-at --user-install

# Start server (moves Gemfile temporarily for compatibility)
mv Gemfile Gemfile.backup 2>/dev/null; export PATH="$(gem environment | grep "USER INSTALLATION DIRECTORY" | cut -d: -f2 | xargs)/bin:$PATH" && jekyll serve --host 0.0.0.0
```

**Alternative (if bundler works):**
```bash
# Install dependencies
bundle install

# Serve the site locally with live reload
bundle exec jekyll serve

# Build the site for production
bundle exec jekyll build

# Build with verbose output for debugging
bundle exec jekyll build --trace
```

**Restore after development:**
```bash
# Restore Gemfile for git commits
mv Gemfile.backup Gemfile 2>/dev/null
```

### Common Jekyll Commands

```bash
# Clean the site (removes _site directory)
bundle exec jekyll clean

# Build and serve with drafts visible
bundle exec jekyll serve --drafts

# Serve on a different port
bundle exec jekyll serve --port 4001
```

## Architecture

### Content Structure

The knowledge base follows a numbered hierarchical organization in `_notes/`:
- `_00_Attachments/` - Centralized storage for all attachments and assets
- `_01_Introduction/` - Entry point documentation
- `_02_HowToUseThisKnowledgebase/` - Usage guides and project TODO
- `_03_Models/` - AI model documentation (inference, performance, training, visions, long context, multimodal, state space models, compound systems, model merging, fine-tuning)
- `_04_Tools/` - AI tools and frameworks (LlamaIndex, vLLM, CUTLASS, Outlines)
- `_05_Workflows/` - Workflows like agents and RAG
- `_06_ConferencesAndPapers/` - Academic resources and research papers
- `_07_AdditionalResources/` - Supporting materials (notes, pics, quotes)
- `_08_Guides/` - Practical guides for finding and using resources
- `_09_Contributing/` - Contribution guidelines
- `_10_CaseStudies/` - Real-world examples
- `_11_Tutorials/` - Step-by-step tutorials (including PyTorch, TensorRT, Cursor)
- `_12_Glossary/` - AI terminology
- `_13_FAQ/` - Frequently asked questions
- `_14_CommunityContributions/` - User contributions
- `_15_Courses/` - Educational content (CUDA Mode, LangGraph, Llama3.2, etc.)
- `_16_Interview/` - Interview preparation materials and coding practice
- `_17_Papers/` - Organized research papers by category (Agents, Multimodal, RAG, Reasoning, Test Time)

### Jekyll Configuration

- **Collections**: Notes are configured as a Jekyll collection with output enabled
- **Permalinks**: Uses pretty URLs (`:slug` pattern)
- **Plugins**: Uses `jekyll-last-modified-at` for tracking content updates
- **Layouts**: Default layout with special "note" layout for `_notes` content
- **Sass**: Compressed CSS output from `_sass/` directory

### Key Files

- `_config.yml` - Jekyll site configuration
- `Gemfile` - Ruby dependencies (Jekyll 4.3, webrick, nokogiri)
- `netlify.toml` - Netlify deployment configuration
- `_layouts/` - HTML templates for pages
- `_includes/` - Reusable HTML components
- `_sass/` - Stylesheet sources
- `assets/` - Static assets (CSS, JS, images)

## Development Notes

- The site is configured for Netlify deployment with build output to `_site/`
- External links open in new tabs by default
- HTML extensions are disabled for cleaner URLs
- The main branch is used for production deployments