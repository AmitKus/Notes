# Jekyll Site Setup Guide

This document explains how to run the Jekyll-based knowledge base locally.

## Prerequisites

- Ruby 3.0+ installed
- Basic gems: jekyll, webrick, nokogiri, jekyll-last-modified-at

## Quick Start

1. **Install required gems:**
   ```bash
   gem install jekyll webrick nokogiri jekyll-last-modified-at --user-install
   ```

2. **Add gem path to your session:**
   ```bash
   # Find your gem path first:
   gem environment | grep "USER INSTALLATION DIRECTORY"

   # Then add the bin directory to PATH (adjust path as needed):
   export PATH="$(gem environment | grep "USER INSTALLATION DIRECTORY" | cut -d: -f2 | xargs)/bin:$PATH"

   # Alternative - use the common path pattern:
   export PATH="$HOME/.local/share/gem/ruby/3.0.0/bin:$PATH"
   ```

3. **Temporarily move Gemfile (Ruby version compatibility issue):**
   ```bash
   mv Gemfile Gemfile.backup
   ```

4. **Start the Jekyll server:**
   ```bash
   jekyll serve --host 0.0.0.0
   ```

5. **Access the site:**
   - Homepage: http://127.0.0.1:4000/
   - All Notes: http://127.0.0.1:4000/all-notes/

## Key URLs

- **Homepage**: http://127.0.0.1:4000/
- **All Categories**: http://127.0.0.1:4000/all-notes/
- **Interview Notes**: http://127.0.0.1:4000/_notes/_16_Interview/Coding/
- **Papers**: http://127.0.0.1:4000/_notes/_17_Papers/
- **Models**: http://127.0.0.1:4000/_notes/_03_Models/

## Important Files Created/Modified

### Custom Plugin
- `_plugins/subdirectory_processor.rb` - Processes markdown files in subdirectories of _notes

### Configuration Changes
- `_config.yml` - Updated collections permalink to `/_notes/:path/`
- `_config.yml` - Added `_pages` to include list

### Pages Added
- `_pages/all-notes.md` - Category listing page
- `_pages/index.md` - Updated homepage with navigation

## Troubleshooting

### Server won't start
- Check Ruby version: `ruby --version`
- Ensure gems are installed: `gem list | grep jekyll`
- Make sure Gemfile is moved: `ls Gemfile*`

### 404 Errors
- Clear browser cache (Ctrl+F5)
- Use correct URLs with trailing slashes
- Check if server is running on correct port

### Missing Notes
- Verify plugin is working: check `_site/_notes/` for generated HTML files
- Restart server after config changes

## One-Line Startup Command

```bash
# Navigate to your notes directory first, then run:
mv Gemfile Gemfile.backup 2>/dev/null; export PATH="$(gem environment | grep "USER INSTALLATION DIRECTORY" | cut -d: -f2 | xargs)/bin:$PATH" && jekyll serve --host 0.0.0.0

# Or with fixed path:
mv Gemfile Gemfile.backup 2>/dev/null; export PATH="$HOME/.local/share/gem/ruby/3.0.0/bin:$PATH" && jekyll serve --host 0.0.0.0
```

## Stopping the Server

Press `Ctrl+C` in the terminal where Jekyll is running.

## Restoring for Git

After you're done, restore the Gemfile:
```bash
mv Gemfile.backup Gemfile
```