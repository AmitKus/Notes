module Jekyll
  class NotesGenerator < Generator
    safe true
    priority :high

    def generate(site)
      notes_collection = site.collections['notes']
      notes_dir = File.join(site.source, '_notes')

      # Find all markdown files in subdirectories
      Dir.glob(File.join(notes_dir, '**', '*.md')).each do |file_path|
        relative_path = Pathname.new(file_path).relative_path_from(Pathname.new(notes_dir)).to_s

        # Skip if already processed (in root of _notes)
        next unless relative_path.include?('/')

        # Create a new document for this file
        doc = Document.new(
          file_path,
          site: site,
          collection: notes_collection
        )

        # Read and process the document
        doc.read

        # Add to collection if not already there
        unless notes_collection.docs.any? { |d| d.path == doc.path }
          notes_collection.docs << doc
        end
      end
    end
  end
end