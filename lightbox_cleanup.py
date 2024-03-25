import json
from tqdm import tqdm

if __name__ == "__main__":
    lightbox_text = "{\n\t\t\t\t\"lightbox_close\": \"Close\",\n\t\t\t\t\"lightbox_next\": \"Next\",\n\t\t\t\t\"lightbox_previous\": \"Previous\",\n\t\t\t\t\"lightbox_error\": \"The requested content cannot be loaded. Please try again later.\",\n\t\t\t\t\"lightbox_start_slideshow\": \"Start slideshow\",\n\t\t\t\t\"lightbox_stop_slideshow\": \"Stop slideshow\",\n\t\t\t\t\"lightbox_full_screen\": \"Full screen\",\n\t\t\t\t\"lightbox_thumbnails\": \"Thumbnails\",\n\t\t\t\t\"lightbox_download\": \"Download\",\n\t\t\t\t\"lightbox_share\": \"Share\",\n\t\t\t\t\"lightbox_zoom\": \"Zoom\",\n\t\t\t\t\"lightbox_new_window\": \"New window\",\n\t\t\t\t\"lightbox_toggle_sidebar\": \"Toggle sidebar\"\n\t\t\t}"
    
    input_filename = 'posts.json'
    output_filename = 'cleaned_posts.json'

    print("Processing posts...")
    with open(input_filename, 'r', encoding='utf-8') as f, open(output_filename, 'w', encoding='utf-8') as out:
        for line in tqdm(f):
            cleaned_line = line.replace(lightbox_text, '')
            out.write(cleaned_line)

    print("Cleanup complete!")
