import os
import math
from PIL import Image, ImageDraw, ImageFont
import re

def get_pairs(directory):
    files = [f for f in os.listdir(directory) if f.endswith('.ppm')]
    pairs = []
    # Build a lookup for cluster images
    cluster_lookup = {}
    for f in files:
        m = re.match(r'dimension_(\d+)_vs_dimension_(\d+)_cluster\.ppm', f)
        if m:
            key = (int(m.group(1)), int(m.group(2)))
            cluster_lookup[key] = f
    # Now look for plain images
    for f in files:
        m = re.match(r'dimension_(\d+)_vs_(\d+)\.ppm', f)
        if m:
            i, j = int(m.group(1)), int(m.group(2))
            cluster_name = cluster_lookup.get((i, j))
            if cluster_name:
                pairs.append((i, j, os.path.join(directory, f), os.path.join(directory, cluster_name)))
            else:
                pairs.append((i, j, os.path.join(directory, f), None))
    # Sort by (i, j)
    pairs.sort()
    return pairs

def combine_pairs_with_labels(directory, output_file='combined.png', font_size=56):
    pairs = get_pairs(directory)
    if not pairs:
        print("No i_vs_j pairs found in", directory)
        return

    # Load images and get max width/height
    images = []
    for i, j, plain, cluster in pairs:
        img_plain = Image.open(plain)
        img_cluster = Image.open(cluster) if cluster else None
        images.append((img_plain, img_cluster, os.path.splitext(os.path.basename(plain))[0],
                       os.path.splitext(os.path.basename(cluster))[0] if cluster else None))
    max_width = max(img.size[0] for img, _, _, _ in images)
    max_height = max(img.size[1] for img, _, _, _ in images)
    label_height = font_size + 10

    # Font for labels
    try:
        font = ImageFont.truetype("arial.ttf", font_size)
    except:
        font = ImageFont.load_default()

    # Each row: plain, cluster
    cols = 2
    rows = len(images)
    grid_width = cols * max_width
    grid_height = rows * (max_height + label_height)
    grid_img = Image.new('RGB', (grid_width, grid_height), color=(255,255,255))
    draw = ImageDraw.Draw(grid_img)

    for idx, (img_plain, img_cluster, name_plain, name_cluster) in enumerate(images):
        y = idx * (max_height + label_height)
        # Plain
        grid_img.paste(img_plain, (0, y))
        label = name_plain
        try:
            bbox = draw.textbbox((0, 0), label, font=font)
            text_w, text_h = bbox[2] - bbox[0], bbox[3] - bbox[1]
        except AttributeError:
            text_w, text_h = font.getsize(label)
        text_x = (max_width - text_w) // 2
        text_y = y + max_height + 2
        draw.text((text_x, text_y), label, fill=(0,0,0), font=font)

        # Cluster
        if img_cluster:
            grid_img.paste(img_cluster, (max_width, y))
            label = name_cluster
            try:
                bbox = draw.textbbox((0, 0), label, font=font)
                text_w, text_h = bbox[2] - bbox[0], bbox[3] - bbox[1]
            except AttributeError:
                text_w, text_h = font.getsize(label)
            text_x = max_width + (max_width - text_w) // 2
            text_y = y + max_height + 2
            draw.text((text_x, text_y), label, fill=(0,0,0), font=font)

    grid_img.save(output_file)
    print(f"Saved combined image as {output_file}")

if __name__ == '__main__':
    import sys
    if len(sys.argv) < 2:
        print("Usage: python combine_images_with_labels.py <directory> [output_file] [font_size]")
    else:
        directory = sys.argv[1]
        output_file = sys.argv[2] if len(sys.argv) > 2 else 'combined.png'
        font_size = int(sys.argv[3]) if len(sys.argv) > 3 else 56
        combine_pairs_with_labels(directory, output_file, font_size) 