"""
Code to make GIFs from all the output prediction plots.

Edit as necessary.
"""
import os

def create_gif(image_folder, output_gif, duration=0.15):
    images = []
    for year in range(2025, 2101):
        for month in range(1, 13):
            image_path = os.path.join(image_folder, f"preds_mri_north_ssp126_{year}_{month:02d}.png")
            if os.path.exists(image_path):
                img = Image.open(image_path).convert("RGBA")
                
                # Create a white background image
                bg = Image.new("RGBA", img.size, (255, 255, 255, 255))
                img = Image.alpha_composite(bg, img).convert("RGB")
                
                images.append(img)
    
    if images:
        images[0].save(output_gif, save_all=True, append_images=images[1:], duration=int(duration * 1000), loop=0)
        print(f"GIF saved as {output_gif}")
    else:
        print("No images found to create GIF.")

image_folder = "/home/users/clelland/Model/Final_plots/North v2 no FWI/MRI-ESM2-0/SSP126"
output_gif = "/home/users/clelland/Model/Final_plots/GIFs v2 no FWI/north_mri_ssp126_v2_no_fwi.gif"
create_gif(image_folder, output_gif)