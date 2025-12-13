import PyPDF2
import fitz  # PyMuPDF
import os
from utils import read_paper_content, extract_images

def render_pages(pdf_path, output_dir="page_images", dpi=300):
    os.makedirs(output_dir, exist_ok=True)
    doc = fitz.open(pdf_path)
    for i, page in enumerate(doc, start=1):
        pix = page.get_pixmap(dpi=dpi)
        out_path = os.path.join(output_dir, f"page_{i}.png")
        pix.save(out_path)
        print(f"Rendered {out_path}")

'''
render_pages("resources/input/fertility_decline.pdf")
'''

extract_images("resources/input/aksoy-et-al-2023-knowledge-about-federal-employment-nondiscrimination-protections-on-the-basis-of-sexual-orientation.pdf")


#extract_images("resources/input/jayachandran-2016-fertility-decline-and-missing-women (1).pdf")
'''
extract_images("resources/input/fertility_decline.pdf")
'''

# Run each figure seperate or tell how many expected figures
