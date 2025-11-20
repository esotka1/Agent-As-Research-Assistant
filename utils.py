import PyPDF2
import fitz  # PyMuPDF
import pandas as pd
from pathlib import Path
from PIL import Image
import os
import shutil
import io
    
def read_paper_content(file_path: str) -> str:
    """Read the content of the paper"""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")

    # If the paper is in PDF format
    if file_path.endswith(".pdf"):
        paper_content = ""
        with open(file_path, "rb") as pdf_file:
            reader = PyPDF2.PdfReader(pdf_file)
            for page in reader.pages:
                paper_content += page.extract_text()
        return paper_content.strip()

    # If the paper is in TXT format
    elif file_path.endswith(".txt"):
        with open(file_path, "r", encoding="utf-8") as text_file:
            return text_file.read().strip()

    # Unsupported file type
    else:
        raise ValueError("Unsupported file type. Please provide a .pdf or .txt file.")
    
'''
def extract_images(pdf_path, 
                   output_dir=Path("resources/output/figures"),
                   input_dir=Path("resources/input/figures"),
                   min_area=20000, 
                   aspect_ratio_threshold=(0.3, 3.0)):

    # Reset directories
    for directory in [output_dir, input_dir]:
        if directory.exists():
            shutil.rmtree(directory)
        os.makedirs(directory, exist_ok=True)

    doc = fitz.open(pdf_path)
    figure_count = 0

    for page_index, page in enumerate(doc, start=1):
        text = page.get_text("text")
        possible_figure_labels = {"figure", "fig."}
        has_figure_label = any(lbl in text.lower() for lbl in possible_figure_labels)

        for img_index, img in enumerate(page.get_images(full=True)):
            xref = img[0]
            base_image = doc.extract_image(xref)
            image_bytes = base_image["image"]

            # Load into PIL
            image = Image.open(io.BytesIO(image_bytes))
            w, h = image.size
            area = w * h
            aspect_ratio = w / h if h else 0

            # Apply heuristics
            if area < min_area:
                continue
            if not (aspect_ratio_threshold[0] <= aspect_ratio <= aspect_ratio_threshold[1]):
                continue
            if not has_figure_label:
                continue

            # Convert to RGB
            if image.mode != "RGB":
                image = image.convert("RGB")

            # Filenames
            pdf_filename = input_dir / f"page{page_index}_figure{img_index}.pdf"
            png_filename = output_dir / f"page{page_index}_figure{img_index}.png"

            # Save as PDF (to input_dir)
            image.save(pdf_filename, "PDF", resolution=100.0)

            # Save as PNG (to output_dir)
            image.save(png_filename, "PNG")

            figure_count += 1

    print(f"✅ Extracted {figure_count} likely figures as PDFs to {input_dir}/ and PNGs to {output_dir}/")
    return True
'''

def extract_images(pdf_path, 
                   output_dir=Path("resources/output/figures"),
                   min_area=20000, 
                   aspect_ratio_threshold=(0.3, 3.0)):

    # Reset directory
    if output_dir.exists():
        shutil.rmtree(output_dir)
    os.makedirs(output_dir, exist_ok=True)

    doc = fitz.open(pdf_path)
    figure_count = 0

    for page_index, page in enumerate(doc, start=1):
        text = page.get_text("text")
        possible_figure_labels = {"figure", "fig."}
        has_figure_label = any(lbl in text.lower() for lbl in possible_figure_labels)

        for img_index, img in enumerate(page.get_images(full=True)):
            xref = img[0]
            base_image = doc.extract_image(xref)
            image_bytes = base_image["image"]

            # Load into PIL
            image = Image.open(io.BytesIO(image_bytes))
            w, h = image.size
            area = w * h
            aspect_ratio = w / h if h else 0

            # Apply heuristics
            if area < min_area:
                continue
            if not (aspect_ratio_threshold[0] <= aspect_ratio <= aspect_ratio_threshold[1]):
                continue
            if not has_figure_label:
                continue

            # Convert to RGB if needed
            if image.mode != "RGB":
                image = image.convert("RGB")

            # Save only PNG
            png_filename = output_dir / f"page{page_index}_figure{img_index}.png"
            image.save(png_filename, "PNG")

            figure_count += 1

    print(f"✅ Extracted {figure_count} PNG figures to {output_dir}/")
    return True

def convert_pngs_to_pdfs(source_dir, dest_dir):

    source_dir = Path(source_dir)
    dest_dir = Path(dest_dir)

    # Reset destination directory
    if dest_dir.exists() and dest_dir.is_dir():
        shutil.rmtree(dest_dir)
    os.makedirs(dest_dir, exist_ok=True)

    # --- Case 1: source is a file ---
    if source_dir.is_file():
        # Move file directly
        try:
            shutil.move(str(source_dir), dest_dir / source_dir.name)
            print(f"📦 Moved file {source_dir.name} to {dest_dir}/")
            return 1
        except Exception as e:
            print(f"⚠️ Could not move file {source_dir}: {e}")
            return 0

    # --- Case 2: source is a directory (original PNG→PDF behavior) ---
    converted_count = 0

    for png_path in source_dir.glob("*.png"):
        try:
            image = Image.open(png_path)

            # Convert to RGB (important for PDF saving)
            if image.mode != "RGB":
                image = image.convert("RGB")

            pdf_filename = dest_dir / f"{png_path.stem}.pdf"
            image.save(pdf_filename, "PDF", resolution=100.0)
            converted_count += 1
        except Exception as e:
            print(f"⚠️ Skipping {png_path.name}: {e}")

    print(f"✅ Converted {converted_count} PNG files to PDFs in {dest_dir}/")
    return converted_count

'''
def convert_pngs_to_pdfs(source_dir, dest_dir):

    # Ensure destination directory exists
    if dest_dir.exists():
        shutil.rmtree(dest_dir)
    os.makedirs(dest_dir, exist_ok=True)

    # Track conversions
    converted_count = 0

    for png_path in source_dir.glob("*.png"):
        try:
            image = Image.open(png_path)

            # Convert to RGB (important for PDF saving)
            if image.mode != "RGB":
                image = image.convert("RGB")

            pdf_filename = dest_dir / f"{png_path.stem}.pdf"
            image.save(pdf_filename, "PDF", resolution=100.0)
            converted_count += 1
        except Exception as e:
            print(f"⚠️ Skipping {png_path.name}: {e}")

    print(f"✅ Converted {converted_count} PNG files to PDFs in {dest_dir}/")
    return converted_count
'''

def summarize_csv(path, sample_rows=10):
    df = pd.read_csv(path)
    summary = {
        "columns": list(df.columns),
        "dtypes": df.dtypes.astype(str).to_dict(),
        "num_rows": len(df),
        "sample": df.head(sample_rows).to_dict(orient="records")
    }
    return summary

def clear_non_folder_items(path: str) -> bool:
    try:
        for item in os.listdir(path):
            full_path = os.path.join(path, item)

            # Only delete files – skip directories
            if os.path.isfile(full_path):
                os.remove(full_path)

        return True
    except Exception as e:
        print(f"Error while clearing folder: {e}")
        return False