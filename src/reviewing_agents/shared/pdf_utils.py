import io

import fitz
from PIL import Image


def pdf_first_page_to_image(pdf_path: str) -> Image.Image:
    doc = fitz.open(pdf_path)
    page = doc[0]
    pix = page.get_pixmap()
    img_data = pix.tobytes("ppm")
    doc.close()
    return Image.open(io.BytesIO(img_data))
