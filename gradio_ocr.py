import gradio as gr
from PIL import Image, ImageFilter
import pytesseract

from panel_ocr import process_comic, summarize_story

pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

def ocr_image(image):
    if image is None:
        return ""

    image = image.resize((image.width * 2, image.height * 2))
    image = image.convert("L")
    image = image.filter(ImageFilter.MedianFilter())
    image = image.point(lambda x: 0 if x < 160 else 255)

    text = pytesseract.image_to_string(image, config="--psm 6")
    return text

with gr.Blocks() as demo:
    gr.Markdown("# Comic Panel OCR & Story Understanding")

    image_input = gr.Image(type="pil", label="Upload comic page")
    extract_btn = gr.Button("Detect Panels & OCR")

    panel_gallery = gr.Gallery(label="Detected Panels", columns=4)
    ocr_output = gr.Textbox(label="Panel-wise OCR Text", lines=12)

    summarize_btn = gr.Button("Summarize Story")
    story_output = gr.Textbox(label="Story Summary", lines=6)

    extract_btn.click(
        fn=process_comic,
        inputs=image_input,
        outputs=[panel_gallery, ocr_output]
    )

    summarize_btn.click(
        fn=summarize_story,
        inputs=ocr_output,
        outputs=story_output
    )

demo.launch()


demo.launch()
