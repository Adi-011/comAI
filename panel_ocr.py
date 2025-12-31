import gradio as gr
import cv2
import numpy as np
from PIL import Image, ImageFilter
import pytesseract
import subprocess

def summarize_story(panel_text):
    if not panel_text or not panel_text.strip():
        return "No text detected to summarize."

    prompt = f"""
You are given OCR text extracted from a comic book, panel by panel, in reading order.

Your task:
- Understand the story
- Fix OCR mistakes if needed
- Produce a concise story summary (3–5 sentences)

Comic Panels:
{panel_text}

Story Summary:
"""

    try:
        result = subprocess.run(
            ["ollama", "run", "gemma:2b"],
            input=prompt,
            text=True,
            capture_output=True,
            timeout=60,          
            check=True           
        )
        return result.stdout.strip()

    except subprocess.TimeoutExpired:
        return "LLM timed out. Try a smaller model or shorter text."

    except subprocess.CalledProcessError as e:
        return f"Ollama error:\n{e.stderr}"


# Windows path (adjust if needed)
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

def detect_panels(pil_img):
    img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Blur + edges
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blur, 50, 150)

    # Close gaps between panel borders
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
    closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(
        closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    panels = []
    h, w = gray.shape

    for cnt in contours:
        x, y, pw, ph = cv2.boundingRect(cnt)
        area = pw * ph

        # Filter small boxes (speech bubbles, noise)
        if area > 0.1 * w * h:
            panel = pil_img.crop((x, y, x + pw, y + ph))
            panels.append((x, y, panel))

    # Sort panels: top-to-bottom, left-to-right
    panels.sort(key=lambda p: (p[1], p[0]))

    return [p[2] for p in panels]


def ocr_panel(panel):
    panel = panel.resize((panel.width * 2, panel.height * 2))
    panel = panel.convert("L")
    panel = panel.filter(ImageFilter.MedianFilter())
    panel = panel.point(lambda x: 0 if x < 160 else 255)

    return pytesseract.image_to_string(panel, config="--psm 6")


def process_comic(image):
    if image is None:
        return [], ""

    panels = detect_panels(image)

    texts = []
    for i, panel in enumerate(panels):
        text = ocr_panel(panel)
        texts.append(f"Panel {i+1}:\n{text.strip()}")

    return panels, "\n\n---\n\n".join(texts)


with gr.Blocks() as demo:
    gr.Markdown("# 📖 Comic Panel OCR")

    image_input = gr.Image(type="pil", label="Upload comic page")
    extract_btn = gr.Button("Detect Panels & OCR")

    panel_gallery = gr.Gallery(label="Detected Panels", columns=4)
    text_output = gr.Textbox(label="Panel-wise OCR Text", lines=12)

    extract_btn.click(
        fn=process_comic,
        inputs=image_input,
        outputs=[panel_gallery, text_output]
    )

demo.launch()
