import os
import base64
import json
from io import BytesIO
from pathlib import Path

from dotenv import load_dotenv
import streamlit as st
from groq import Groq
from pdf2image import convert_from_bytes

# -------------------------------------------------------
# Load environment and initialize Groq client
# -------------------------------------------------------
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
client = Groq(api_key=GROQ_API_KEY)

st.title("🔎 LLaMA-4 Scout Vision OCR (via Groq)")
st.caption("Upload images, PDFs, text files, or Jupyter notebooks (.ipynb)")

uploaded = st.file_uploader("Upload any file", type=None)
prompt = st.text_input("Prompt (e.g., 'Extract all text from this file')")

# -------------------------------------------------------
# Helper: send image to Groq
# -------------------------------------------------------
def process_image(image_bytes: bytes, user_prompt: str) -> str:
    image_b64 = base64.b64encode(image_bytes).decode("utf-8")
    resp = client.chat.completions.create(
        model="meta-llama/Llama-4-Maverick-17B-128E-Instruct",
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": user_prompt},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_b64}"}}
                ],
            }
        ],
        max_tokens=500,
    )
    return resp.choices[0].message.content


# -------------------------------------------------------
# Run button
# -------------------------------------------------------
if st.button("Run"):
    if not uploaded or not prompt.strip():
        st.error("Please upload a file and enter a prompt.")
    else:
        with st.spinner("Processing..."):
            name = uploaded.name.lower()
            suffix = Path(name).suffix
            results = []

            # ---------- PDFs ----------
            if suffix == ".pdf":
                pages = convert_from_bytes(uploaded.read())
                for i, page in enumerate(pages, start=1):
                    buf = BytesIO()
                    page.save(buf, format="JPEG")
                    text = process_image(buf.getvalue(), f"{prompt} (Page {i})")
                    results.append(f"### Page {i}\n{text}")

            # ---------- Images ----------
            elif suffix in [".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"]:
                text = process_image(uploaded.read(), prompt)
                results.append(text)

            # ---------- Text-like (txt, csv, md, json, py, html, log, ipynb) ----------
            elif suffix in [".txt", ".csv", ".md", ".json", ".py",
                            ".html", ".log", ".ipynb"]:
                try:
                    raw_bytes = uploaded.read()
                    if suffix == ".ipynb":
                        nb = json.loads(raw_bytes.decode("utf-8", errors="ignore"))
                        cells_text = []
                        for cell in nb.get("cells", []):
                            if cell.get("cell_type") in {"code", "markdown"}:
                                cells_text.append("".join(cell.get("source", [])))
                        text_data = "\n\n".join(cells_text)
                    else:
                        text_data = raw_bytes.decode("utf-8", errors="ignore")

                    resp = client.chat.completions.create(
                        model="meta-llama/Llama-4-Maverick-17B-128E-Instruct",
                        messages=[{"role": "user", "content": f"{prompt}\n\n{text_data}"}],
                        max_tokens=800,
                    )
                    results.append(resp.choices[0].message.content)

                except Exception as e:
                    results.append(f"⚠️ Could not decode text: {e}")

            # ---------- Unsupported ----------
            else:
                results.append(
                    f"⚠️ The file type `{suffix}` isn't supported directly.\n"
                    f"Convert it to PDF, image, or text first."
                )

            st.subheader("Extracted Output:")
            st.markdown("\n\n".join(results))
