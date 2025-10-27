import fitz  # PyMuPDF

def extract_text_from_pdf(file_storage):
    """
    Extract plain text from a PDF uploaded via Flask (file_storage object).
    Returns extracted text as a string.
    """
    try:
        with fitz.open(stream=file_storage.read(), filetype="pdf") as doc:
            text = ""
            for page in doc:
                text += page.get_text("text") + "\n"
        return text.strip()
    except Exception as e:
        raise RuntimeError(f"Error reading PDF: {e}")
