import numpy as np
import nltk
import gradio as gr
from sentence_transformers import SentenceTransformer
from pypdf import PdfReader

# Ensure sentence tokenizer is ready locally
try:
    nltk.data.find('tokenizers/punkt_tab')
except LookupError:
    nltk.download('punkt_tab', quiet=True)

print("Loading Transformer Core for UI...")
model = SentenceTransformer('all-mpnet-base-v2')

def calculate_cosine_similarity(vec1, vec2):
    return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

def extract_text_from_pdf(pdf_file):
    """Reads text from an uploaded file object path"""
    if pdf_file is None:
        return ""
    try:
        reader = PdfReader(pdf_file.name)
        text = ""
        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
        return text
    except Exception as e:
        return f"Error reading PDF: {str(e)}"

def run_ui_scan(source_pdf, student_pdf, threshold_val):
    # Pass the uploaded file structures into our background reader
    source_text = extract_text_from_pdf(source_pdf)
    student_text = extract_text_from_pdf(student_pdf)
    
    if not source_text.strip() or not student_text.strip():
        return "Please upload both the Original Source PDF and the Student Submission PDF."
    
    decimal_threshold = threshold_val / 100.0
    
    source_sentences = nltk.sent_tokenize(source_text)
    student_sentences = nltk.sent_tokenize(student_text)
    
    source_embeddings = model.encode(source_sentences)
    student_embeddings = model.encode(student_sentences)
    
    plagiarism_count = 0
    flags_output = []
    
    for i, stud_sent in enumerate(student_sentences):
        if len(stud_sent.strip()) < 15:
            continue
            
        max_score = 0.0
        best_match = ""
        
        for j, src_sent in enumerate(source_sentences):
            score = calculate_cosine_similarity(student_embeddings[i], source_embeddings[j])
            if score > max_score:
                max_score = score
                best_match = src_sent
                
        if max_score >= decimal_threshold:
            plagiarism_count += 1
            flags_output.append(
                f"### 🚨 FLAG ({int(max_score * 100)}% Match):\n"
                f'>>> "{stud_sent.strip()}"\n\n'
                f"**Closest Source Line from PDF:**\n"
                f'*{best_match.strip()}*\n'
            )
        else:
            flags_output.append(
                f"### ✅ CLEAN ({int(max_score * 100)}% Match):\n"
                f'>>> "{stud_sent.strip()}"\n'
            )
            
    if not flags_output:
        return "## ✅ No readable academic sentences processed from the PDFs."
        
    risk_pct = int((plagiarism_count / len(flags_output)) * 100)
    
    if risk_pct >= 70:
        header = f"## 🚨 Critical Risk Level: {risk_pct}% Match Detected in PDF Analysis\n\n---"
    elif risk_pct >= 40:
        header = f"## ⚠️ Moderate Risk Level: {risk_pct}% Match Detected in PDF Analysis\n\n---"
    else:
        header = f"## ✅ Low Risk / Clean Document: {risk_pct}% Match\n\n---"
        
    return header + "\n" + "\n".join(flags_output)

# Building the Presentation Web UI Dashboard for complete PDF Verification
with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown(
        """
        # 🛡️ Deep Semantic PDF Plagiarism Detection Workspace
        ### MTech Minor Project Presentation Interface | Powered by 768-D MPNet Embeddings
        """
    )
    
    with gr.Row():
        with gr.Column(scale=1):
            # These two lines build the interactive file upload drag-and-drop slots
            source_input = gr.File(label="Upload Original Source PDF (Reference Paper)", file_types=[".pdf"])
            student_input = gr.File(label="Upload Student Submission PDF (Target Project)", file_types=[".pdf"])
            
            threshold_slider = gr.Slider(
                minimum=10, 
                maximum=90, 
                value=50, 
                step=5, 
                label="AI Alignment Sensitivity Threshold (%)"
            )
            submit_btn = gr.Button("🚀 Run Complete PDF Semantic Scan", variant="primary")
            
        with gr.Column(scale=1):
            output_report = gr.Markdown(label="Real-time PDF Structural Alignment Report")

    submit_btn.click(
        fn=run_ui_scan,
        inputs=[source_input, student_input, threshold_slider],
        outputs=output_report
    )

if __name__ == "__main__":
    demo.launch(inbrowser=True)