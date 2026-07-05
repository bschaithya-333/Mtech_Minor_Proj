# Semantic Plagiarism Detection Framework using Deep Learning

An MTech Minor Project focusing on identifying smart, structural paraphrasing in textual PDF submissions using dense vector space sentence alignments.

## 🚀 Project Development & Evolution Timeline
- **Phase 1: Baseline Architecture** Established a baseline configuration utilizing lexical string matching (TF-IDF and Jaccard-style frequencies). Quantified structural failure modes against heavy synonym swapping and active-passive voice shifting.
- **Phase 2: Semantic Core Prototyping** Engineered a tokenized chunk-level alignment engine utilizing lightweight transformer architectures (`all-MiniLM-L6-v2`) to capture context over raw syntax.
- **Phase 3: Interactive Interface Deployment** Designed and deployed a graphical web client dashboard using Gradio, allowing real-time PDF native uploads and automated textual comparative rendering.
- **Phase 4: Full System Optimization** Upgraded the backend engine to a state-of-the-art 768-dimensional transformer backbone (`all-mpnet-base-v2`) to eliminate structural alignment biases and optimize long-context extraction accuracy.

## 🛠️ Architecture & Pipeline
1. **PDF Text Extraction:** Reads multi-page documents directly using a decoupled `pypdf` parsing engine.
2. **Text Segmentation:** Extracted text is cleaned and segmented into logical, individual sentence chunks via unsupervised `NLTK` tokenizers.
3. **Dense Vector Embedding:** Tokenized clauses are mapped into a high-dimensional vector space using a pre-trained Siamese-BERT model network (`MPNet`).
4. **Matrix Alignment:** A vectorized cross-document Cosine Similarity metric computes all-to-all spatial distances to identify hidden conceptual matches.
5. **Classification:** Sentences crossing the user-defined dynamic sensitivity slider threshold are immediately flagged for manual structural auditing.

## 📦 Installation & Setup

1. Open your terminal and navigate to the project directory:
   ```bash
   cd Mtech_minor_proj

2. Install all core machine learning, NLP, and UI interface dependencies:
   bash
   pip install numpy nltk gradio sentence-transformers pypdf

3. Open your web browser and navigate to the local hosting interface URL (e.g.,http://127.0.0.1:7860) to begin running PDF document scans.
