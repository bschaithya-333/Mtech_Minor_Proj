import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import nltk

# Ensure tokenizers are downloaded
nltk.download('punkt', quiet=True)

class TFIDFPlagiarismBaseline:
    def __init__(self):
        self.vectorizer = TfidfVectorizer()

    def preprocess_to_sentences(self, text: str):
        """Splits raw document text into clean, individual sentences."""
        sentences = nltk.sent_tokenize(text)
        return [s.strip() for s in sentences if len(s.strip()) > 5]

    def compute_similarity_matrix(self, doc1_sents: list, doc2_sents: list):
        """Computes TF-IDF cosine similarity matrix between two sets of sentences."""
        all_sentences = doc1_sents + doc2_sents
        
        # Fit and transform the combined vocabulary
        tfidf_matrix = self.vectorizer.fit_transform(all_sentences)
        
        # Split matrices back to original documents
        matrix_doc1 = tfidf_matrix[:len(doc1_sents)]
        matrix_doc2 = tfidf_matrix[len(doc1_sents):]
        
        # Calculate pairwise similarity
        return cosine_similarity(matrix_doc1, matrix_doc2)

# ==========================================
# TEST DEMO (The exact case for your mentor)
# ==========================================
if __name__ == "__main__":
    doc1_text = "Machine learning is a subset of artificial intelligence that enables systems to learn from data. The dog chased the cat down the street."
    doc2_text = "ML is a branch of AI that allows computers to improve through experience without being explicitly programmed. A hound pursued the feline along the road."

    baseline = TFIDFPlagiarismBaseline()
    
    sents1 = baseline.preprocess_to_sentences(doc1_text)
    sents2 = baseline.preprocess_to_sentences(doc2_text)
    
    sim_matrix = baseline.compute_similarity_matrix(sents1, sents2)
    
    print("--- Week 1 Baseline TF-IDF Matrix Results ---")
    for i, s1 in enumerate(sents1):
        for j, s2 in enumerate(sents2):
            print(f"\nDoc 1: '{s1}'")
            print(f"Doc 2: '{s2}'")
            print(f"TF-IDF Similarity Score: {sim_matrix[i][j]:.4f}")