import numpy as np
import nltk
from sentence_transformers import SentenceTransformer

# Download the sentence tokenizer if not already present
try:
    nltk.data.find('tokenizers/punkt_tab')
except LookupError:
    nltk.download('punkt_tab', quiet=True)

def calculate_cosine_similarity(vec1, vec2):
    return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

def run_hybrid_pipeline():
    print("=" * 60)
    print("INITIALIZING WEEK 3 HYBRID PARAGRAPH PIPELINE...")
    print("=" * 60)
    
    model = SentenceTransformer('all-MiniLM-L6-v2')
    
    # Simulating a real student submission paragraph vs a source document
    source_document = (
        "Deep learning models require a massive amount of computational power. "
        "Engineers typically utilize high-end graphics cards to accelerate neural network training. "
        "Without proper hardware optimization, training complex architectures can take several weeks."
    )
    
    # Highly paraphrased submission trying to cheat the system
    student_submission = (
        "Advanced neural architectures demand substantial processing resources. "
        "Software developers generally deploy powerful GPUs to speed up model optimization. "
        "A completely unrelated sentence about drinking coffee on a rainy afternoon."
    )
    
    print("\n[Step 1] Tokenizing paragraphs into individual sentences...")
    source_sentences = nltk.sent_tokenize(source_document)
    student_sentences = nltk.sent_tokenize(student_submission)
    
    print(f"  -> Source Sentences Found: {len(source_sentences)}")
    print(f"  -> Submission Sentences Found: {len(student_sentences)}")
    
    print("\n[Step 2] Generating semantic embeddings...")
    source_embeddings = model.encode(source_sentences)
    student_embeddings = model.encode(student_sentences)
    
    print("\n[Step 3] Running Matrix Alignment Check...")
    print("#" * 50)
    print("            WEEK 3 DETAILED REPORT            ")
    print("#" * 50)
    
    plagiarism_count = 0
    THRESHOLD = 0.50  # Based on your Week 2 observation!
    
    for i, stud_sent in enumerate(student_sentences):
        max_score = 0.0
        best_match = ""
        
        # Check this submission sentence against ALL sentences in the source document
        for j, src_sent in enumerate(source_sentences):
            score = calculate_cosine_similarity(student_embeddings[i], source_embeddings[j])
            if score > max_score:
                max_score = score
                best_match = src_sent
        
        print(f"\nAnalyzing Submission Sentence {i+1}:")
        print(f"  Text: \"{stud_sent}\"")
        print(f"  Best Source Match: \"{best_match}\"")
        print(f"  --> Similarity Confidence: {max_score:.4f}")
        
        if max_score >= THRESHOLD:
            print("  🚨 ALERT: Likely Paraphrased Plagiarism Detected!")
            plagiarism_count += 1
        else:
            print("  ✅ CLEAN: Sentence appears original or unrelated.")
            
    overall_plagiarism_ratio = (plagiarism_count / len(student_sentences)) * 100
    print("\n" + "=" * 60)
    print(f"FINAL PIPELINE SUMMARY:")
    print(f"Total Sentences flagged as plagiarized: {plagiarism_count}/{len(student_sentences)}")
    print(f"Overall Paragraph Plagiarism Risk: {overall_plagiarism_ratio:.1f}%")
    print("=" * 60)

run_hybrid_pipeline()