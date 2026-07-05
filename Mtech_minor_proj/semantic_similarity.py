import torch
from sentence_transformers import SentenceTransformer, util

def run_standalone_test():
    print("Loading lightweight vector encoder...")
    model = SentenceTransformer('all-mpnet-base-v2')
    
    # Standalone demo sentences
    source_sentence = "Deep learning architectures significantly optimize feature extraction metrics."
    student_sentence = "Neural network models greatly improve performance in extracting features."
    
    # Compute vector embeddings
    emb1 = model.encode(source_sentence, convert_to_tensor=True)
    emb2 = model.encode(student_sentence, convert_to_tensor=True)
    
    # Calculate geometric cosine alignment
    similarity = util.cos_sim(emb1, emb2).item()
    
    print("\n--- Standalone Verification Test ---")
    print(f"Source Line:  {source_sentence}")
    print(f"Student Line: {student_sentence}")
    print(f"Calculated Semantic Math Score: {round(similarity, 4)} ({round(similarity * 100, 2)}% Match)")

if __name__ == "__main__":
    run_standalone_test()