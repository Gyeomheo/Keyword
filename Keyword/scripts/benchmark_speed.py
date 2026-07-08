import time
import torch
import numpy as np
import sys
import os

# Ensure src can be imported
sys.path.append(os.getcwd())

try:
    from src.inference import SBERTVectorizer
    
    def benchmark():
        with open("benchmark_progress.txt", "w") as f:
            f.write("Starting...\n")
            
        vec = SBERTVectorizer()
        
        with open("benchmark_progress.txt", "a") as f:
            f.write("Model Loaded.\n")
            
        # Generate 1000 dummy keywords
        texts = ["samsung galaxy s24 ultra price"] * 1000
        
        # Warm up
        _ = vec.encode(texts[:10])
        
        with open("benchmark_progress.txt", "a") as f:
            f.write("Warm up done.\n")
            
        start = time.time()
        vectors = vec.encode(texts, batch_size=128)
        elapsed = time.time() - start
        
        res = f"Elapsed Time: {elapsed:.4f}s\nSpeed: {len(texts)/elapsed:.2f} keywords/sec\n"
        with open("benchmark_result.txt", "w") as f:
            f.write(res)
            
    if __name__ == "__main__":
        benchmark()
except Exception as e:
    with open("benchmark_error.txt", "w") as f:
        f.write(str(e))
        import traceback
        f.write("\n")
        f.write(traceback.format_exc())
