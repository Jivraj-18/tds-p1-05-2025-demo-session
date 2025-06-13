# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "httpx",
#     "semantic-text-splitter",
#     "numpy",
#     "tqdm",
#   "google-genai",
# ]
# ///


import hashlib
import httpx
import json
import numpy as np
import os
import time
from pathlib import Path
from semantic_text_splitter import MarkdownSplitter
from tqdm import tqdm
from google import genai


class RateLimiter:
    def __init__(self, requests_per_minute=60, requests_per_second=2):
        self.requests_per_minute = requests_per_minute
        self.requests_per_second = requests_per_second
        self.request_times = []
        self.last_request_time = 0
    
    def wait_if_needed(self):
        current_time = time.time()
        
        # Per-second rate limiting
        time_since_last = current_time - self.last_request_time
        if time_since_last < (1.0 / self.requests_per_second):
            sleep_time = (1.0 / self.requests_per_second) - time_since_last
            time.sleep(sleep_time)
        
        # Per-minute rate limiting
        current_time = time.time()
        self.request_times = [t for t in self.request_times if current_time - t < 60]
        
        if len(self.request_times) >= self.requests_per_minute:
            sleep_time = 60 - (current_time - self.request_times[0])
            if sleep_time > 0:
                time.sleep(sleep_time)
                # Clean up old requests after sleeping
                current_time = time.time()
                self.request_times = [t for t in self.request_times if current_time - t < 60]
        
        self.request_times.append(current_time)
        self.last_request_time = current_time

rate_limiter = RateLimiter(requests_per_minute=5, requests_per_second=2)


def get_embedding(text: str, max_retries: int = 3) -> list[float]:
    """Get embedding for text chunk with rate limiting and retry logic"""
    
    client = genai.Client(api_key=os.environ.get("GENAI_API_KEY"))
    
    for attempt in range(max_retries):
        try:
            # Apply rate limiting
            rate_limiter.wait_if_needed()
            
            result = client.models.embed_content(
                model="gemini-embedding-exp-03-07",
                contents=text
            )
            
            embedding = result.embeddings[0].values
            return embedding
            
        except Exception as e:
            if "rate limit" in str(e).lower() or "quota" in str(e).lower():
                # Exponential backoff for rate limit errors
                wait_time = 2 ** attempt
                print(f"Rate limit hit, waiting {wait_time} seconds...")
                time.sleep(wait_time)
            elif attempt == max_retries - 1:
                print(f"Failed to get embedding after {max_retries} attempts: {e}")
                raise
            else:
                print(f"Attempt {attempt + 1} failed: {e}, retrying...")
                time.sleep(1)
    
    raise Exception("Max retries exceeded")



def get_chunks(file_path: str, chunk_size: int = 15000):
    with open(file_path, 'r', encoding='utf-8') as file:
        content = file.read()
    splitter = MarkdownSplitter(chunk_size)
    chunks = splitter.chunks(content)
    return chunks

if __name__ == "__main__":
    # files stores all markdown files in the "markdowns" directory
    files = [*Path("markdowns").glob("*.md"), *Path("markdowns").rglob("*.md")]
    all_chunks = []
    all_embeddings = []
    total_chunks = 0
    file_chunks = {}
    # Getting chinks from all markdown files and store in file_chunks with filepath as key and chuck as value
    for file_path in files:
        chunks = get_chunks(file_path)
        file_chunks[file_path] = chunks
        total_chunks += len(chunks)

    # for the chunks get embeddings and store them in all_embeddings 
    with tqdm(total=total_chunks, desc="Processing embeddings") as pbar:
        for file_path, chunks in file_chunks.items():
            for chunk in chunks:
                try:
                    embedding = get_embedding(chunk)
                    all_chunks.append(chunk)
                    all_embeddings.append(embedding)
                    pbar.set_postfix({"file": file_path.name, "chunks": len(all_chunks)})
                    pbar.update(1)
                except Exception as e:
                    print(f"Skipping chunk from {file_path.name} due to error: {e}")
                    pbar.update(1)
                    continue
    
    # save all the embeddings and chunks to a numpy archive file
    np.savez("embeddings.npz", chunks=all_chunks, embeddings=all_embeddings)

    file_path = ("markdowns/Contribute1.md")
    chunks = get_chunks(file_path, chunk_size=1000)
    embedding = get_embedding(chunks[0])    
    print(len(embedding))

