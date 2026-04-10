import os
import json
import tempfile
import faiss
import numpy as np
from typing import List, Dict, Any

class VectorStore:
    def __init__(self, metadata_path: str, vector_dim: int = 768):
        """
        Initializes the Faiss index and loads the pre-computed timestamps.
        
        Args:
            metadata_path: Path to the metadata.json file created by VideoProcessor.
            vector_dim: The dimension of the embeddings.
        """

        # Initialize the Faiss index
        self.vector_dim = vector_dim
        self.index = faiss.IndexFlatIP(self.vector_dim)
        
        # Initialize a mapping from faiss integer ID to a dictionary: {"video_id": "vid1", "chunk_id": "chunk_00000"}
        self.id_to_chunk_map = {}
        self.current_id = 0
        
        # Load the timestamp metadata
        print(f"Loading metadata from {metadata_path}...")
        with open(metadata_path, 'r') as f:
            self.chunk_timestamps = json.load(f)

    def add_embeddings(self, embedding_records: List[Dict[str, Any]]):
        """
        Adds frame vectors to the faiss index and stores their metadata mapping.
        
        Args:
            embedding_records: A list of dictionaries: [{"video_id": "vid1", "chunk_id": "chunk_00000", "vector": [...]}, ...]
        """

        # Guard for empty input
        if not embedding_records:
            print("No embeddings to add.")
            return

        # Extract vectors into a single contiguous numpy array
        vectors = np.vstack([record["vector"] for record in embedding_records]).astype(np.float32)
        
        # Add to faiss
        self.index.add(vectors)
        
        # Map the internal faiss IDs to the video/chunk metadata
        for record in embedding_records:
            self.id_to_chunk_map[self.current_id] = {
                "video_id": record["video_id"],
                "chunk_id": record["chunk_id"]
            }
            self.current_id += 1
            
        print(f"Successfully added {len(embedding_records)} frame vectors to Faiss. Total index size: {self.index.ntotal}")
    
    def save_index(self, index_path: str):
        """Saves the Faiss index and the ID mapping to disk.

        Args:
            index_path: The file path to save the Faiss index. 
            Based on this path, the faiss-ID-to-metadata mapping will be saved as {index_path}.map.json.
        """

        # Save the faiss index to disk
        faiss.write_index(self.index, index_path)
        
        # Save the faiss-ID-to-metadata mapping too
        map_path = f"{index_path}.map.json"
        with open(map_path, 'w') as f:
            json.dump(self.id_to_chunk_map, f)
        print(f"Saved Faiss index to {index_path} and mapping to {map_path}")

    def load_index(self, index_path: str):
        """Loads a pre-computed faiss index and faiss-ID-to-metadata mapping from disk.
        
        Args:
            index_path: The file path to the saved faiss index.
        """

        # Load the faiss index from disk
        self.index = faiss.read_index(index_path)
        
        # Load the faiss-ID-to-metadata mapping
        map_path = f"{index_path}.map.json"
        with open(map_path, 'r') as f:
            loaded_map = json.load(f)
            # String keys are converted back to integers for faiss lookup
            self.id_to_chunk_map = {int(k): v for k, v in loaded_map.items()}
            
        self.current_id = len(self.id_to_chunk_map)
        print(f"Loaded Faiss index with {self.index.ntotal} vectors.")

    def search_and_pool(self, query_vector: np.ndarray, target_k_chunks: int = 15) -> List[Dict[str, Any]]:
        """
        Searches for the query and applies Late Score Max Pooling to return unique chunks.
        
        Args:
            query_vector: The 1D normalized numpy array from the text query.
            target_k_chunks: The number of unique video chunks to return.
        
        Returns:
            A list of dictionaries: [{"video_id": "vid1", "chunk_id": "chunk_00000", "score": 0.95, "start_sec": 10.0, "end_sec": 20.0}, ...]
        """

        # Ensure query is float32 and shape (1, dim)
        query_vector = query_vector.astype(np.float32).reshape(1, -1)
        
        # Retrieve a large pool of frames because a single video chunk may contain multiple top-scoring frames
        search_depth = target_k_chunks * 5 
        
        # Perform the search
        faiss_scores, faiss_ids = self.index.search(query_vector, search_depth)
        
        chunk_scores = {}
        
        # Iterate through the retrieved frames
        for rank in range(search_depth):
            faiss_id = faiss_ids[0][rank]
            score = faiss_scores[0][rank]
            
            # If faiss returns -1, it means we've exhausted the index
            if faiss_id == -1:
                break
                
            # Retrieve the corresponding video and chunk metadata
            chunk_info = self.id_to_chunk_map[faiss_id]

            # Create a unique string identifier for the chunk
            chunk_key = f"{chunk_info['video_id']}||{chunk_info['chunk_id']}"
            
            # Late score max pooling by keeping the highest score for each unique chunks
            if chunk_key not in chunk_scores or score > chunk_scores[chunk_key]["score"]:
                chunk_scores[chunk_key] = {
                    "video_id": chunk_info['video_id'],
                    "chunk_id": chunk_info['chunk_id'],
                    "score": float(score)
                }
                
        # Sort the unique chunks by their max score in descending order
        sorted_chunks = sorted(chunk_scores.values(), key=lambda x: x["score"], reverse=True)
        
        # Keep only the top K unique chunks
        top_k_results = sorted_chunks[:target_k_chunks]

        # Add timestamps from the metadata to the results
        return self._format_with_timestamps(top_k_results)

    def _format_with_timestamps(self, results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Adds start and end timestamps to the search results.

        Args:
            results: A list of dictionaries with "video_id", "chunk_id", and "score" keys.

        Returns:
            A list of dictionaries enriched with "start_sec" and "end_sec" from the metadata.
        """

        enriched_results = []
        for result in results:
            video_id = result["video_id"]
            chunk_id = result["chunk_id"]
            
            # Lookup the timestamps from the metadata using video_id and chunk_id
            try:
                times = self.chunk_timestamps[video_id][chunk_id]
                result["start_sec"] = times["start_sec"]
                result["end_sec"] = times["end_sec"]
            except KeyError:
                # Fallback just in case metadata is malformed
                result["start_sec"] = 0.0
                result["end_sec"] = 0.0
                
            enriched_results.append(result)
            
        return enriched_results

if __name__ == "__main__":
    # Dummy example execution   
    print("Running vector store test...")
    
    # Create a temporary mock metadata JSON
    mock_metadata = {
        "video_001": {
            "chunk_00000": {"start_sec": 0.0, "end_sec": 10.0},
            "chunk_00001": {"start_sec": 10.0, "end_sec": 20.0}
        }
    }
    
    # Write the mock metadata to a temporary file
    with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.json') as f:
        json.dump(mock_metadata, f)
        temp_meta_path = f.name

    try:
        # Initialize the store
        store = VectorStore(metadata_path=temp_meta_path, vector_dim=768)
        
        # Generate mock normalized frame vectors
        mock_records = []
        for i in range(5):
            # 5 frames for chunk_00000
            vec1 = np.random.rand(768).astype(np.float32)
            vec1 = vec1 / np.linalg.norm(vec1)
            mock_records.append({"video_id": "video_001", "chunk_id": "chunk_00000", "vector": vec1})
            
            # 5 frames for chunk_00001
            vec2 = np.random.rand(768).astype(np.float32)
            vec2 = vec2 / np.linalg.norm(vec2)
            mock_records.append({"video_id": "video_001", "chunk_id": "chunk_00001", "vector": vec2})

        # Add mock embeddings to the store
        print("\nAdding mock embeddings to Faiss...")
        store.add_embeddings(mock_records)
        
        # Create a mock search query
        mock_query = np.random.rand(768).astype(np.float32)
        mock_query = mock_query / np.linalg.norm(mock_query)
        
        # Search and pool results
        print("\nSearching and applying Late Score Max Pooling...")
        results = store.search_and_pool(mock_query, target_k_chunks=2) # len(results) should be 2 unique chunks
        
        # Print the search results with timestamps
        print("\nSearch Results:")
        for res in results:
            print(f"Video: {res['video_id']}, Chunk: {res['chunk_id']}, "
                  f"Duration: {res['start_sec']}s - {res['end_sec']}s, Max score: {res['score']:.4f}")
                  
    finally:
        # Clean up the temporary test file
        os.remove(temp_meta_path)