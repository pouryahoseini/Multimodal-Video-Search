import os
import argparse
import time
from src.video_processor import VideoProcessor
from src.embedder import MultimodalEmbedder
from src.vector_store import VectorStore
from src.reranker import Reranker

def build_index(raw_dir: str, frames_dir: str, index_path: str, max_chunk_duration: float = 15.0, 
                overlap: float = 2.0, content_threshold: float = 27.0, fps : float = 1.0, 
                model_name: str = "google/siglip-base-patch16-224") -> (MultimodalEmbedder, VectorStore):
    """Processes videos, extracts frames, and builds the faiss index.

    Args:
        raw_dir: Directory containing raw videos.
        frames_dir: Directory to save processed frames and metadata.
        index_path: Path to save the faiss index file.
        max_chunk_duration: Maximum duration of each video chunk.
        overlap: Overlap duration in seconds between consecutive chunks.
        content_threshold: Threshold for PySceneDetect content detection.
        fps: Frames per second for video processing.
        model_name: The vision transformer model to use for embedding.

    Returns:
        A tuple of (embedder, vector_store) ready for search.
    """

    print("\n" + "="*50)
    print("PHASE 1: INGESTION & INDEXING")
    print("="*50)
    
    # Video processing (chunking and extraction)
    print("\n[1/3] Starting Video Processing...")
    processor = VideoProcessor(raw_dir=raw_dir, output_dir=frames_dir, max_chunk_duration=max_chunk_duration,
                               overlap=overlap, content_threshold=content_threshold, fps=fps)
    processor.process_all()
    
    metadata_path = os.path.join(frames_dir, "metadata.json")
    
    # Embedding and vector store initialization
    print("\n[2/3] Initializing Embedder and Vector Store...")
    embedder = MultimodalEmbedder(model_name=model_name)
    vector_store = VectorStore(metadata_path=metadata_path, vector_dim=embedder.embedding_size)
    
    # Stream embeddings into faiss
    print("\n[3/3] Streaming Vectors into Faiss...")
    embedded_records = embedder.process_directory(frames_dir, batch_size=64)
    if embedded_records:
        vector_store.add_embeddings(embedded_records)
        print(f"  -> Successfully embedded and indexed {len(embedded_records)} total frames.")
    else:
        print("  -> WARNING: No frames were found or embedded.")
    
    # Save the faiss index to disk
    print(f"\nSaving vectors to disk...")
    vector_store.save_index(index_path)
    print(f"\nIndexing Complete! Total vectors in Faiss: {vector_store.index.ntotal}")
    
    return embedder, vector_store

def load_search_models(frames_dir: str, index_path: str, fps: float = 1.0,
                       embedding_model_name: str = "google/siglip-base-patch16-224",
                       vlm_model_name: str = "Qwen/Qwen2-VL-7B-Instruct"):
    """Loads the embedding model, vector store, and reranker into memory.
    
    Args:
        frames_dir: Directory containing processed frames and metadata.
        index_path: Path to the faiss index file.
        fps: Frames per second to use for the reranker.
        embedding_model_name: The multimodal encoder model to use for embedding.
        vlm_model_name: The generative model to use for reranking.
    
    Returns:
        A tuple of (embedder, vector_store, reranker) ready for search.
    """
    
    print("\n" + "="*50)
    print("LOADING SEARCH MODELS")
    print("="*50)
    
    # Load embedding model
    print("\n[1/3] Loading Embedding Model...")
    embedder = MultimodalEmbedder(model_name=embedding_model_name)

    # Load faiss vector store
    print("\n[2/3] Loading Vector Store...")
    metadata_path = os.path.join(frames_dir, "metadata.json")
    vector_store = VectorStore(metadata_path=metadata_path, vector_dim=embedder.embedding_size)
    vector_store.load_index(index_path)
    
    # Load reranker
    print("\n[3/3] Loading Reranker Model...")
    reranker = Reranker(model_name=vlm_model_name, fps=fps)
    
    return embedder, vector_store, reranker

def run_search(query: str, embedder: MultimodalEmbedder, vector_store: VectorStore, 
               reranker: Reranker, frames_dir: str, top_k: int = 15, 
               reranking_fusion_alpha: float = 0.3):
    """Executes the two-stage retrieval pipeline.

    Args:
        query: The natural language search query.
        embedder: The multimodal embedder.
        vector_store: The faiss vector store.
        reranker: The generative reranker.
        frames_dir: Directory where processed frames and metadata are stored.
        top_k: Number of top chunks to retrieve from stage 1 for reranking.
        reranking_fusion_alpha: Weight for stage 1 retrieval score in final fusion.
            (1 - alpha) is the weight for stage 2 VLM score)
    """

    print("\n" + "="*50)
    print(f"MULTIMODAL SEARCH")
    print(f"Query: '{query}'")
    
    start_time = time.time()

    # Stage 1: High-recall spatial retrieval (SigLIP + Faiss)
    query_vector = embedder.embed_text(query)
    stage1_results = vector_store.search_and_pool(query_vector, target_k_chunks=top_k)
    
    # Stage 2: High-precision generative reranking (Qwen2-VL)
    final_results = reranker.rerank(
        query=query,
        top_k_results=stage1_results,
        frames_base_dir=frames_dir,
        alpha=reranking_fusion_alpha
    )
    
    # Display results 
    elapsed_time = time.time() - start_time
    print("-"*30)
    print(f"SEARCH RESULTS (Completed in {elapsed_time:.2f}s)")
    print("="*50)
    
    for rank, res in enumerate(final_results[:top_k]):
        print(f"\nRank {rank + 1}:")
        print(f"  Video File : {res['video_id']}.mp4")
        print(f"  Timestamps : {res['start_sec']}s - {res['end_sec']}s")
        print(f"  Confidence : {res['final_fused_score']:.3f} "
              f"(Retrieval score: {res['norm_stage1_score']:.2f}, Reranking score: {res['vlm_yes_prob']:.2f})")
    print("\n" + "="*50)

    return final_results[:top_k]

if __name__ == "__main__":
    # Command-line interface to build index, load models, or run interactive search
    parser = argparse.ArgumentParser(description="Multimodal Video Retrieval Pipeline")
    parser.add_argument("--build", action="store_true", help="Flag to build the Faiss index from scratch.")
    parser.add_argument("--load-models", action="store_true", help="Flag to load models.")
    parser.add_argument("--query", action="store_true", help="The natural language search query.")
    parser.add_argument("--raw-dir", type=str, default="./data/raw_videos", help="Directory containing raw videos.")
    parser.add_argument("--frames-dir", type=str, default="./data/processed_frames", help="Directory containing extracted frames.")
    parser.add_argument("--index-path", type=str, default="./data/faiss_index.bin", help="Path to save/load the faiss index file.")
    parser.add_argument("--top-k", type=int, default=10, help="Number of chunks to send to Stage 2.")
    parser.add_argument("--max-chunk-duration", type=float, default=10.0, help="Maximum duration of video chunks in seconds.")
    parser.add_argument("--overlap", type=float, default=2.0, help="Overlap duration in seconds between video chunks.")
    parser.add_argument("--content-threshold", type=float, default=27.0, help="PySceneDetect content detection threshold.")
    parser.add_argument("--fps", type=float, default=1.0, help="Frames per second to extract from videos for embedding.")
    parser.add_argument("--reranking-fusion-alpha", type=float, default=0.3, help="Weight for stage 1 retrieval score in final fusion.")
    parser.add_argument("--embedding-model-name", type=str, default="google/siglip-base-patch16-224", help="The multimodal embedding model.")
    parser.add_argument("--vlm-model-name", type=str, default="Qwen/Qwen2-VL-7B-Instruct", help="The VLM model for reranking.")
    
    args = parser.parse_args()

    # Default to interactive query mode if no primary action is specified
    if not args.build and not args.load_models and not args.query:
        args.query = True

    # Ensure directories exist
    os.makedirs(args.raw_dir, exist_ok=True)
    os.makedirs(args.frames_dir, exist_ok=True)

    reranker, embedder, vector_store = None, None, None

    # If requested, build the index from raw videos
    if args.build:
        build_index(raw_dir=args.raw_dir, frames_dir=args.frames_dir, 
                    index_path=args.index_path, max_chunk_duration=args.max_chunk_duration,
                    overlap=args.overlap, content_threshold=args.content_threshold, 
                    fps=args.fps, model_name=args.embedding_model_name)

    # Create a helper function to load models and build index if needed
    def load_models():
        if os.path.exists(args.index_path) and os.path.exists(args.frames_dir):
            embedder, vector_store, reranker = load_search_models(
                frames_dir=args.frames_dir, 
                index_path=args.index_path, 
                fps=args.fps,
                embedding_model_name=args.embedding_model_name, 
                vlm_model_name=args.vlm_model_name
            )
        else:
            print("Faiss index not found. Building the index...")
            embedder, vector_store = build_index(raw_dir=args.raw_dir, frames_dir=args.frames_dir, index_path=args.index_path, 
                                                 max_chunk_duration=args.max_chunk_duration, overlap=args.overlap, 
                                                 content_threshold=args.content_threshold, fps=args.fps, 
                                                 model_name=args.embedding_model_name)
            print("Building Faiss index complete. Loading reranker model...")
            reranker = Reranker(model_name=args.vlm_model_name, fps=args.fps)
            print("Reranker model loaded.")
        return embedder, vector_store, reranker
    
    # Load models if requested
    if args.load_models:
        load_models()
        
    # Enter search mode if requested
    if args.query:
        # If models are not loaded yet, load them now
        if reranker is None or embedder is None or vector_store is None:
            print("Models not loaded. Loading now...")
            embedder, vector_store, reranker = load_models()
        
        print("\n" + "="*50)
        print("INTERACTIVE SEARCH MODE")
        print("Type 'exit' or 'quit' to stop.")
        print("="*50)
        
        while True:
            try:
                # Prompt the user for a search query
                user_query = input("\nEnter search query: ").strip()

                # Exit condition
                if user_query.lower() in ['exit', 'quit']:
                    break
                if not user_query:
                    continue
                
                # Run the search
                run_search(query=user_query, embedder=embedder, vector_store=vector_store, 
                           reranker=reranker, frames_dir=args.frames_dir, top_k=args.top_k, 
                           reranking_fusion_alpha=args.reranking_fusion_alpha)
            except KeyboardInterrupt:
                print("\nExiting search mode.")
                break
