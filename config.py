# Number of top results to return from the search
top_k = 5

# The alpha coefficient controls how much weight to put on the initial visual retrieval score 
# vs the VLM reranking score when fusing them together for the final ranking.
reranking_fusion_alpha = 0.3

# Video processing configs
fps = 1.0
max_chunk_duration = 10.0
chunk_overlap = 2.0
scene_content_threshold = 27.0

# Model names
embedding_model_name = "google/siglip-base-patch16-224"
# embedding_model_name = "openai/clip-vit-base-patch32"
vlm_model_name = "Qwen/Qwen2-VL-7B-Instruct"

# Paths
processed_frames_dir = "./data/processed_frames"
faiss_index_path = "./data/faiss_index.bin"
raw_videos_dir = "./data/raw_videos"
video_file_extension = "mp4"

# Output WebP configs
output_webp_size = (270, 480)
webp_save_dir = "./assets"