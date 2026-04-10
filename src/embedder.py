import os
import torch
import numpy as np
from PIL import Image
from typing import List, Dict, Union, Optional
from transformers import AutoProcessor, AutoModel
import torch.nn.functional as F
from tqdm import tqdm

class MultimodalEmbedder:
    def __init__(self, model_name: str = "google/siglip-base-patch16-224", device: Optional[str] = None):
        """
        Initializes the SigLIP model and processor for multimodal embedding.
        
        Args:
            model_name: The HuggingFace model identifier.
            device: 'cuda' or 'cpu'. Auto-detects if None.
        """
        # Determine the device to run ons
        self.device = device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Loading {model_name} onto {self.device}...")
        
        # Load the processor and model from HuggingFace
        self.processor = AutoProcessor.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(self.device)
        self.model.eval()

        # Save the embedding size
        self.embedding_size = self.model.config.text_config.hidden_size

    @torch.no_grad()
    def embed_text(self, query: str) -> np.ndarray:
        """
        Embeds a natural language query into the multimodal latent space.
        
        Args:
                query: The input text query to embed.
        
        Returns:
            A numpy array representing the embedded query vector.
        """

        # Tokenize the input text, then pass through the model to get text features
        inputs = self.processor(text=[query], padding="max_length", return_tensors="pt").to(self.device)
        text_features = self.model.get_text_features(**inputs).pooler_output

        # L2 normalize so that inner product equals cosine similarity
        text_features = F.normalize(text_features, p=2, dim=-1)

        return text_features.cpu().numpy()

    @torch.no_grad()
    def embed_images(self, image_paths: List[str], batch_size: int = 64) -> np.ndarray:
        """
        Processes a list of image paths in batches and returns their embeddings.

        Args:
            image_paths: List of file paths to the images to be embedded.
            batch_size: Number of images to process in a single batch.

        Returns:
            A 2D numpy array where each row corresponds to the embedding of an image.
        """

        # Initialize a list to hold all embeddings
        all_embeddings = []
        
        # Process images in batches to manage memory usage
        for i in tqdm(range(0, len(image_paths), batch_size), desc="Embedding Images"):
            batch_paths = image_paths[i:i + batch_size]
            
            # Load images
            images = [Image.open(p).convert("RGB") for p in batch_paths]
            
            # Process the batch of images through the model to get image features
            inputs = self.processor(images=images, return_tensors="pt").to(self.device)
            image_features = self.model.get_image_features(**inputs).pooler_output

            # L2 normalize and append to the list of all embeddings
            image_features = F.normalize(image_features, p=2, dim=-1)
            all_embeddings.append(image_features.cpu().numpy())
            
        return np.vstack(all_embeddings)

    def process_directory(self, frames_dir: str, batch_size: int = 64) -> List[Dict[str, Union[str, np.ndarray]]]:
        """
        Traverses the processed frames directory, embeds all images, and pairs them with their metadata.

        Args:
            frames_dir: The root directory containing the processed frame images organized by video and chunk.
            batch_size: Number of images to process in a single batch for embedding.
        
        Returns:
            A list of dictionaries: [{"video_id": "vid1", "chunk_id": "chunk_00000", "vector": [...]}]
        """

        image_paths = []
        metadata_records = []
        
        # Walk the directory structure: frames_dir / video_name / chunk_id / frame.jpg/jpeg/png
        for video_id in os.listdir(frames_dir):
            # Construct the path to the video directory
            video_path = os.path.join(frames_dir, video_id)
            if not os.path.isdir(video_path):
                continue
            
            # Walk through each chunk directory within the video directory
            for chunk_id in os.listdir(video_path):
                chunk_path = os.path.join(video_path, chunk_id)
                if not os.path.isdir(chunk_path):
                    continue
                
                # Collect all image paths and their corresponding metadata
                for frame_file in os.listdir(chunk_path):
                    if frame_file.endswith(('.jpg', '.jpeg', '.png')):
                        image_paths.append(os.path.join(chunk_path, frame_file))
                        metadata_records.append({
                            "video_id": video_id,
                            "chunk_id": chunk_id
                        })
                        
        # If no images were found, return an empty list
        if not image_paths:
            print("No frames found to embed.")
            return []

        # Generate embeddings for all collected paths
        embeddings = self.embed_images(image_paths, batch_size)
        
        # Merge embeddings with their metadata records
        for i, record in enumerate(metadata_records):
            record["vector"] = embeddings[i]
            
        return metadata_records

if __name__ == "__main__":
    # Example execution
    processed_frames_dir = "./data/processed_frames"
    embedder = MultimodalEmbedder(
        model_name="google/siglip-base-patch16-224"
    )
    
    print(f"Processing directory: {processed_frames_dir}")
    vector_data = embedder.process_directory(processed_frames_dir, batch_size=64)
    print(f"Generated {len(vector_data)} dense vectors.")