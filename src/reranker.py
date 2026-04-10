import os
import torch   
import random
from typing import List, Dict, Any, Optional
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info

class Reranker:
    def __init__(self, model_name: str = "Qwen/Qwen2-VL-7B-Instruct", fps: float = 1.0, device: Optional[str] = None):
        """
        Initializes the Qwen2-VL model for generative reranking.

        Args:
            model_name: The HuggingFace model identifier for Qwen2-VL.
            fps: The frames per second for video processing by Qwen2-VL.
            device: 'cuda' or 'cpu'. Auto-detects if None.
        """

        self.fps = fps

        # Determine the device to run
        self.device = device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Loading {model_name} onto {self.device}...")
        
        # Load model in bfloat16 to save VRAM
        self.model = Qwen2VLForConditionalGeneration.from_pretrained(
            model_name, 
            torch_dtype=torch.bfloat16, 
            device_map=self.device
        )
        self.model.eval()
        self.processor = AutoProcessor.from_pretrained(model_name)
        
        # Pre-compute target token IDs to extract their probabilities
        self.yes_tokens = self.processor.tokenizer.encode("Yes yes", add_special_tokens=False)

    def _normalize_scores(self, results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Applies Min-Max normalization strictly within the top-K pool.

        Args:
            results: List of dictionaries containing at least the "score" key.
        
        Returns:
            The same list of dictionaries with an added "norm_stage1_score" key for the normalized score.
        """

        # Guard for empty input
        if not results:
            return results
        
        # Calculate min and max scores for normalization
        scores = [res["score"] for res in results]
        min_score = min(scores)
        max_score = max(scores)
        
        # Calculate the denominator for normalization
        denominator = max_score - min_score if max_score > min_score else 1e-6
        
        # Apply Min-Max normalization to each score and store in a new key
        for res in results:
            res["norm_stage1_score"] = (res["score"] - min_score) / denominator
            
        return results

    @torch.no_grad()
    def _get_yes_probability(self, chunk_dir: str, query: str) -> float:
        """
        Loads the frames of a chunk, prompts Qwen2-VL, and extracts the probability of the 'Yes' token.

        Args:
            chunk_dir: Directory containing the frames for a specific video chunk.
            query: The text query to be used in the prompt.

        Returns:
            The probability of the 'Yes' token.
        """

        # Gather frames for the chunk
        frame_files = sorted([f for f in os.listdir(chunk_dir) if f.endswith(('.jpg', '.jpeg', '.png'))])
        if not frame_files:
            return 0.0
            
        video_input = [f"file://{os.path.join(chunk_dir, f)}" for f in frame_files]

        # Construct the prompt
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "video",
                        "video": video_input,
                        "fps": self.fps,
                    },
                    {
                        "type": "text", 
                        "text": f'Answer with a single word. Does this video show "{query}"? Answer strictly "Yes" or "No".'
                    },
                ],
            }
        ]

        # Format inputs for Qwen2-VL
        text_prompt = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs = process_vision_info(messages)
        
        inputs = self.processor(
            text=[text_prompt],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        ).to(self.device)

        # Forward pass: Generate 1 token and extract logits
        outputs = self.model.generate(
            **inputs, 
            max_new_tokens=1, 
            return_dict_in_generate=True, 
            output_scores=True,
            do_sample=False,
            temperature=None,
            top_p=None,
            top_k=None
        )
        
        # Take the logits of the first generated token
        first_token_logits = outputs.scores[0][0] 
        
        # Convert logits to probabilities via softmax
        probabilities = torch.nn.functional.softmax(first_token_logits, dim=-1)
        
        # Sum the probabilities of the target "yes" tokens
        yes_prob = sum(probabilities[token_id].item() for token_id in self.yes_tokens)
        
        return yes_prob

    def rerank(self, query: str, top_k_results: List[Dict[str, Any]], frames_base_dir: str, alpha: float = 0.3) -> List[Dict[str, Any]]:
        """
        Executes the reranking and fuses the scores.
        
        Args:
            query: The user's text query.
            top_k_results: The list of dictionaries returned by VectorStore.search_and_pool().
            frames_base_dir: The root directory where video chunks are saved.
            alpha: Weight for stage 1 retrieval. (1 - alpha) is the weight for stage 2 reranking.

        Returns:
            A re-ranked list of dictionaries with added "vlm_yes_prob" and "final_fused_score" keys, 
            sorted by "final_fused_score" in descending order.
        """

        # print(f"\nReranking Top {len(top_k_results)} chunks using Qwen2-VL...")
        
        # Normalize stage 1 scores
        results = self._normalize_scores(top_k_results)
        
        # Generate VLM scores
        for i, res in enumerate(results):
            chunk_dir = os.path.join(frames_base_dir, res["video_id"], res["chunk_id"])
            
            # Extract the raw Yes probability
            vlm_prob = self._get_yes_probability(chunk_dir, query)
            res["vlm_yes_prob"] = vlm_prob
            
            # Score fusion
            res["final_fused_score"] = (alpha * res["norm_stage1_score"]) + ((1.0 - alpha) * vlm_prob)

        # Sort by the new fused score
        final_ranked_results = sorted(results, key=lambda x: x["final_fused_score"], reverse=True)
        
        return final_ranked_results

if __name__ == "__main__":
    # Dummy test execution for the reranker
    print("Running Stage 2 Reranker math test...")
    
    # Create a mock class to bypass the model download for testing
    class MockReranker(Reranker):
        def __init__(self):
            # Override __init__ to prevent downloading Qwen2-VL during tests
            print("MockReranker initialized (Skipping heavy VLM load).")
            
        def _get_yes_probability(self, chunk_dir: str, query: str) -> float:
            # Return a random Yes probability
            prob = random.uniform(0.1, 0.99)
            print(f"  [Mock VLM] Looked at {chunk_dir}, Yes Probability: {prob:.3f}")
            return prob

    # Generate fake top-K results from stage 1 (faiss)
    fake_top_k = [
        {"video_id": "vid_1", "chunk_id": "chunk_0", "score": 0.34},
        {"video_id": "vid_2", "chunk_id": "chunk_3", "score": 0.28},
        {"video_id": "vid_1", "chunk_id": "chunk_2", "score": 0.45},
        {"video_id": "vid_3", "chunk_id": "chunk_1", "score": 0.15},
    ]
    
    # Run the fusion test with alpha=0.3 (30% weight to stage 1, 70% to VLM)
    print("\n--- Starting Mock Fusion ---")
    tester = MockReranker()
    
    final_results = tester.rerank(
        query="A Nike backpack", 
        top_k_results=fake_top_k, 
        frames_base_dir="", 
        alpha=0.3
    )
    
    # Display the sorted output
    print("\n--- Final Sorted Output ---")
    for rank, res in enumerate(final_results):
        print(f"Rank {rank+1}: Video {res['video_id']} | Chunk {res['chunk_id']} | "
              f"Normalized retrieval score: {res['norm_stage1_score']:.3f} | "
              f"VLM probability: {res['vlm_yes_prob']:.3f} | "
              f"Fused score: {res['final_fused_score']:.3f}")