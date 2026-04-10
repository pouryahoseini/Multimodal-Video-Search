import os
import warnings

# Suppress warnings from transformers library to keep the output clean
os.environ["TRANSFORMERS_VERBOSITY"] = "error"
warnings.filterwarnings("ignore", category=UserWarning, module="transformers")

import streamlit as st
from main import load_search_models, run_search

# Page Config
st.set_page_config(page_title="Semantic Video Search Engine", layout="wide")
st.title("Semantic Video Search Engine")

# Inject CSS to change the text input focus outline color to green
st.markdown(
    """
    <style>
    div[data-baseweb="input"]:focus-within {
        border-color: green !important;
        box-shadow: 0 0 0 1px green !important;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Check if the necessary data files exist before proceeding
if not os.path.exists("./data/processed_frames") or not os.path.exists("./data/faiss_index.bin"):
    st.error("Processed frames or Faiss index not found. Please run \"python main.py --build\" first.")
    st.stop()

# Cache the heavy models so they only load once
@st.cache_resource
def load_backend():
    frames_dir = "./data/processed_frames"
    index_path = "./data/faiss_index.bin"
    embedder, vector_store, reranker = load_search_models(frames_dir=frames_dir, 
                                                          index_path=index_path,
                                                          fps=1.0,
                                                          embedding_model_name="google/siglip-base-patch16-224",
                                                          vlm_model_name="Qwen/Qwen2-VL-7B-Instruct"
                                                          )

    return embedder, vector_store, reranker

with st.spinner("Loading AI Models..."):
    embedder, vector_store, reranker = load_backend()

# The UI
with st.form("search_form"):
    query = st.text_input("What are you looking for?", placeholder="e.g., A Nike backpack...")
    submitted = st.form_submit_button("Search Videos")

# Auto-focus the search input box using a JavaScript injection
st.iframe(
    """
    <script>
    const input = window.parent.document.querySelector('input[type="text"]');
    if (input) {
        input.focus();
    }
    </script>
    """
)

if submitted and query:
    with st.spinner("Searching..."):
        # Run search and get results
        results = run_search(
            query=query, 
            embedder=embedder, 
            vector_store=vector_store, 
            reranker=reranker, 
            frames_dir="./data/processed_frames", 
            top_k=5,
            reranking_fusion_alpha=0.3
        )
        
        st.success(f"Found {len(results)} matches!")
        
        # Render the results dynamically with columns
        for rank, res in enumerate(results):
            st.markdown(f"### Rank {rank + 1} | Confidence: {res['final_fused_score']:.3f}")
            
            # Create three columns. The [3, 2, 5] ratio means the left text column gets 30% of the space, 
            # and the middle video column gets 20%. The last column is just a spacer to push the video to the left.
            col1, col2, _ = st.columns([3, 2, 5])
            
            # Put the text metadata in the left column
            with col1:
                st.markdown(f"**File:** `{res['video_id']}.mp4`")
                st.markdown(f"**Time:** `{res['start_sec']}s - {res['end_sec']}s`")
                st.caption(f"Visual Retrieval Score: {res['norm_stage1_score']:.2f}")
                st.caption(f"VLM Rerank Score: {res['vlm_yes_prob']:.2f}")
                
            # Put the video in the middle column
            with col2:
                video_file = os.path.join("./data/raw_videos", f"{res['video_id']}.mp4")
                if os.path.exists(video_file):
                    # Safely handle sub-second rounding collisions
                    start_t = int(res['start_sec'])
                    end_t = int(res['end_sec'])
                    if end_t <= start_t:
                        end_t = start_t + 1
                    
                    # Show the video clip for this result
                    st.video(video_file, start_time=start_t, end_time=end_t)
                else:
                    st.error("Original video file not found on disk.")
            
            # Add a divider after each result
            st.divider() 