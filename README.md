# Multimodal Video Search Engine

An end-to-end, two-stage multimodal AI pipeline for zero-shot semantic video retrieval. 

This system ingests raw video files, intelligently chunks them using scene detection, and allows users to search for highly specific visual moments using natural language. Instead of returning entire video files, the pipeline retrieves video segments with start and end timestamps.

---

## 🧠 Architectural & Design Highlights

This system bypasses simple keyword matching by relying entirely on dense vector representations and generative visual logic. It uses a **Two-Stage Retrieval Architecture** to balance extreme speed with high precision.

### 1. Stage 1: High-Recall Spatial Retrieval (SigLIP + Faiss)
* **The Concept:** We use Google's `SigLIP-base` to project both text queries and video frames into a shared 768-dimensional contrastive latent space. 
* **The Implementation:** Videos are processed into distinct visual chunks. Frames are extracted at 1 FPS, embedded, and stored in a CPU-bound `Faiss IndexFlatIP` database. 
* **Why this works:** Faiss allows us to search tens of thousands of frames in milliseconds. By keeping the index on the CPU, we reserve critical VRAM for the Generative VLM in Stage 2.

### 2. Stage 2: High-Precision Generative Reranking (Qwen2-VL)
* **The Concept:** Contrastive models (like SigLIP) are excellent at identifying objects but struggle with complex spatial relationships. Because `Qwen2-VL-7B-Instruct` processes video frames natively, it acts as a logical gatekeeper capable of understanding both spatial layout and temporal sequence.
* **The Engineering:** Instead of prompting the VLM to generate a string of text (which is slow and hard to parse), we force it to output a single token ("Yes" or "No"). We extract the raw mathematical logits of the "Yes" token using PyTorch, convert it to a probability via softmax, and use it as a continuous scoring function.
* **Score Fusion:** The final confidence score is a convex combination of the normalized Faiss inner-product (stage 1) and the VLM's logical probability (stage 2): $S_{final} = \alpha * S_{stage1} + (1 - \alpha) * S_{stage2}$.

### 3. Semantic Chunking & Timestamping
* **The Concept:** Returning a 5-minute video for a 3-second action is a poor user experience.
* **The Implementation:** The ingestion pipeline uses `PySceneDetect` to semantically slice videos based on visual content changes, combined with a bounded sliding window mechanism to restrict maximum duration. Overlapping these bounded chunks ensures continuous temporal coverage and guarantees no fast-moving action is lost across boundaries.

---

## 📂 Repository Structure

    multimodal-video-search/
    ├── data/
    │   ├── raw_videos/                 # Place raw .mp4 files here
    │   ├── processed_frames/           # Auto-generated visual chunks
    │   │   └── metadata.json           # Auto-generated chunk-to-video relationship mapping
    │   ├── faiss_index.bin             # Auto-generated disk-backed vector store
    │   └── faiss_index.bin.map.json    # Auto-generated persistent ID-to-chunk Faiss mapping
    ├── src/
    │   ├── video_processor.py          # Video chunking & frame extraction
    │   ├── embedder.py                 # SigLIP tensor processing
    │   ├── vector_store.py             # Faiss indexing & persistence
    │   └── reranker.py                 # Qwen2-VL logit extraction
    ├── main.py                         # CLI Orchestrator (Build/Search)
    ├── app.py                          # Streamlit Web UI
    ├── requirements.txt
    └── README.md

---

## 🚀 Installation & Execution

### 1. Environment Setup
*Note: An NVIDIA GPU with at least 16GB VRAM is strongly recommended for CUDA acceleration.*

First, install PyTorch configured for your specific CUDA version (Example for CUDA 12.9):

`pip install torch torchvision --index-url https://download.pytorch.org/whl/cu129`

Then, install the remaining pipeline dependencies:

`pip install -r requirements.txt`

### 2. Building the Index (Data Ingestion)
Place your `.mp4` files into `data/raw_videos/`. Then, run the build command to chunk the videos, embed the frames, and save the Faiss index to disk:

`python main.py --build`

### 3. Running the Search Engine
You can interact with the search engine using either the web interface or the terminal. Both interfaces will automatically load the saved Faiss index from disk without re-embedding the videos.

**Option A: Web UI**
Launch the Streamlit web interface for a rich, visual search experience with interactive video playback:

`streamlit run app.py`

**Option B: Command Line Interface (CLI)**
Run the search engine directly in your terminal for fast, text-based interactive querying:

`python main.py --query`

---

## 🔍 Retrieval Examples

The system handles zero-shot, highly descriptive queries. Examples include:
* "Indoor scenes with laptops and coffee"
* "Vehicles on a highway at night"
* "A Nike backpack"

A few successful examples of top retrievals in a dataset of ~130 videos are shown below.

<div style="display:inline-block; margin:10px; text-align:center;">
  <img src="https://github.com/user-attachments/assets/d93be5d3-7809-419b-a85c-b00a7567d492" width="29%" alt="riding on a street">
  <div>riding on a street</div>
</div>

<div style="display:inline-block; margin:10px; text-align:center;">
  <img src="https://github.com/user-attachments/assets/ecea60fb-d1af-4178-ada9-27adf59718fd" width="29%" alt="shovel">
  <div>shovel</div>
</div>

<div style="display:inline-block; margin:10px; text-align:center;">
  <img src="https://github.com/user-attachments/assets/9860e8ee-39cf-4fbe-a8ba-7eb43c01f34b" width="29%" alt="wine salute">
  <div>wine salute</div>
</div>

<div style="display:inline-block; margin:10px; text-align:center;">
  <img src="https://github.com/user-attachments/assets/126285c9-421d-40bc-8d1a-c2ad2ef74277" width="29%" alt="hockey goalie">
  <div>hockey goalie</div>
</div>

<div style="display:inline-block; margin:10px; text-align:center;">
  <img src="https://github.com/user-attachments/assets/31914be5-bf4e-452d-9cd8-35e0cc211d54" width="29%" alt="a person is playing the piano">
  <div>a person is playing the piano</div>
</div>

The UI returns the top 5 ranking segments, dynamically rendering a video player clipped to the exact `start_sec` and `end_sec` of the identified action.

---

## ⚠️ Known Limitations & Failure Modes

* **Text-in-Image Hallucinations:** While SigLIP is highly robust, it can occasionally struggle with exact OCR. For example, searching for a specific brand logo might yield a visually similar logo of a different brand if the VLM reranker's confidence is not high enough to correct it.
* **Complex Spatial Negation:** Queries involving negation (e.g., "A street with NO cars") often trip up contrastive models like SigLIP, as the embedding of the word "cars" forces the vector closer to images of cars. The Stage 2 Qwen2-VL reranker mitigates this, but an exceptionally high Stage 1 score can occasionally overpower the fusion equation.

---

## 📈 Scaling to Enterprise Volume

While this architecture performs exceptionally well on datasets of thousands of video clips, scaling to **millions of videos** requires modifications to the underlying ingestion and retrieval mechanics:

1. **Overcoming RAM Exhaustion (Ingestion):** Currently, the `IndexFlatIP` Faiss backend accumulates vectors in system RAM before writing to disk. At 10+ million chunks, this would trigger an Out-Of-Memory (OOM) crash. To scale, the pipeline should be upgraded to use `OnDiskInvertedLists` or Index Sharding, allowing vectors to stream directly to NVMe SSDs during the `add()` operation.
2. **Vector Compression (Search):** Storing 10 million 768-dimensional uncompressed vectors requires ~30GB of RAM. The index should be migrated to `IndexIVFPQ` (Inverted File with Product Quantization) to mathematically compress the latent space, reducing the memory footprint by roughly 30x with negligible impact on retrieval recall.
3. **Audio Modality Integration:** The underlying architecture can be extended with the potential integration of audio processing to capture conversational or environmental audio context alongside the visual data.
