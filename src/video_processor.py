import os
import cv2
import json
from tqdm import tqdm
from scenedetect import detect, ContentDetector

class VideoProcessor:
    def __init__(self, raw_dir: str, output_dir: str, max_chunk_duration: float = 15.0, overlap: float = 2.0,
                 content_threshold: float = 27.0, fps: float = 1.0):
        """
        Initializes the Video Processor.
        
        Args:
            raw_dir: Directory containing the raw video files.
            output_dir: Directory to save the extracted chunk frames.
            max_chunk_duration: Maximum allowed length for a scene (seconds).
            overlap: Overlap duration for sliding windows when a scene is too long (seconds).
            content_threshold: Threshold for scene detection using ContentDetector.
            fps: Frames per second to extract.
        """
        self.raw_dir = raw_dir
        self.output_dir = output_dir
        self.max_chunk_duration = max_chunk_duration
        self.overlap = overlap
        self.content_threshold = content_threshold
        self.fps = fps
        self.metadata = {}  # Central dictionary to cache timestamps
        
        os.makedirs(self.output_dir, exist_ok=True)

    def process_all(self):
        """Iterates through all videos in the raw directory and saves metadata."""

        # List all video files in the raw directory
        video_files = [f for f in os.listdir(self.raw_dir) if f.endswith(('.mp4', '.avi', '.mov'))]
        
        if not video_files:
            print(f"No video files found in {self.raw_dir}")
            return

        # Process each video file
        for video_file in tqdm(video_files, desc="Processing videos"):
            video_path = os.path.join(self.raw_dir, video_file)
            self._process_single_video(video_path, video_file)
            
        # Write the central metadata JSON to disk
        metadata_path = os.path.join(self.output_dir, "metadata.json")
        with open(metadata_path, 'w') as f:
            json.dump(self.metadata, f, indent=4)
        print(f"\nProcessing complete. Metadata saved to {metadata_path}")

    def _split_long_scene(self, start_sec: float, end_sec: float):
        """Generator that yields sub-chunks with overlap if a scene exceeds the max duration.

        Args:
            start_sec: Start time of the scene.
            end_sec: End time of the scene.
        """

        # Calculate the total duration of the scene and determine if it needs to be split
        duration = end_sec - start_sec
        if duration <= self.max_chunk_duration:
            yield start_sec, end_sec
            return
        
        # Use a sliding window approach to create overlapping chunks
        current_start = start_sec
        while current_start < end_sec:
            current_end = min(current_start + self.max_chunk_duration, end_sec)
            yield current_start, current_end
            
            # If we've reached the end of the scene, break the loop
            if current_end == end_sec:
                break
                
            # Move the start forward
            current_start = current_end - self.overlap

    def _process_single_video(self, video_path: str, video_filename: str):
        """Detects scenes, applies chunking logic, and delegates frame extraction.
        
        Args:
            video_path: Path to the video file.
            video_filename: Name of the video file, used for metadata caching.
        """

        # Initialize the dictionary for this video in the central metadata cache
        video_name = os.path.splitext(video_filename)[0]
        self.metadata[video_name] = {}
        
        # Detect scenes changes in the video using PySceneDetect
        try:
            scene_list = detect(video_path, ContentDetector(threshold=self.content_threshold))
        except Exception as e:
            print(f"Error detecting scenes for {video_filename}: {e}")
            scene_list = []
        
        # Fallback if no scenes are detected
        if not scene_list:
            cap = cv2.VideoCapture(video_path)
            video_fps = cap.get(cv2.CAP_PROP_FPS)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            duration = total_frames / video_fps if video_fps > 0 else 0
            cap.release()
            scene_list = [(0.0, duration)]
        else:
            scene_list = [(scene[0].get_seconds(), scene[1].get_seconds()) for scene in scene_list]

        # Iterate through scenes, apply sliding window if needed, and extract
        chunk_counter = 0
        for start_sec, end_sec in scene_list:
            
            # Sub-divide the scene if it is longer than max_chunk_duration
            for sub_start, sub_end in self._split_long_scene(start_sec, end_sec):
                chunk_id = f"chunk_{chunk_counter:05d}"
                
                # Cache the timestamps in the metadata dictionary
                self.metadata[video_name][chunk_id] = {
                    "start_sec": round(sub_start, 2),
                    "end_sec": round(sub_end, 2)
                }
                
                # Create a directory for this chunk and extract frames
                chunk_dir = os.path.join(self.output_dir, video_name, chunk_id)
                os.makedirs(chunk_dir, exist_ok=True)
                
                self._extract_frames(video_path, sub_start, sub_end, chunk_dir)
                chunk_counter += 1

    def _extract_frames(self, video_path: str, start_sec: float, end_sec: float, output_dir: str):
        """Extracts frames at the configured FPS from the specified video time window.

        Args:            
            video_path: Path to the video file.
            start_sec: Start time of the chunk (seconds).
            end_sec: End time of the chunk (seconds).
            output_dir: Directory to save the extracted frames.
        """

        # Open the video file
        cap = cv2.VideoCapture(video_path)
        video_fps = cap.get(cv2.CAP_PROP_FPS)
        
        if not cap.isOpened() or video_fps == 0:
            return

        current_sec = start_sec
        frame_count = 0
        
        # Loop through the specified time window and extract frames
        while current_sec <= end_sec:
            # Calculate the exact frame index for this second
            frame_idx = int(current_sec * video_fps)
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            
            ret, frame = cap.read()
            if not ret:
                break  # Video ended prematurely
                
            frame_path = os.path.join(output_dir, f"frame_{frame_count:05d}.jpg")
            cv2.imwrite(frame_path, frame)
            
            current_sec += 1.0 / self.fps  # Move forward in time by the specified extraction fps
            frame_count += 1
            
        cap.release()

if __name__ == "__main__":
    # Execute the video processing pipeline
    processor = VideoProcessor(
        raw_dir="./data/raw_videos", 
        output_dir="./data/processed_frames",
        max_chunk_duration=15.0,
        overlap=2.0,
        content_threshold=27.0,
        fps=1.0
    )
    processor.process_all()