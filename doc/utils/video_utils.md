# video_utils.py - Documentation

## 1. Overview
`utils/video_utils.py` provides robust wrappers around OpenCV's video I/O capabilities. Reading and writing video files in Python can be error-prone (codec issues, corrupted frames, index errors). This module abstracts those complexities into clean, safe Context Managers.

## 2. Technical Overview
- **Type**: I/O Utility
- **Dependencies**: `cv2` (OpenCV), `pathlib`.
- **Patterns**: Context Manager (`with VideoReader(...) as ...`) ensures resources are released automatically.

## 3. Detailed Technical
- **VideoReader**:
    - Safely opens video files.
    - Provides a Python Generator `read_frames()` to iterate through video memory-efficiently (lazy loading).
    - Metadata extraction (FPS, Count, W/H).
- **VideoWriter**:
    - Automatic output directory creation.
    - Consistent codec handling ("mp4v" default).
    - Auto-resizing: If a frame passed to `write()` doesn't match the init dimensions, it resizes it to prevent crashes.

## 4. Functions

### `VideoReader` Class

##### `read_frames(self, max_frames) -> generator`
- **Purpose**: Iteration.
- **Details**: Yields `(frame_idx, frame_image)` tuples. Handles end-of-file gracefully.

### `VideoWriter` Class

##### `write_frame(self, frame)`
- **Purpose**: Saving.
- **Details**: Writes the NumPy image array to the video container. Includes safety check for dimensions.

### Helper Functions

##### `extract_frames(video_path, output_dir)`
- **Purpose**: Debugging.
- **Details**: Dumps individual JPGs from video.

##### `create_video_from_frames(frames_dir, output_path)`
- **Purpose**: Reconstruction.
- **Details**: Stitches a folder of JPGs back into an MP4.
