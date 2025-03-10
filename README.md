# Voronoi Audiovisual Generator

[![Void Cartographer - Consciousness](https://img.youtube.com/vi/2bIQQsa84vk/maxresdefault.jpg)](https://www.youtube.com/watch?v=2bIQQsa84vk)

**Consciousness** is a code-based audiovisual generative video project that uses Voronoi tessellation to create dynamic visualizations synced to audio. This project is built in Python, leveraging libraries like MoviePy, SciPy, and FFmpeg for audio processing, frame generation, and video encoding. The video is rendered in 1080p at 60 fps.

## Features
- **Generative Visualization**: Points form a Voronoi-like structure, expanding outward based on bass (20-200 Hz) and sub-bass (20-60 Hz) frequencies.
- **Treble-Driven Spin**: The rotation of points is controlled by treble frequencies (2 kHz - 8 kHz), syncing motion to high-pitched audio elements.
- **Dynamic Point Reset**: Points reset to the center after reaching a threshold, creating a continuous flow of motion.
- **High-Quality Output**: Renders at 1080x1920 resolution, 60 fps, with options for H.264 or H.265 encoding.

## Prerequisites
Before running the project, ensure you have the following installed:

- **Python 3.13.2+**
- **FFmpeg**: Needed for video encoding. Install via Homebrew on macOS (`brew install ffmpeg`) or download from [FFmpeg’s website](https://ffmpeg.org/download.html).
- **Dependencies**:
  - `numpy`: For numerical computations.
  - `moviepy`: For video generation and editing.
  - `Pillow`: For image processing.
  - `scipy`: For audio filtering (e.g., bandpass filters).
  - `soundfile`: For reading audio files.

Install dependencies using pip:
```bash
pip install numpy moviepy Pillow scipy soundfile
```

## Installation
1. Clone this repository:
   ```bash
   git clone https://github.com/yourusername/voronoi-audiovisual-generator.git
   cd voronoi-audiovisual-generator
   ```
2. Ensure FFmpeg is installed and accessible in your system’s PATH.
3. Place your audio file (`audio.wav`) in the project directory, or update the `audio_path` variable in the script to point to your audio file.

## Usage
**Run the Script**:
The main script (`voronoi_generator.py`) generates the video based on the provided audio file. To render the full video:
```bash
python voronoi_generator.py
```
- The script will process the audio, generate frames, and output a video file named `output.mp4`.
- The video duration matches the length of the audio file (e.g., ~60 seconds for "Consciousness v2.wav").

## Customization
You can tweak the following variables in the script to adjust the visualization:
- `amplitude_scale` (default: 100.0): Controls the intensity of bass-driven expansion.
- `sub_bass_amplitude_scale` (default: 60.0): Adjusts sub-bass expansion intensity.
- `treble_amplitude_scale` (default: 50.0): Modifies the treble-driven spin speed.
- `spring_strength` (default: 0.1): Affects how quickly points return to their base radius.
- `reset_probability` (default: 0.02): Probability of resetting points to the center when they exceed the threshold.

## Credits
- **Voice**: Maya from [Sesame AI Inc.](https://www.sesame.com/)
- **Libraries**: Thanks to the creators of MoviePy, SciPy, NumPy, Pillow, and FFmpeg.

## License
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
