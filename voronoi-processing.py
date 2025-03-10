import numpy as np
from moviepy.editor import VideoClip, AudioFileClip
from PIL import Image, ImageDraw
import logging
from scipy.signal import butter, lfilter
import soundfile as sf

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger()

# Customization variables
initial_point_count = 800
min_point_count = 500
max_dist_original = 100.0
point_size = 4
amplitude_scale = 100.0
sub_bass_amplitude_scale = 60.0
treble_amplitude_scale = 50.0
jitter_offset_base = 0.05
jitter_decay_factor = 0.9
spring_strength = 0.1
angular_velocity_base = 0.002
waveform_influence_factor = 0.0025
outer_spin_multiplier = 5
replace_chance = 0.25
replace_points_min = 3
replace_points_max = 10
reset_probability = 0.02

# Video and rendering settings
frame_rate = 60
original_width = 600
original_height = 800
render_width = 1080
render_height = 1920
internal_width = 2160
internal_height = 3840
scale_factor = internal_height / original_height
effective_max_radius = min(internal_width / 1.1, internal_height / 1.1)
max_dist = max_dist_original * scale_factor

reset_threshold = effective_max_radius * 1.5

# Audio file path
audio_path = "audio.wav"

# Initialize points
points = []
radii = []
angles = []
jitter_offsets = []
prev_radii = []
center = np.array([original_width / 2, original_height / 2]) * scale_factor
for _ in range(initial_point_count):
    if np.random.random() < 0.7:
        r = np.random.uniform(0, effective_max_radius / 10)
    else:
        r = np.random.uniform(effective_max_radius / 10, effective_max_radius / 2)
    angle = np.random.uniform(0, 2 * np.pi)
    points.append([r * np.cos(angle), r * np.sin(angle)])
    radii.append(r)
    angles.append(angle)
    jitter_offsets.append(0.0)
    prev_radii.append(r)
points = np.array(points)
radii = np.array(radii)
angles = np.array(angles)
jitter_offsets = np.array(jitter_offsets)
prev_radii = np.array(prev_radii)

# Load audio with soundfile
logger.info("Loading audio with soundfile...")
samples, sample_rate = sf.read(audio_path)
if len(samples.shape) > 1:
    samples = np.mean(samples, axis=1)  # Convert to mono
full_duration = len(samples) / sample_rate
if samples.max() > 0:
    samples = samples / samples.max()  # Normalize to [-1, 1]
logger.info(f"Loaded audio. Sample Rate: {sample_rate} Hz, Full Duration: {full_duration:.2f}s, Samples: {len(samples)}, Max Amplitude: {np.max(np.abs(samples)):.3f}")

# Apply bandpass filter for bass (20-200 Hz)
lowcut_bass = 20.0
highcut_bass = 200.0
nyquist = 0.5 * sample_rate
low_bass = lowcut_bass / nyquist
high_bass = highcut_bass / nyquist
b_bass, a_bass = butter(3, [low_bass, high_bass], btype='band')
bass_samples = lfilter(b_bass, a_bass, samples)
bass_samples = np.nan_to_num(bass_samples, nan=0.0, posinf=0.0, neginf=0.0)
logger.info(f"Filtered bass (20-200 Hz). Max Filtered Amplitude: {np.max(np.abs(bass_samples)):.3f}")

# Apply bandpass filter for sub-bass (20-60 Hz)
lowcut_sub = 20.0
highcut_sub = 60.0
low_sub = lowcut_sub / nyquist
high_sub = highcut_sub / nyquist
b_sub, a_sub = butter(3, [low_sub, high_sub], btype='band')
sub_bass_samples = lfilter(b_sub, a_sub, samples)
sub_bass_samples = np.nan_to_num(sub_bass_samples, nan=0.0, posinf=0.0, neginf=0.0)
logger.info(f"Filtered sub-bass (20-60 Hz). Max Filtered Amplitude: {np.max(np.abs(sub_bass_samples)):.3f}")

# Apply bandpass filter for treble (2 kHz - 8 kHz)
lowcut_treble = 2000.0
highcut_treble = 8000.0
low_treble = lowcut_treble / nyquist
high_treble = highcut_treble / nyquist
b_treble, a_treble = butter(3, [low_treble, high_treble], btype='band')
treble_samples = lfilter(b_treble, a_treble, samples)
treble_samples = np.nan_to_num(treble_samples, nan=0.0, posinf=0.0, neginf=0.0)
logger.info(f"Filtered treble (2 kHz - 8 kHz). Max Filtered Amplitude: {np.max(np.abs(treble_samples)):.3f}")

# Frame settings
samples_per_frame = int(sample_rate / frame_rate)

# Smoothing variables
prev_bass_amplitude = 0.0
bass_alpha = 0.2
prev_sub_bass_amplitude = 0.0
sub_bass_alpha = 0.2
prev_treble_amplitude = 0.0
treble_alpha = 0.2
prev_jitter = 0.0
jitter_alpha = 0.3
prev_bass_expansion = 0.0
bass_expansion_alpha = 0.1
prev_sub_bass_expansion = 0.0
sub_bass_expansion_alpha = 0.1

# Frame generation function
def make_frame(t):
    global points, radii, angles, jitter_offsets, prev_radii, prev_bass_amplitude, prev_sub_bass_amplitude, prev_treble_amplitude, prev_jitter, prev_bass_expansion, prev_sub_bass_expansion

    try:
        frame_idx = int(t * frame_rate)
        sample_start = frame_idx * samples_per_frame
        sample_end = min(sample_start + samples_per_frame, len(bass_samples))
        if sample_start >= len(bass_samples):
            raw_segment = np.zeros(samples_per_frame, dtype=np.float64)
            bass_segment = np.zeros(samples_per_frame, dtype=np.float64)
            sub_bass_segment = np.zeros(samples_per_frame, dtype=np.float64)
            treble_segment = np.zeros(samples_per_frame, dtype=np.float64)
        else:
            raw_segment = samples[sample_start:sample_end]
            bass_segment = bass_samples[sample_start:sample_end]
            sub_bass_segment = sub_bass_samples[sample_start:sample_end]
            treble_segment = treble_samples[sample_start:sample_end]
            if len(bass_segment) < samples_per_frame:
                pad_length = samples_per_frame - len(bass_segment)
                raw_segment = np.pad(raw_segment, (0, pad_length), 'constant')
                bass_segment = np.pad(bass_segment, (0, pad_length), 'constant')
                sub_bass_segment = np.pad(sub_bass_segment, (0, pad_length), 'constant')
                treble_segment = np.pad(treble_segment, (0, pad_length), 'constant')

        # Compute RMS amplitudes
        raw_amplitude = np.sqrt(np.mean(raw_segment ** 2)) if len(raw_segment) > 0 else 0.0
        bass_amplitude_raw = np.sqrt(np.mean(bass_segment ** 2)) * amplitude_scale
        bass_amplitude = bass_alpha * bass_amplitude_raw + (1 - bass_alpha) * prev_bass_amplitude
        bass_amplitude = max(min(bass_amplitude, 100.0), 0.0)
        prev_bass_amplitude = bass_amplitude

        sub_bass_amplitude_raw = np.sqrt(np.mean(sub_bass_segment ** 2)) * sub_bass_amplitude_scale
        sub_bass_amplitude = sub_bass_alpha * sub_bass_amplitude_raw + (1 - sub_bass_alpha) * prev_sub_bass_amplitude
        sub_bass_amplitude = max(min(sub_bass_amplitude, 100.0), 0.0)
        prev_sub_bass_amplitude = sub_bass_amplitude

        treble_amplitude_raw = np.sqrt(np.mean(treble_segment ** 2)) * treble_amplitude_scale
        treble_amplitude = treble_alpha * treble_amplitude_raw + (1 - treble_alpha) * prev_treble_amplitude
        treble_amplitude = max(min(treble_amplitude, 100.0), 0.0)
        prev_treble_amplitude = treble_amplitude

        img = Image.new('RGB', (internal_width, internal_height), color=(0, 0, 0))
        draw = ImageDraw.Draw(img)

        sub_width = int(original_width * scale_factor)
        x_offset = (internal_width - sub_width) // 2
        adjusted_center = np.array([internal_width / 2, internal_height / 2])
        for i in range(len(points)):
            base_radius = radii[i]
            raw_jitter = np.interp(bass_amplitude, [0, 100], [0, effective_max_radius * jitter_offset_base])
            jitter = jitter_alpha * raw_jitter + (1 - jitter_alpha) * prev_jitter
            prev_jitter = jitter
            jitter_offset = jitter_offsets[i] * jitter_decay_factor + jitter
            jitter_offsets[i] = jitter_offset

            spring_force = (base_radius - radii[i]) * spring_strength
            base_radius += spring_force
            expansion_factor = 1.0 if bass_amplitude <= 20 else 2.0
            normalized_bass = (bass_amplitude / 100) ** 2
            bass_expansion_base = normalized_bass * (effective_max_radius * 1.5) * expansion_factor
            bass_expansion = bass_expansion_alpha * bass_expansion_base + (1 - bass_expansion_alpha) * prev_bass_expansion
            prev_bass_expansion = bass_expansion

            sub_bass_expansion_base = (sub_bass_amplitude / 100) ** 2 * (effective_max_radius * 1.5)
            sub_bass_expansion = sub_bass_expansion_alpha * sub_bass_expansion_base + (1 - sub_bass_expansion_alpha) * prev_sub_bass_expansion
            prev_sub_bass_expansion = sub_bass_expansion

            new_radius = base_radius + jitter_offset + bass_expansion + sub_bass_expansion

            adjusted_radius = 0.6 * new_radius + 0.4 * prev_radii[i]
            if adjusted_radius > effective_max_radius * 2.0 or bass_amplitude < 15:
                adjusted_radius *= 0.99

            adjusted_radius = np.clip(adjusted_radius, 0, effective_max_radius * 2.0)
            prev_radii[i] = adjusted_radius

            # Reset points that go too far
            if adjusted_radius > reset_threshold and np.random.random() < reset_probability:
                new_r = np.random.uniform(0, effective_max_radius / 15)
                points[i] = [new_r * np.cos(angles[i]), new_r * np.sin(angles[i])]
                radii[i] = new_r
                prev_radii[i] = new_r

            angle = angles[i]
            max_angular_velocity = angular_velocity_base * (1.0 + outer_spin_multiplier * (adjusted_radius / effective_max_radius))
            angular_velocity = max_angular_velocity * (treble_amplitude / 100)
            waveform_influence = min(bass_amplitude * waveform_influence_factor, 0.1)
            angle += angular_velocity + waveform_influence
            angles[i] = angle

            new_x = adjusted_center[0] + np.cos(angle) * adjusted_radius
            new_y = adjusted_center[1] + np.sin(angle) * adjusted_radius
            new_x = np.clip(new_x, 0, internal_width - 1)
            new_y = np.clip(new_y, 0, internal_height - 1)
            points[i] = [(new_x - x_offset) / scale_factor, (new_y) / scale_factor]

            if i == 0:
                logger.info(f"Frame: {frame_idx}, Time: {t:.3f}s, Raw Amplitude: {raw_amplitude:.3f}, "
                            f"Bass Amplitude: {bass_amplitude:.3f}, Sub-Bass Amplitude: {sub_bass_amplitude:.3f}, "
                            f"Treble Amplitude: {treble_amplitude:.3f}, Jitter: {jitter:.3f}, "
                            f"Bass Expansion: {bass_expansion:.3f}, Sub-Bass Expansion: {sub_bass_expansion:.3f}, "
                            f"Adjusted Radius: {adjusted_radius:.3f}, Angular Velocity: {angular_velocity:.6f}")

        for i in range(len(points)):
            p1 = tuple((points[i] * scale_factor + [x_offset, 0]).astype(int))
            for j in range(i + 1, min(i + 20, len(points))):
                p2 = tuple((points[j] * scale_factor + [x_offset, 0]).astype(int))
                dist = np.linalg.norm(np.array(p1) - np.array(p2))
                if dist < max_dist:
                    safe_bass_amplitude = 0 if np.isnan(bass_amplitude) else bass_amplitude
                    r = min(int(50 + bass_amplitude * 2), 255)
                    g = min(int(100 + safe_bass_amplitude * 1.5), 255)
                    b = min(int(150 + bass_amplitude * 2), 255)
                    color = (r, g, b)
                    draw.line([p1, p2], fill=color, width=2)

        for p in points:
            p_scaled = tuple((p * scale_factor + [x_offset, 0]).astype(int))
            draw.ellipse([p_scaled[0] - point_size, p_scaled[1] - point_size,
                          p_scaled[0] + point_size, p_scaled[1] + point_size],
                         fill=(255, 255, 255))

        if np.random.random() < replace_chance or len(points) < min_point_count:
            points_to_replace = np.random.randint(replace_points_min, replace_points_max + 1)
            points_needed = max(min_point_count - len(points), 0)
            points_to_replace = max(points_to_replace, points_needed)

            points_to_remove = min(points_to_replace, max(0, len(points) - min_point_count))
            if points_to_remove > 0:
                remove_indices = np.random.choice(len(points), points_to_remove, replace=False)
                points = np.delete(points, remove_indices, axis=0)
                radii = np.delete(radii, remove_indices)
                angles = np.delete(angles, remove_indices)
                jitter_offsets = np.delete(jitter_offsets, remove_indices)
                prev_radii = np.delete(prev_radii, remove_indices)

            points_to_add = points_to_replace - points_to_remove
            for _ in range(points_to_add):
                if np.random.random() < 0.7:
                    r = np.random.uniform(0, effective_max_radius / 10)
                else:
                    r = np.random.uniform(effective_max_radius / 10, effective_max_radius / 2)
                angle = np.random.uniform(0, 2 * np.pi)
                points = np.append(points, [[r * np.cos(angle), r * np.sin(angle)]], axis=0)
                radii = np.append(radii, r)
                angles = np.append(angles, angle)
                jitter_offsets = np.append(jitter_offsets, 0.0)
                prev_radii = np.append(prev_radii, r)

        img = img.resize((render_width, render_height), Image.LANCZOS)
        return np.array(img)
    except Exception as e:
        logger.error(f"Error generating frame at time {t:.3f}s: {str(e)}")
        # Return a black frame as a fallback
        return np.zeros((render_height, render_width, 3), dtype=np.uint8)

# Generate video
audio_clip = AudioFileClip(audio_path)
video_clip = VideoClip(make_frame, duration=full_duration)
video_clip = video_clip.set_audio(audio_clip)
video_clip.write_videofile(
    "output.mp4",
    fps=frame_rate,
    codec="libx264",
    audio_codec="aac",
    bitrate="8000k",
    ffmpeg_params=[
        "-r", str(frame_rate),  # Force constant frame rate
        "-profile:v", "main",  # Use Main profile for H.264
        "-level", "4.0",       # Set H.264 level for compatibility
        "-g", str(frame_rate),  # Set keyframe interval to 1 second (60 frames at 60 fps)
        "-loglevel", "verbose"
    ]
)

print("Video generation complete: output.mp4")
