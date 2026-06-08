"""
video_face_encoder.py
Extracts face encodings from a short enrollment video (person turning face 360°).
Samples frames evenly, filters blurry ones, picks the best diverse encodings.
"""
import cv2, face_recognition, pickle, os, numpy as np
from pathlib import Path

PERSONS_DIR    = Path('authorized_persons')
ENCODINGS_FILE = 'face_encodings.pkl'
SAMPLE_FRAMES  = 60      # How many frames to sample from video (more = better coverage)
BLUR_THRESHOLD = 40.0    # Laplacian variance; below this = too blurry, skip
                         # Lowered from 80 — browser webm is inherently softer
MAX_ENC_PER_PERSON = 12  # Store up to 12 diverse encodings per person


def is_blurry(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return cv2.Laplacian(gray, cv2.CV_64F).var() < BLUR_THRESHOLD


def extract_encodings_from_video(video_path, sample_count=SAMPLE_FRAMES):
    """Sample frames from video, return list of face encodings.

    Uses sequential reading so it works with .webm and other container formats
    where CAP_PROP_FRAME_COUNT is unreliable or returns -1.
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"    [ENCODER] Cannot open video: {video_path}")
        return []

    # Read all frames first (works for webm/mkv where frame count is unknown)
    all_frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        all_frames.append(frame)
    cap.release()

    total = len(all_frames)
    if total == 0:
        print(f"    [ENCODER] No frames read from: {video_path}")
        return []

    print(f"    [ENCODER] Read {total} frames from video")

    # Sample evenly across all frames
    indices = np.linspace(0, total - 1, min(sample_count, total), dtype=int)
    encodings = []

    for idx in indices:
        frame = all_frames[int(idx)]
        if is_blurry(frame):
            continue

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Try HOG first (fast). If it finds nothing, try with upsample=1
        # which helps catch smaller faces in wide-angle webcam frames.
        locs = face_recognition.face_locations(rgb, model='hog', number_of_times_to_upsample=1)
        if not locs:
            # Retry with upsample=2 on a slightly upscaled frame
            h, w = rgb.shape[:2]
            if h < 480:
                rgb_up = cv2.resize(rgb, (w*2, h*2))
                locs = face_recognition.face_locations(rgb_up, model='hog', number_of_times_to_upsample=1)
                encs = face_recognition.face_encodings(rgb_up, locs)
            else:
                encs = []
        else:
            encs = face_recognition.face_encodings(rgb, locs)

        if encs:
            encodings.append(encs[0])   # Take first (largest) face

    print(f'    [ENCODER] {len(indices)} frames sampled, {len(encodings)} faces found')
    if not encodings:
        print(f'    [ENCODER] TIP: Make sure face is well-lit and clearly visible.')
        print(f'              If video is very short (<3s), try re-recording a longer one.')
    return encodings


def diversify_encodings(encodings, max_count=MAX_ENC_PER_PERSON):
    """
    Greedy farthest-point selection to keep diverse encodings.
    This ensures we capture different face angles rather than near-duplicates.
    """
    if len(encodings) <= max_count:
        return encodings

    selected = [encodings[0]]
    remaining = encodings[1:]

    while len(selected) < max_count and remaining:
        # Find the encoding most different from all selected
        max_min_dist = -1
        best = None
        best_idx = 0
        for i, enc in enumerate(remaining):
            dists = face_recognition.face_distance(selected, enc)
            min_dist = np.min(dists)
            if min_dist > max_min_dist:
                max_min_dist = min_dist
                best = enc
                best_idx = i
        selected.append(best)
        remaining.pop(best_idx)

    return selected


def encode_all_persons():
    """Scan authorized_persons/ folder, encode videos, save .pkl database."""
    known_encodings = []
    known_names     = []
    known_ids       = []
    summary         = {}

    if not PERSONS_DIR.exists():
        print(f'[ENCODER] Folder not found: {PERSONS_DIR}')
        return [], [], []

    for person_folder in sorted(PERSONS_DIR.iterdir()):
        if not person_folder.is_dir():
            continue

        # Read metadata if exists
        meta_file = person_folder / 'meta.txt'
        person_id = person_folder.name
        display_name = person_id.replace('_', ' ').title()

        if meta_file.exists():
            meta = {}
            for line in meta_file.read_text().splitlines():
                if ':' in line:
                    k, v = line.split(':', 1)
                    meta[k.strip()] = v.strip()
            display_name = meta.get('name', display_name)

        # Find video file
        video_path = None
        for ext in ['*.mp4', '*.mov', '*.avi', '*.webm', '*.mkv']:
            videos = list(person_folder.glob(ext))
            if videos:
                video_path = videos[0]
                break

        if not video_path:
            print(f'  [{display_name}] No video found — skipping')
            continue

        print(f'  [{display_name}] Processing {video_path.name}...')
        raw_encodings = extract_encodings_from_video(video_path)

        if not raw_encodings:
            print(f'  [{display_name}] No clear faces found in video')
            continue

        diverse = diversify_encodings(raw_encodings)
        for enc in diverse:
            known_encodings.append(enc)
            known_names.append(display_name)
            known_ids.append(person_id)

        summary[display_name] = len(diverse)
        print(f'  [{display_name}] {len(raw_encodings)} raw → {len(diverse)} diverse encodings kept')

    # Save database
    with open(ENCODINGS_FILE, 'wb') as f:
        pickle.dump({
            'encodings': known_encodings,
            'names':     known_names,
            'ids':       known_ids,
        }, f)

    total = len(known_names)
    persons = len(summary)
    print(f'\n[ENCODER] Done. {total} encodings for {persons} persons saved to {ENCODINGS_FILE}')
    return known_encodings, known_names, known_ids


if __name__ == '__main__':
    encode_all_persons()