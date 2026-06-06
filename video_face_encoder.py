"""
video_face_encoder.py
Extracts face encodings from a short enrollment video (person turning face 360°).
Samples frames evenly, filters blurry ones, picks the best diverse encodings.
"""
import cv2, face_recognition, pickle, os, numpy as np
from pathlib import Path

PERSONS_DIR    = Path('authorized_persons')
ENCODINGS_FILE = 'face_encodings.pkl'
SAMPLE_FRAMES  = 40      # How many frames to sample from video
BLUR_THRESHOLD = 80.0    # Laplacian variance; below this = too blurry, skip
MAX_ENC_PER_PERSON = 12  # Store up to 12 diverse encodings per person


def is_blurry(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return cv2.Laplacian(gray, cv2.CV_64F).var() < BLUR_THRESHOLD


def extract_encodings_from_video(video_path, sample_count=SAMPLE_FRAMES):
    """Sample frames from video, return list of face encodings."""
    cap = cv2.VideoCapture(str(video_path))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total == 0:
        cap.release()
        return []

    indices = np.linspace(0, total - 1, min(sample_count, total), dtype=int)
    encodings = []

    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ret, frame = cap.read()
        if not ret:
            continue
        if is_blurry(frame):
            continue

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        locs = face_recognition.face_locations(rgb, model='hog')
        encs = face_recognition.face_encodings(rgb, locs)
        if encs:
            encodings.append(encs[0])   # Take first (largest) face

    cap.release()
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
