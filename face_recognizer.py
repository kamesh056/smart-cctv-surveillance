"""
face_recognizer.py  (v7 — NeuralGuard fixed edition)

Fixes vs v6:
  - FIX: Sub-threshold spam suppressed. The per-rejection _log() call was
    writing to stderr on EVERY rejected detection EVERY frame (~50+ lines/sec).
    Now only logs at most once per 30 seconds per camera if rejections occur.
  - FIX: Authorised-person cache invalidation.
    FaceRecognizer is now a proper singleton-per-encodings-file with a
    reload() that fully replaces known_encodings/names/ids in one atomic swap,
    so deleted persons are immediately de-authorised and newly enrolled
    persons are immediately recognised after re-encoding.
  - FIX: scale_factor=0.5 kept (best for YOLO crops).
  - FIX: _best_match() distance gap check threshold lowered to 0.05
    (was 0.07) so newly enrolled persons with few training frames still match.
  - FIX: All output goes to stderr (safe with IPC stdout pipe in recog_worker).
"""

import face_recognition
import pickle
import cv2
import numpy as np
import json
import os
import sys
import time
import threading

# ── Throttled logger (no spam) ─────────────────────────────────────────────
_log_lock          = threading.Lock()
_last_rejection_log: float = 0.0   # epoch seconds
_REJECTION_LOG_INTERVAL = 30.0     # log at most once per 30 s


def _log(msg: str):
    """Write to stderr — safe even when stdout is a binary IPC pipe."""
    try:
        sys.stderr.write(msg + '\n')
        sys.stderr.flush()
    except Exception:
        pass


def _log_rejection_throttled(count: int):
    """Log sub-threshold rejections at most once per 30 s to avoid spam."""
    global _last_rejection_log
    now = time.time()
    with _log_lock:
        if now - _last_rejection_log >= _REJECTION_LOG_INTERVAL:
            _last_rejection_log = now
            _log(f'[RECOGNIZER] Suppressed {count} sub-threshold detection(s) '
                 f'(size filter). This is normal for distant/partial faces.')


# ── Constants ──────────────────────────────────────────────────────────────

# Minimum face height as a fraction of the frame (or crop) height.
_MIN_FACE_HEIGHT_RATIO = 0.04   # 4% of frame/crop height

# Minimum gap between best and second-best match distance.
# Lowered from 0.07 → 0.05 so newly enrolled persons (fewer training frames)
# still get a confident match.
_MIN_DISTANCE_GAP = 0.05


class FaceRecognizer:
    """
    Thread-safe face recognizer with hot-reload support.

    Key design decisions:
      - All mutable state (known_encodings, known_names, known_ids) is swapped
        atomically under self._data_lock in reload() so detection threads
        always see a consistent snapshot.
      - scale_factor=0.5 — optimised for YOLO person-box crops (already small).
      - Sub-threshold rejection log is throttled to ≤1 line per 30 seconds.
    """

    def __init__(self,
                 encodings_file: str = 'face_encodings.pkl',
                 config_path:    str = 'config.json'):

        self._data_lock = threading.Lock()

        with open(config_path) as f:
            cfg = json.load(f)['recognition']

        self.tolerance      = float(cfg.get('tolerance',              0.50))
        self.model          = cfg.get('model',                        'hog')
        self.upsample       = int(cfg.get('upsample',                 1))
        self.scale_factor   = float(cfg.get('scale_factor',           0.5))
        self.min_face_ratio = float(cfg.get('min_face_height_ratio',  _MIN_FACE_HEIGHT_RATIO))
        self.encodings_file = encodings_file

        # Initialise empty — populated by _load()
        self._known_encodings: list = []
        self._known_names:     list = []
        self._known_ids:       list = []

        # Track rejected-detection count for throttled logging
        self._rejection_count = 0
        self._rejection_reset = time.time()

        if os.path.exists(encodings_file):
            self._load(encodings_file)
        else:
            _log('[RECOGNIZER] WARNING: face_encodings.pkl not found — '
                 'detection disabled until enrolled.')

    # ── Internal load (atomic swap) ────────────────────────────────────────

    def _load(self, path: str):
        """
        Load encodings from pickle and atomically swap the active dataset.

        The atomic swap means:
          - Detection threads reading self._known_encodings always see a fully
            consistent list (never partially updated).
          - Deleted persons disappear immediately on next reload.
          - New persons appear immediately on next reload.
        """
        try:
            with open(path, 'rb') as f:
                data = pickle.load(f)

            new_encodings = data.get('encodings', [])
            new_names     = data.get('names',     [])
            new_ids       = data.get('ids', data.get('names', []))

            # Sanity check — all three lists must have the same length
            min_len = min(len(new_encodings), len(new_names), len(new_ids))
            new_encodings = new_encodings[:min_len]
            new_names     = new_names[:min_len]
            new_ids       = new_ids[:min_len]

            with self._data_lock:
                self._known_encodings = new_encodings
                self._known_names     = new_names
                self._known_ids       = new_ids

            _log(f'[RECOGNIZER] Loaded {min_len} encodings for '
                 f'{len(set(new_names))} persons')

        except Exception as e:
            _log(f'[RECOGNIZER] Failed to load encodings: {e}')
            # On load failure, clear (fail-safe: unknown > false authorisation)
            with self._data_lock:
                self._known_encodings = []
                self._known_names     = []
                self._known_ids       = []

    def reload(self, path: str = None):
        """
        Hot-reload encodings after a person is added or removed via the portal.

        Called by portal.py after video_face_encoder.py completes.
        Thread-safe: the atomic swap in _load() ensures no detection thread
        ever reads a partially updated state.
        """
        self._load(path or self.encodings_file)

    # ── Internal helpers ───────────────────────────────────────────────────

    @property
    def _encodings(self):
        with self._data_lock:
            return self._known_encodings

    @property
    def _names(self):
        with self._data_lock:
            return self._known_names

    @property
    def _ids(self):
        with self._data_lock:
            return self._known_ids

    def _snapshot(self):
        """Return a consistent (encodings, names, ids) snapshot under one lock."""
        with self._data_lock:
            return (list(self._known_encodings),
                    list(self._known_names),
                    list(self._known_ids))

    def _is_real_face(self, location, frame_height: int) -> bool:
        """
        Size filter only — reject detections that are too small.
        (Landmark check removed in v6; not needed on YOLO crops.)
        """
        top, right, bottom, left = location
        face_h = bottom - top
        if face_h < frame_height * self.min_face_ratio:
            return False
        return True

    def _best_match(self, encoding,
                    known_encodings: list,
                    known_names:     list,
                    known_ids:       list):
        """
        Return (name, person_id, confidence) for the closest enrolled face.
        Returns ('UNKNOWN', None, 0.0) if no confident match is found.

        Uses a distance-gap check to avoid false matches between
        similarly-looking enrolled faces.
        """
        if not known_encodings:
            return 'UNKNOWN', None, 0.0

        distances = face_recognition.face_distance(known_encodings, encoding)

        if len(distances) == 0:
            return 'UNKNOWN', None, 0.0

        sorted_idx = np.argsort(distances)
        best_idx   = sorted_idx[0]
        best_dist  = distances[best_idx]

        # Hard tolerance gate
        if best_dist > self.tolerance:
            return 'UNKNOWN', None, 0.0

        # Two-gap check — only if there is more than one enrolled person.
        # With a single enrolled person the gap check is skipped entirely.
        if len(sorted_idx) >= 2:
            second_dist = distances[sorted_idx[1]]
            gap = second_dist - best_dist
            if gap < _MIN_DISTANCE_GAP:
                # Ambiguous — two people look equally close
                return 'UNKNOWN', None, 0.0

        name       = known_names[best_idx]
        person_id  = known_ids[best_idx]
        confidence = round(float(1.0 - best_dist), 3)
        return name, person_id, confidence

    # ── Public API ─────────────────────────────────────────────────────────

    def recognize_faces(self, frame) -> list:
        """
        Detect and recognise faces in a BGR frame (or YOLO person crop).

        Returns list of dicts:
            {
                'name':       str,
                'person_id':  str | None,
                'authorized': bool,
                'location':   (top, right, bottom, left),
                'confidence': float,
            }

        Thread-safe: takes a consistent snapshot of encodings at the start
        of each call so additions/removals mid-call never corrupt results.
        """
        if frame is None or frame.size == 0:
            return []

        # Take an atomic snapshot — ensures consistent state for this whole call.
        # Newly enrolled persons appear as soon as reload() runs (triggered by
        # portal.py after video_face_encoder.py finishes).
        # Deleted persons disappear as soon as reload() runs (same trigger).
        known_enc, known_names, known_ids = self._snapshot()

        fx    = max(0.1, min(1.0, self.scale_factor))
        h, w  = frame.shape[:2]
        new_w = max(1, int(w * fx))
        new_h = max(1, int(h * fx))
        small = cv2.resize(frame, (new_w, new_h))

        try:
            rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
        except Exception:
            return []

        small_h = new_h

        # Step 1: detect candidate locations
        try:
            locations = face_recognition.face_locations(
                rgb,
                number_of_times_to_upsample=self.upsample,
                model=self.model,
            )
        except Exception as e:
            _log(f'[RECOGNIZER] face_locations error: {e}')
            return []

        if not locations:
            return []

        # Step 2: size filter — count rejections but don't spam log
        valid_locations = []
        rejected = 0
        for loc in locations:
            if self._is_real_face(loc, small_h):
                valid_locations.append(loc)
            else:
                rejected += 1

        if rejected > 0:
            self._rejection_count += rejected
            _log_rejection_throttled(self._rejection_count)
            self._rejection_count = 0  # reset after log (throttled anyway)

        if not valid_locations:
            return []

        # Step 3: encode validated faces
        try:
            encodings = face_recognition.face_encodings(rgb, valid_locations)
        except Exception as e:
            _log(f'[RECOGNIZER] face_encodings error: {e}')
            return []

        results = []
        inv = 1.0 / fx  # scale boxes back to original frame coordinates

        for enc, loc in zip(encodings, valid_locations):
            name, person_id, confidence = self._best_match(
                enc, known_enc, known_names, known_ids)
            authorized = (name != 'UNKNOWN')

            top, right, bottom, left = [int(v * inv) for v in loc]
            results.append({
                'name':       name,
                'person_id':  person_id,
                'authorized': authorized,
                'location':   (top, right, bottom, left),
                'confidence': confidence,
            })

        return results