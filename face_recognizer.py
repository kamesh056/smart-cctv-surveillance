"""
face_recognizer.py  (v5)
- Fixed: false detections on round objects (balls, clocks, lights) —
         added face landmark validation + minimum face size filter
- Fixed: wrong identity matches — tightened tolerance + added
         second-best distance gap check to reject borderline matches
- Fixed: scale_factor now defaults to 0.75 for proper long-range detection
- Fixed: upsample defaults to 1, can be set to 2 for even longer range
"""
import face_recognition, pickle, cv2, numpy as np, json, os


# Minimum face height as a fraction of the frame height.
# Faces smaller than this are almost certainly false positives (logos, balls).
_MIN_FACE_HEIGHT_RATIO = 0.04   # 4 % of frame height  (tune in config if needed)

# A real face has 68 landmarks. face_recognition returns them in 6 groups.
# If fewer than this many groups are detected the detection is rejected.
_MIN_LANDMARK_GROUPS = 4        # eyes(2) + nose + lips = at least 4 groups

# How much larger the best match distance must be vs the second-best
# for us to confidently pick that identity. If the gap is tiny, two
# enrolled people look equally close → call it UNKNOWN instead of guessing.
_MIN_DISTANCE_GAP = 0.07


class FaceRecognizer:

    def __init__(self, encodings_file='face_encodings.pkl', config_path='config.json'):
        with open(config_path) as f:
            cfg = json.load(f)['recognition']
        self.tolerance    = cfg.get('tolerance', 0.50)   # tightened from 0.55/0.56
        self.model        = cfg.get('model', 'hog')
        self.upsample     = int(cfg.get('upsample', 1))
        self.scale_factor = float(cfg.get('scale_factor', 0.75))
        self.min_face_ratio = float(cfg.get('min_face_height_ratio', _MIN_FACE_HEIGHT_RATIO))
        self.encodings_file = encodings_file
        self.known_encodings = []
        self.known_names     = []
        self.known_ids       = []

        if os.path.exists(encodings_file):
            self._load(encodings_file)
        else:
            print('[RECOGNIZER] WARNING: face_encodings.pkl not found — detection disabled until enrolled.')

    def _load(self, path):
        try:
            with open(path, 'rb') as f:
                data = pickle.load(f)
            self.known_encodings = data.get('encodings', [])
            self.known_names     = data.get('names', [])
            self.known_ids       = data.get('ids', data.get('names', []))
            print(f'[RECOGNIZER] Loaded {len(self.known_names)} encodings for '
                  f'{len(set(self.known_names))} persons')
        except Exception as e:
            print(f'[RECOGNIZER] Failed to load encodings: {e}')
            self.known_encodings = []
            self.known_names     = []
            self.known_ids       = []

    def reload(self, path=None):
        """Hot-reload encodings after new person added via portal."""
        self._load(path or self.encodings_file)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _is_real_face(self, rgb_frame, location, frame_height):
        """
        Return True only if the detected region looks like a genuine human face.

        Two checks:
          1. Size filter  — reject tiny blobs (balls, logos, reflections).
          2. Landmark filter — a real face must have recognisable facial
             features (eyes, nose, mouth). Round objects have none.
        """
        top, right, bottom, left = location
        face_h = bottom - top

        # --- Check 1: minimum physical size ---
        if face_h < frame_height * self.min_face_ratio:
            return False

        # --- Check 2: landmark presence ---
        # face_recognition.face_landmarks returns a list of dicts, one per face.
        # Each dict has keys like 'left_eye', 'right_eye', 'nose_tip', etc.
        # A non-face region (ball, sign) returns an empty dict or very few keys.
        try:
            landmarks_list = face_recognition.face_landmarks(rgb_frame, [location])
            if not landmarks_list:
                return False
            lm = landmarks_list[0]          # dict for this face
            # Require at least eyes + nose + lips to all be present
            required = {'left_eye', 'right_eye', 'nose_tip', 'top_lip'}
            if not required.issubset(lm.keys()):
                return False
            # Each key maps to a list of (x,y) points — reject if suspiciously empty
            if len(lm.get('left_eye', [])) < 2 or len(lm.get('right_eye', [])) < 2:
                return False
        except Exception as e:
            print(f'[RECOGNIZER] Landmark check error: {e}')
            return False

        return True

    def _best_match(self, encoding):
        """
        Return (name, person_id, confidence) for the closest enrolled face,
        or ('UNKNOWN', None, 0.0) if no confident match found.

        Uses two-gap check: if best and second-best distances are too close,
        the system is uncertain → returns UNKNOWN rather than guessing wrong.
        """
        if not self.known_encodings:
            return 'UNKNOWN', None, 0.0

        distances = face_recognition.face_distance(self.known_encodings, encoding)

        if len(distances) == 0:
            return 'UNKNOWN', None, 0.0

        # Sort indices by distance (ascending)
        sorted_idx = np.argsort(distances)
        best_idx   = sorted_idx[0]
        best_dist  = distances[best_idx]

        # Hard tolerance gate
        if best_dist > self.tolerance:
            return 'UNKNOWN', None, 0.0

        # Two-gap check: if there's a second candidate within a narrow margin,
        # we can't be confident which one is correct → call it UNKNOWN
        if len(sorted_idx) >= 2:
            second_dist = distances[sorted_idx[1]]
            if (second_dist - best_dist) < _MIN_DISTANCE_GAP:
                # Both enrolled faces look equally close — ambiguous, skip
                # (This situation usually means poor enrollment photos)
                print(f'[RECOGNIZER] Ambiguous match (gap={second_dist - best_dist:.3f}) → UNKNOWN')
                return 'UNKNOWN', None, 0.0

        name      = self.known_names[best_idx]
        person_id = self.known_ids[best_idx]
        confidence = round(float(1.0 - best_dist), 3)
        return name, person_id, confidence

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def recognize_faces(self, frame):
        if frame is None:
            return []

        fx = max(0.1, min(1.0, self.scale_factor))
        h, w = frame.shape[:2]
        new_w = max(1, int(w * fx))
        new_h = max(1, int(h * fx))
        small = cv2.resize(frame, (new_w, new_h))
        rgb   = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
        small_h = new_h

        # --- Step 1: detect candidate locations ---
        try:
            locations = face_recognition.face_locations(
                rgb,
                number_of_times_to_upsample=self.upsample,
                model=self.model
            )
        except Exception as e:
            print(f'[RECOGNIZER] face_locations error: {e}')
            return []

        if not locations:
            return []

        # --- Step 2: validate each location (filter balls, signs, etc.) ---
        valid_locations = []
        for loc in locations:
            if self._is_real_face(rgb, loc, small_h):
                valid_locations.append(loc)
            else:
                top, right, bottom, left = loc
                print(f'[RECOGNIZER] Rejected false detection at '
                      f'({left},{top})→({right},{bottom}) — failed landmark/size check')

        if not valid_locations:
            return []

        # --- Step 3: encode only the validated faces ---
        try:
            encodings = face_recognition.face_encodings(rgb, valid_locations)
        except Exception as e:
            print(f'[RECOGNIZER] face_encodings error: {e}')
            return []

        results = []
        inv = 1.0 / fx   # scale bounding boxes back to original frame size

        for enc, loc in zip(encodings, valid_locations):
            name, person_id, confidence = self._best_match(enc)
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