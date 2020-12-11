import net
import cv2

# HIST is the number of past faces to remember.
# At 30 fps 30 faces would be one second of recording.
HIST = 256
RED = (0, 0, 255)
GREEN = (0, 255, 0)
THICKNESS = 2

def predict(eye):
    """Returns False if the input eye image is closed. True if open."""

    return net.classify(eye)

class Eye:
    """Represents a detected eyes's state"""

    def __init__(self, opened, coords):
        self.opened = opened
        self.coords = coords

class Face:
    """Represents a detected face's state"""

    def __init__(self, eyes, coords):
        self.eyes = eyes
        self.past = []
        self.coords = coords

class Frame:
    """Represents a single frame's state"""

    def __init__(self, data, coords, past):
        self.coords = coords
        self.data = cv2.resize(data, self.coords, fx=0.6, fy=0.6)
        self.gray = cv2.cvtColor(self.data, cv2.COLOR_BGR2GRAY)
        self.faces = []
        self.past = past

    def carve_eyes(self):
        """Returns the data of just the eyes from the frame."""

        eyes = []
        for face in self.faces:
            x, y, w, h = face.coords
            left_face = self.data[y:y+h, x+int(w/2):x+w]
            right_face = self.data[y:y+h, x:x+int(w/2)]

            feyes = [[], []]
            for i, eye in enumerate(face.eyes):
                if eye == None: continue
                side = left_face if i else right_face
                ex, ey, ew, eh = eye.coords
                feyes[i] = side[ey:ey+eh, ex:ex+ew]

            eyes.append(tuple(feyes))

        return eyes

    def render(self):
        """Returns a frame with rendered shapes on it."""

        draw = self.data.copy()
        for face in self.faces:
            x, y, w, h = face.coords
            cv2.rectangle(draw, (x, y), (x+w, y+h), GREEN, THICKNESS)
            left_face = draw[y:y+h, x+int(w/2):x+w]
            right_face = draw[y:y+h, x:x+int(w/2)]

            for i, eye in enumerate(face.eyes):
                if eye == None: continue
                color = GREEN if eye.opened else RED
                side = left_face if i else right_face
                ex, ey, ew, eh = eye.coords
                cv2.rectangle(side, (ex, ey), (ex+ew, ey+eh), color, THICKNESS)

        return draw

    def detect(self, face_detector, open_eyes_detector, left_eye_detector, right_eye_detector):
        """Detect faces and eyes within the frame."""

        # Detect faces
        faces = face_detector.detectMultiScale(
            self.gray,
            scaleFactor=1.2,
            minNeighbors=5,
            minSize=(50, 50),
            flags=cv2.CASCADE_SCALE_IMAGE
        )

        # Iterate coordinates of each detected face
        for (x, y, w, h) in faces:

            # Extract just the face from the frame
            face = self.data[y:y+h, x:x+w]
            gray_face = self.gray[y:y+h, x:x+w]

            # Separate the face into left and right sides
            left_face = self.data[y:y+h, x+int(w/2):x+w]
            left_face_gray = self.gray[y:y+h, x+int(w/2):x+w]
            right_face = self.data[y:y+h, x:x+int(w/2)]
            right_face_gray = self.gray[y:y+h, x:x+int(w/2)]

            # Detect the left eye
            left_eye = left_eye_detector.detectMultiScale(
                left_face_gray,
                scaleFactor=1.1,
                minNeighbors=5,
                minSize=(30, 30),
                flags = cv2.CASCADE_SCALE_IMAGE
            )

            # Detect the right eye
            right_eye = right_eye_detector.detectMultiScale(
                right_face_gray,
                scaleFactor=1.1,
                minNeighbors=5,
                minSize=(30, 30),
                flags = cv2.CASCADE_SCALE_IMAGE
            )

            # Check if each eye is closed or not and draw a rectangle of the proper color
            re, le = [], []
            for (ex, ey, ew, eh) in right_eye:
                re.append(Eye(predict(right_face[ey:ey+eh, ex:ex+ew]), (ex, ey, ew, eh)))
            for (ex, ey, ew, eh) in left_eye:
                le.append(Eye(predict(left_face[ey:ey+eh, ex:ex+ew]), (ex, ey, ew, eh)))

            if len(re) < 1: re.append(None)
            if len(le) < 1: le.append(None)

            self.faces.append(Face((re[0], le[0]), (x, y, w, h)))
      
        if len(self.faces) == 2:
            nf = []
            for face in self.faces:
                if face.eyes[0] != None or face.eyes[1] != None:
                    nf.append(face)
            self.faces = nf

        self.past.append(self.faces)
        self.past = self.past[-HIST:]
