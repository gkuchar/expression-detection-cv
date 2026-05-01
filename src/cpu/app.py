import cv2
import streamlit as st
import av
import os
from PIL import Image
from model import predict_emotion
from backend import draw_emoji, normalize

from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, WebRtcMode

st.set_page_config(page_title="Emotion to Emoji", layout = "wide")
st.title("Live Emotion -> Emoji")

SIZE = 48
FRAME_REFRESH_COUNT = 5
FACE_PADDING = 0.25
BASE = os.path.dirname(os.path.abspath(__file__))

EMOTION_EMOJI = {
    0: Image.open(os.path.join(BASE, "../assets/positive.png")).convert("RGBA"),
    1: Image.open(os.path.join(BASE, "../assets/neutral.png")).convert("RGBA"),
    2: Image.open(os.path.join(BASE, "../assets/negative.png")).convert("RGBA")
}


class VideoProcessor(VideoProcessorBase):
    def __init__(self):
        self.frame_count = 0
        self.current_emoji = EMOTION_EMOJI[1]
        self.face_box = None
        cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        self.face_cascade = cv2.CascadeClassifier(cascade_path)

    def _square_face_box(self, face, frame_shape):
        x, y, w, h = face
        frame_height, frame_width = frame_shape[:2]
        side = int(max(w, h) * (1.0 + FACE_PADDING))
        center_x = x + w // 2
        center_y = y + h // 2

        x1 = max(0, center_x - side // 2)
        y1 = max(0, center_y - side // 2)
        x2 = min(frame_width, x1 + side)
        y2 = min(frame_height, y1 + side)

        if x2 - x1 < side:
            x1 = max(0, x2 - side)
        if y2 - y1 < side:
            y1 = max(0, y2 - side)

        return int(x1), int(y1), int(x2), int(y2)

    def _detect_face_box(self, gray):
        faces = self.face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(80, 80),
        )
        if len(faces) == 0:
            return None

        largest_face = max(faces, key=lambda face: face[2] * face[3])
        return self._square_face_box(largest_face, gray.shape)

    def _preprocess_face(self, gray, face_box):
        x1, y1, x2, y2 = face_box
        face = gray[y1:y2, x1:x2]
        face = cv2.resize(face, (SIZE, SIZE))
        face = normalize(face)
        face = face.astype("float32") / 127.5 - 1.0
        return face.reshape(1, SIZE, SIZE)

    def _draw_face_feedback(self, img):
        if self.face_box is not None:
            x1, y1, x2, y2 = self.face_box
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(
                img,
                "Face Detected:",
                (x1, max(25, y1 - 10)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2,
                cv2.LINE_AA,
            )
            return

        frame_height, frame_width = img.shape[:2]
        side = min(frame_width, frame_height) // 2
        x1 = (frame_width - side) // 2
        y1 = (frame_height - side) // 2
        x2 = x1 + side
        y2 = y1 + side
        cv2.rectangle(img, (x1, y1), (x2, y2), (255, 255, 0), 2)
        cv2.putText(
            img,
            "Move face into this area",
            (x1, max(25, y1 - 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 0),
            2,
            cv2.LINE_AA,
        )

    # recv runs once per frame
    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        # convert frame to NumPy image array
        # bgr24 format: Blue, Green, Red with each getting 8 bits (3 bytes total)
        img = frame.to_ndarray(format="bgr24")
        img = cv2.flip(img, 1)

        self.frame_count += 1
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        self.face_box = self._detect_face_box(gray)

        if self.frame_count % FRAME_REFRESH_COUNT == 0:
            if self.face_box is not None:
                model_input = self._preprocess_face(gray, self.face_box)

                emotion = predict_emotion(model_input) # EMOTION NN CALL

                self.current_emoji = EMOTION_EMOJI[emotion]

        self._draw_face_feedback(img)
        cv2.putText(
            img,
            "Predicted Emoji:",
            (20, 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )
        img = draw_emoji(img, self.current_emoji, (290, 20))
        cv2.putText(
            img,
            "Live Camera",
            (20, img.shape[0] - 20),  # bottom of frame
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

        return av.VideoFrame.from_ndarray(img, format="bgr24")

# set up the webRTC media session
col1, col2, col3 = st.columns([1, 3, 1])
with col2:
    webrtc_streamer(
        key="emotion-camera",
        mode=WebRtcMode.SENDRECV,
        video_processor_factory=VideoProcessor,
        media_stream_constraints={"video": True, "audio": False},
        async_processing=True,
    )