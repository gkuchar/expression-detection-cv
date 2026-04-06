import cv2
import streamlit as st
import av
from model import predict_emotion
from backend import draw_emoji, normalize

from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, WebRtcMode

st.set_page_config(page_title="Emotion to Emoji", layout = "wide")
st.title("Live Emotion -> Emoji")
st.write("CPU Implementation")

SIZE = 128
FRAME_REFRESH_COUNT = 5

EMOTION_EMOJI = {
    0: "😠",
    1: "🤢",
    2: "😑",
    3: "😀",
    4: "😨",
    5: "🙁",
    6: "😮",
    7: "😐",
}


class VideoProcessor(VideoProcessorBase):
    def __init__(self):
        self.frame_count = 0
        self.current_emoji = "😐"

    # recv runs once per frame
    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        # convert frame to NumPy image array
        # bgr24 format: Blue, Green, Red with each getting 8 bits (3 bytes total)
        img = frame.to_ndarray(format="bgr24")
        img = cv2.flip(img, 1)

        self.frame_count += 1
        if self.frame_count % FRAME_REFRESH_COUNT == 0:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            gray = cv2.resize(gray, (SIZE, SIZE))
            gray = normalize(gray)
            
            emotion = predict_emotion(gray)  # NN MODEL API CALL
            self.current_emoji = EMOTION_EMOJI[emotion]

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
            "CPU Live Camera Prototype",
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