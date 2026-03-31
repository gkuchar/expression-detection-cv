import cv2
import streamlit as st
import av

from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, WebRtcMode

st.set_page_config(page_title="Emotion to Emoji", layout = "wide")
st.title("Live Emotion -> Emoji")
st.write("CPU Implementation")

EMOJI_PLACEHOLDER = "🙂"

class VideoProcessor(VideoProcessorBase):
    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        img = frame.to_ndarray(format="bgr24")

        img = cv2.flip(img, 1)

        cv2.putText(
            img,
            f"Predicted Emoji: {EMOJI_PLACEHOLDER}",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )

        cv2.putText(
            img,
            "CPU Live Camera Prototype",
            (20, 80),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

        return av.VideoFrame.from_ndarray(img, format="bgr24")

webrtc_streamer(
    key="emotion-camera",
    mode=WebRtcMode.SENDRECV,
    video_processor_factory=VideoProcessor,
    media_stream_constraints={"video": True, "audio": False},
    async_processing=True,
)