# CNN Emotion Detection

**Authors:** Spencer Scherger & Griffin Kuchar

A real-time facial emotion detection application that maps a live webcam feed to an emoji representing the user's current emotion. A custom CNN is trained on a labeled facial emotion dataset and classifies faces into three categories: positive, neutral, and negative. The frontend is built with Streamlit and processes the live video feed frame-by-frame, overlaying the predicted emoji in real time.

---

## Setup & Run

**1. Clone the repository:**
```bash
git clone https://github.com/TCU-COSC-MCS-GPU/final-project-griffin-and-spencer.git
cd final-project-griffin-and-spencer
```

**2. Create a virtual environment:**
```bash
python -m venv venv
```

**3. Activate the virtual environment:**
- Mac/Linux: `source venv/bin/activate`
- Windows (PowerShell): `.\venv\Scripts\Activate.ps1`

**4. Install dependencies:**
```bash
pip install -r requirements.txt
```

**5. Run the application:**
```bash
streamlit run src/cpu/app.py
```

From the web app frontend, select a device to receive live video feed input and start making some faces!