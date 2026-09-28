# ☀️ SolarGuard — Solar Panel Dust & Crack Checker

A computer-vision dashboard that inspects solar panel images and instantly reports:

- **Soiling level (%)** — how much dust is likely covering the panel, estimated using HSV color-space analysis
- **Crack detection** — flags panels that may have surface cracks, using Canny edge-density analysis
- **Health Score** — a single 0–100 gauge combining soiling and crack confidence
- **Session history** — track every image checked in one run
- **PDF report export** — download a summary of all checked panels

Built with **Python, OpenCV, Plotly, and Streamlit** — no pretrained ML model needed, so it installs and runs in minutes.

## 🖼️ Screenshots

**Dashboard home**

![Dashboard home](screenshots/dashboard-home.png)

**Inspection result — health gauge, soiling %, crack confidence**

![Inspection result](screenshots/inspection-result.png)

## ⚙️ How It Works

| Step | Technique |
|------|-----------|
| Soiling estimation | HSV saturation/brightness analysis — dustier surfaces are lighter and less saturated |
| Crack detection | Canny edge detection — high edge density suggests irregular surface damage |
| Health Score | Weighted combination of soiling % and crack confidence, shown as a gauge |
| Report generation | fpdf2 — compiles session results into a downloadable PDF |

## 🚀 Getting Started

```bash
git clone <your-repo-url>
cd SolarProject
pip install -r requirements.txt
streamlit run app.py
```

Then open `http://localhost:8501` in your browser.

## 🌐 Deploy It (for your portfolio link)

The easiest way to get a live link to put on your resume/LinkedIn:

1. Push this project to a public GitHub repo
2. Go to [share.streamlit.io](https://share.streamlit.io) and sign in with GitHub
3. Select the repo, set the main file to `app.py`, and deploy
4. You'll get a free public URL like `https://your-app.streamlit.app`

## 📁 Project Structure

```
SolarProject/
├── app.py                 # Main Streamlit dashboard
├── requirements.txt        # Python dependencies
├── screenshots/             # App screenshots (for this README)
└── README.md
```

## 🛣️ Possible Improvements

- Swap the classical CV heuristics for a trained YOLOv8 model for higher accuracy
- Add thermal-camera support for hotspot detection
- Store results in a database instead of session memory
- Add a cost/ROI calculator for cleaning scheduling

## 📄 License

MIT — free to use and modify.
