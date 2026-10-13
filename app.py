import streamlit as st
from ultralytics import YOLO
from PIL import Image
import os

st.set_page_config(page_title="YOLOv8 Image Tester", page_icon="🔎", layout="wide")

st.title("YOLOv8 Image Tester")
st.write("Upload an image to run inference using the model in this repository.")

MODEL_PATH = os.path.join(os.path.dirname(__file__), "best.pt")

@st.cache_resource
def load_model(model_path):
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            "Model file not found. Add your trained YOLOv8 weights to the repository "
            "with the filename best.pt."
        )
    return YOLO(model_path)

with st.sidebar:
    st.header("Detection settings")
    confidence = st.slider("Confidence threshold", 0.05, 0.95, 0.25, 0.05)
    image_size = st.select_slider("Inference image size", options=[320, 416, 512, 640, 768, 960], value=640)

uploaded_file = st.file_uploader(
    "Upload an image",
    type=["jpg", "jpeg", "png", "webp", "bmp"]
)

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.subheader("Input image")
    st.image(image, use_container_width=True)

    if st.button("Run detection", type="primary"):
        try:
            with st.spinner("Running YOLOv8 inference..."):
                model = load_model(MODEL_PATH)
                result = model.predict(
                    source=image,
                    conf=confidence,
                    imgsz=image_size,
                    verbose=False
                )[0]

            annotated_rgb = result.plot()[:, :, ::-1]
            left, right = st.columns(2)
            with left:
                st.subheader("Detection output")
                st.image(annotated_rgb, use_container_width=True)
            with right:
                st.subheader("Summary")
                st.metric("Objects detected", len(result.boxes))

                rows = []
                for box in result.boxes:
                    class_id = int(box.cls.item())
                    rows.append({
                        "Class": model.names[class_id],
                        "Confidence": round(float(box.conf.item()), 4),
                        "X1": round(float(box.xyxy[0][0].item()), 1),
                        "Y1": round(float(box.xyxy[0][1].item()), 1),
                        "X2": round(float(box.xyxy[0][2].item()), 1),
                        "Y2": round(float(box.xyxy[0][3].item()), 1),
                    })

                if rows:
                    st.dataframe(rows, use_container_width=True)
                    csv_data = "Class,Confidence,X1,Y1,X2,Y2\n" + "\n".join(
                        f'{r["Class"]},{r["Confidence"]},{r["X1"]},{r["Y1"]},{r["X2"]},{r["Y2"]}'
                        for r in rows
                    )
                    st.download_button(
                        "Download detections as CSV",
                        data=csv_data,
                        file_name="detections.csv",
                        mime="text/csv"
                    )
                else:
                    st.info("No objects detected above the selected confidence threshold.")

        except Exception as exc:
            st.error(f"Could not run the model: {exc}")
            st.caption("Check that best.pt is a compatible Ultralytics YOLO model and that dependencies installed successfully.")
else:
    st.info("Choose an image above to begin.")
