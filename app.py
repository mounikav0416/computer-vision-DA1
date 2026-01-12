import streamlit as st
import cv2
import numpy as np
import pandas as pd
from PIL import Image

# --------------------------------------------------
# Page Configuration
# --------------------------------------------------
st.set_page_config(
    page_title="Shape & Object Boundary Analyzer",
    layout="wide"
)

st.title("🔍 Shape & Object Boundary Analyzer")
st.caption(
    "Contour-based object detection with visualization of image processing stages"
)

# --------------------------------------------------
# Sidebar Controls
# --------------------------------------------------
st.sidebar.header("⚙️ Controls")

uploaded_file = st.sidebar.file_uploader(
    "Upload an Image",
    type=["jpg", "jpeg", "png"]
)

min_area = st.sidebar.slider(
    "Minimum Area Threshold",
    min_value=50,
    max_value=3000,
    value=200,
    step=50
)

# --------------------------------------------------
# Main Logic
# --------------------------------------------------
if uploaded_file is not None:
    # Load image
    image = Image.open(uploaded_file)
    image_np = np.array(image)

    # --------------------------------------------------
    # Image Processing Pipeline
    # --------------------------------------------------
    gray = cv2.cvtColor(image_np, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)

    # Edge detection
    edges = cv2.Canny(blur, 50, 150)

    # Morphological closing
    kernel = np.ones((3, 3), np.uint8)
    closed = cv2.morphologyEx(
        edges, cv2.MORPH_CLOSE, kernel, iterations=2
    )

    # Find contours
    contours, _ = cv2.findContours(
        closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    output = image_np.copy()
    results = []

    # --------------------------------------------------
    # Object Detection & Annotation
    # --------------------------------------------------
    object_id = 1
    for cnt in contours:
        area = cv2.contourArea(cnt)

        if area > min_area:
            perimeter = cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, 0.02 * perimeter, True)

            # Shape classification
            if len(approx) == 3:
                shape = "Triangle"
            elif len(approx) == 4:
                shape = "Rectangle"
            elif len(approx) > 4:
                shape = "Circular / Irregular"
            else:
                shape = "Unknown"

            # Size classification
            if area < 800:
                size = "Small Object"
            elif area < 3000:
                size = "Medium Object"
            else:
                size = "Large Object"

            x, y, w, h = cv2.boundingRect(approx)

            # Draw bounding box
            cv2.rectangle(
                output, (x, y), (x + w, y + h), (0, 255, 0), 2
            )

            # Draw contour
            cv2.drawContours(
                output, [approx], -1, (255, 0, 0), 2
            )

            # Label text
            label = f"ID {object_id}: {shape}, {size}"

            cv2.putText(
                output,
                label,
                (x, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 0, 255),
                2,
                cv2.LINE_AA
            )

            results.append({
                "Object ID": object_id,
                "Shape": shape,
                "Size Category": size,
                "Area": round(area, 2),
                "Perimeter": round(perimeter, 2)
            })

            object_id += 1

    # --------------------------------------------------
    # Display Image Processing Stages
    # --------------------------------------------------
    st.subheader("🧪 Image Processing Pipeline")

    col1, col2 = st.columns(2)
    with col1:
        st.image(image_np, caption="Original Image", use_container_width=True)
        st.image(gray, caption="Grayscale Image", use_container_width=True)
        st.image(blur, caption="Blurred Image", use_container_width=True)

    with col2:
        st.image(edges, caption="Edge Detection (Canny)", use_container_width=True)
        st.image(closed, caption="Morphological Closing", use_container_width=True)
        st.image(output, caption="Detected Object Boundaries & Labels", use_container_width=True)

    # --------------------------------------------------
    # Results Table & Metrics
    # --------------------------------------------------
    if results:
        df = pd.DataFrame(results)

        st.subheader("📊 Object Detection Results")

        m1, m2, m3 = st.columns(3)
        m1.metric("Total Objects Detected", len(df))
        m2.metric("Average Area", round(df["Area"].mean(), 2))
        m3.metric("Average Perimeter", round(df["Perimeter"].mean(), 2))

        st.dataframe(df, use_container_width=True)

        st.download_button(
            label="⬇️ Download Results (CSV)",
            data=df.to_csv(index=False),
            file_name="object_detection_results.csv",
            mime="text/csv"
        )

        st.success(
            "Objects detected successfully using edge-based contour analysis."
        )
    else:
        st.warning(
            "No objects detected. Try lowering the minimum area threshold."
        )

else:
    st.info("👈 Upload an image from the sidebar to start detection.")
