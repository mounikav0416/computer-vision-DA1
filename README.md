An interactive Streamlit-based computer vision application that detects object boundaries, analyzes geometric shapes, and extracts features such as area and perimeter using classical image processing techniques.

📌 Project Overview

This project implements a contour-based object detection system that processes an input image through multiple image processing stages including grayscale conversion, edge detection, morphological operations, and contour analysis. The application visualizes each processing step and presents detection results through an interactive dashboard.

The project is designed for educational and analytical purposes, demonstrating how traditional computer vision methods can be applied to real-world images without deep learning models.

✨ Features

Upload and analyze images through an interactive UI

Visualize complete image processing pipeline

Detect multiple objects in a single image

Extract object area and perimeter

Classify objects based on shape and size

Display bounding boxes and contour outlines

View results in tabular format

Download detection results as CSV

Adjustable minimum area threshold

🛠️ Methods Used

Grayscale Conversion

Gaussian Blurring

Canny Edge Detection

Morphological Closing

Contour Detection

Contour Approximation

Feature Extraction (Area & Perimeter)

Shape Classification

Size Classification

🧰 Technologies Used

Python 3.11

OpenCV

NumPy

Pandas

Streamlit

📂 Project Structure
📁 Shape-Object-Boundary-Analyzer
│
├── app.py                 # Main Streamlit application
├── requirements.txt       # Required Python libraries
├── README.md              # Project documentation

📦 Installation
1️⃣ Clone the repository
git clone https://github.com/your-username/shape-object-boundary-analyzer.git
cd shape-object-boundary-analyzer

2️⃣ Create virtual environment (recommended)
python -m venv venv
venv\Scripts\activate

3️⃣ Install dependencies
pip install -r requirements.txt

▶️ Run the Application
streamlit run app.py


The app will open in your browser at:

http://localhost:8501

📊 Output Description

The application displays:

Original image

Grayscale image

Blurred image

Edge-detected image

Morphologically processed image

Detected object boundaries with labels

Object count, average area, and perimeter

Detailed results table with CSV download option

📈 Applications

Object boundary detection

Image feature extraction

Educational demonstrations of computer vision

Preprocessing for advanced vision systems

⚠️ Limitations

Cannot assign real-world semantic object names

Performance depends on image quality

Overlapping objects may merge into single contours

🚀 Future Enhancements

Real-time webcam detection

AI-based object recognition

Adaptive thresholding

Improved shape classification

Deployment on Streamlit Cloud

👨‍💻 Author

Mounika V
Computer Vision Project – Streamlit & OpenCV

📜 License

This project is for educational and academic use.
