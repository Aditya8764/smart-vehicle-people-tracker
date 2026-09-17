# Real-Time Object Counter & Tracker

Welcome to the **Real-Time Object Counter & Tracker**, a robust and interactive computer vision application designed to monitor, track, and analyze object movement across designated zones. 

Built entirely in Python, this application leverages the power of **Streamlit** for a seamless, browser-based user interface, **Ultralytics YOLOv8** for state-of-the-art, high-speed object detection, and the **Supervision** library for advanced object tracking and spatial logic. 

Whether you are building a smart parking system to count vehicle entry and exit, analyzing foot traffic at a retail storefront, or simply experimenting with real-time video analytics, this application provides an out-of-the-box solution. It allows users to dynamically draw virtual boundaries across live webcam feeds or uploaded video files, accurately tracking specific object classes (such as people, cars, or trucks) and maintaining a persistent, timestamped log of crossing events without recounting the same object twice.

## Features

- **Live Video Processing**: Supports both uploaded video files (mp4/mov) and live webcam feeds.
- **YOLOv8 Detection**: High-performance, frame-by-frame object detection using pretrained YOLOv8 models.
- **Persistent Tracking**: Utilizes the ByteTrack algorithm (via the `supervision` library) to assign persistent IDs to objects and track them across frames, ensuring objects are not recounted.
- **Line-Crossing Counters**: Set a custom horizontal or vertical line. The app independently tracks objects entering ("In") and exiting ("Out") across the line.
- **Interactive UI**: Built with Streamlit for a clean, intuitive web dashboard. Adjust confidence thresholds and select specific classes to track on the fly.
- **Data Export**: Logs every crossing event with a timestamp and direction, and allows you to export this data directly to a CSV file.

## Tech Stack

- **Python**: Core programming language.
- **[Streamlit](https://streamlit.io/)**: For the interactive web application interface.
- **[Ultralytics](https://github.com/ultralytics/ultralytics)**: For loading the YOLOv8 model and running object detection.
- **[Supervision](https://github.com/roboflow/supervision)**: Provides the `ByteTrack` tracker and `LineZone` counting utilities, as well as powerful drawing annotators.
- **[OpenCV (cv2)](https://opencv.org/)**: For robust video reading and frame manipulation.
- **Pandas**: For structuring and exporting the logged event data.

## Installation

1. **Clone or Download the Repository** to your local machine.

2. **Navigate to the Project Directory**:
   ```bash
   cd path/to/smart-vehicle-people-tracker
   ```

3. **Set Up a Virtual Environment (Recommended)**:
   ```bash
   python -m venv venv
   # On Windows:
   venv\Scripts\activate
   # On Mac/Linux:
   source venv/bin/activate
   ```

4. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```
   *(Note: The first time you run the app, Ultralytics will automatically download the lightweight `yolov8n.pt` weights file.)*

## Usage

1. **Start the Application**:
   Run the following command in your terminal:
   ```bash
   streamlit run app.py
   ```
   This will launch the app in your default web browser (typically at `http://localhost:8501`).

2. **Configure the App**:
   - **Input Source**: Choose between uploading a video file or using your webcam.
   - **Select Classes**: Use the multi-select dropdown to pick which objects to track (e.g., `person`, `car`, `truck`).
   - **Confidence Threshold**: Adjust the slider to filter out low-confidence detections.
   - **Line Placement**: Choose horizontal or vertical orientation and use the slider to position the counting line across the frame.

3. **Start Counting**:
   - Click **Start** to begin processing the video.
   - The annotated video will display bounding boxes, tracking traces, labels, and the counting line. The "In" and "Out" counts will update dynamically.
   - Click **Stop Video** to pause processing.

4. **Export Data**:
   - Once processing is stopped, scroll to the bottom of the page to view a preview of the crossing logs.
   - Click **Download log as CSV** to save the data for external analysis.

## Output & Demonstration

When the application is running successfully, you can expect the following output and capabilities:

![Live Tracking Demonstration](Screenshot/Screenshot%202026-09-17%20215859%20-%20Copy.png)

- **Live Annotations**: The video feed will display real-time bounding boxes around detected objects (e.g., cars, people) with their assigned tracking IDs and confidence scores.
- **Dynamic Tracking Lines**: A visible horizontal or vertical line will be overlaid on the video. As an object's center point crosses this line, the corresponding counter increments.
- **Directional Counting**: The app distinctly counts objects based on their direction of travel relative to the line. 
  - *Note: If all objects (e.g., cars on a divided highway) are moving in the same direction, only one counter (either "In" or "Out") will increment.*

![Data Logging Demonstration](Screenshot/Screenshot%202026-09-17%20215920%20-%20Copy.png)

- **Data Logging**: Every crossing event is recorded in a data table at the bottom of the app, detailing the exact timestamp and the direction of the crossing. This table can be instantly exported to CSV for further analysis.

## Project Structure

```text
├── app.py             # Main Streamlit application and UI logic
├── counter.py         # Wraps Supervision's LineZone for line crossing logic
├── detector.py        # Wraps Ultralytics YOLOv8 for object detection
├── tracker.py         # Wraps Supervision's ByteTrack for object tracking
├── utils.py           # Helper functions for drawing bounding boxes, lines, and annotations
├── requirements.txt   # List of Python package dependencies
└── README.md          # Project documentation
```

## Troubleshooting

- **Webcam not working**: Ensure your browser has permission to access the webcam, and no other application (like Zoom or Teams) is currently using the camera. 
- **Performance issues**: Processing high-resolution video can be intensive. If the app feels sluggish, it is recommended to run it on a machine with a dedicated GPU.
- **Supervision Library Errors**: The `supervision` library updates frequently. If you see errors regarding `scene` or `frame` arguments in annotators, ensure your library version matches the syntax in `utils.py`.
