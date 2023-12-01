import datetime
from ultralytics import YOLO
import cv2
from helper import create_video_writer
from deep_sort_realtime.deepsort_tracker import DeepSort
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict
import streamlit as st
from helper import create_video_writer
from helper import create_video_writer
import base64
import tempfile
import uuid


# Constants
CONFIDENCE_THRESHOLD = 0.8
MODEL_PATH = "yolov8x.pt"
MAX_AGE = 50
COLORS = [(np.random.rand(), np.random.rand(), np.random.rand()) for _ in range(1000)]

# RTSP URLs or Video Paths
CAM1_URL = "rtsp://your_camera_1_url"
CAM2_URL = "rtsp://your_camera_2_url"


# Initialize the model and tracker globally
model = YOLO(MODEL_PATH)
tracker = DeepSort(max_age=MAX_AGE)


def setup_ui():
    st.title('Object Detection and Tracking')
    col1, col2 = st.columns(2)
    return col1, col2


def get_video_capture(rtsp_url):
    """
    Creates a video capture object from a hardcoded RTSP URL.

    Args:
        rtsp_url (str): The RTSP URL to capture the video stream from.

    Returns:
        cv2.VideoCapture: An OpenCV video capture object.
    """
    return cv2.VideoCapture(rtsp_url)


def detect_objects_and_track(frame, model, tracker, class_count, confidence_threshold):
    """
    Detects objects in a frame using a YOLO model and tracks them using a tracker.

    Args:
        frame (numpy.ndarray): The current video frame.
        model (YOLO): The YOLO object detection model.
        tracker (DeepSort): The object tracker.
        class_count (defaultdict): A dictionary to keep track of class counts.
        confidence_threshold (float): The confidence threshold for detection.

    Returns:
        tuple: A tuple containing the updated frame with tracking, the results, and the updated class count.
    """
    # Perform detection
    detections = model(frame)[0]

    # Initialize results list for tracking
    results = []

    for data in detections.boxes.data.tolist():
        confidence = data[4]
        if float(confidence) < confidence_threshold:
            continue

        xmin, ymin, xmax, ymax = map(int, data[:4])
        class_id = int(data[5])
        results.append([[xmin, ymin, xmax - xmin, ymax - ymin], confidence, class_id])

        # Draw bounding box and label on the frame
        class_name = model.names[class_id]
        cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
        cv2.putText(frame, f"{class_name} {confidence:.2f}", (xmin, ymin - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        # Normalize class name (handle plurals)
        normalized_class_name = class_name.rstrip('s')
        class_count[normalized_class_name] += 1  # Increment class count

    # Update the tracks with the new frame and detections
    tracks = tracker.update_tracks(results, frame=frame)

    return frame, results, class_count



def process_frame(video_cap, model, tracker, class_count, CONFIDENCE_THRESHOLD):
    ret, frame = video_cap.read()
    if not ret:
        return None, None, class_count  # If frame is not read correctly

    results, obj_count, class_count = detect_objects_and_track(frame, model, tracker, class_count, CONFIDENCE_THRESHOLD)
    return frame, results, class_count



def update_time_series_data(class_name, class_count, class_time_series, time_list, count_list, start, obj_count):
    """
    Updates the time series data for object detection counts.

    Args:
        class_name (str): The name of the class.
        class_count (defaultdict[int]): The count of detected objects per class.
        class_time_series (defaultdict[list]): The historical time series data for each class.
        time_list (list[datetime]): The list of timestamps for each frame processed.
        count_list (list[int]): The list of total object counts per frame.
        start (datetime): The timestamp when the current frame was processed.
        obj_count (int): The total object count for the current frame.

    Returns:
        None: This function updates the time series data in-place.
    """
    # Update the historical time series data for each class
    for cls, count in class_count.items():
        if cls not in class_time_series:
            class_time_series[cls] = []
        class_time_series[cls].append(count)

    # Update the time list with the start time of the current frame
    time_list.append(start)

    # Update the count list with the object count of the current frame
    count_list.append(obj_count)

    # Reset the class count for the next frame
    class_count.clear()


def draw_plots(ax, time_list, count_list, class_time_series, colors):
    """
    Draws the object detection count over time and by class type on the given axes.

    Args:
        ax (matplotlib.axes.Axes): The axes on which to draw the plots.
        time_list (list): List of timestamps for detections.
        count_list (list): List of total detected objects per timestamp.
        class_time_series (dict): Dictionary tracking the number of objects per class over time.
        colors (list): List of colors for plotting each class.

    Returns:
        img (numpy.ndarray): An image representation of the plot.
    """
    ax[0].cla()
    ax[0].plot(time_list, count_list, label='Total Count')
    ax[0].set_title("Object Count Over Time")
    ax[0].set_xlabel("Time")
    ax[0].set_ylabel("Count")
    ax[0].legend()

    ax[1].cla()
    for idx, (class_label, series) in enumerate(class_time_series.items()):
        ax[1].plot(series, label=class_label, color=colors[idx % len(colors)])
    ax[1].legend()
    ax[1].set_title("Object Count By Class Type Over Time")
    ax[1].set_xlabel("Frame")
    ax[1].set_ylabel("Count")

    plt.tight_layout()

    # Convert plot to image
    fig = plt.gcf()
    fig.canvas.draw()
    img = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    img = img.reshape(fig.canvas.get_width_height()[::-1] + (3,))

    return img


def display_frame(frame_placeholder, frame):
    """
    Display the current video frame in Streamlit.

    Args:
        frame_placeholder (streamlit.delta_generator.DeltaGenerator): Streamlit placeholder for the frame.
        frame (numpy.ndarray): The current video frame.

    Returns:
        None
    """
    frame_placeholder.image(frame, channels="BGR", use_column_width=True)


def display_plots(plot_placeholder, img):
    """
    Display the plot image in Streamlit.

    Args:
        plot_placeholder (streamlit.delta_generator.DeltaGenerator): Streamlit placeholder for the plot.
        img (numpy.ndarray): The plot image.

    Returns:
        None
    """
    plot_placeholder.image(img, channels="BGR", use_column_width=True)


# Initialize session state for the toggle switches if not already done
if 'camera1_on' not in st.session_state:
    st.session_state.camera1_on = False
if 'camera2_on' not in st.session_state:
    st.session_state.camera2_on = False

# Define the toggle functions for cameras
def toggle_camera1():
    st.session_state.camera1_on = not st.session_state.camera1_on
    if st.session_state.camera1_on:
        st.write("Camera 1 is ON")
        # Call a function to start processing camera 1 feed
        process_camera_feed(camera_id=1)
    else:
        st.write("Camera 1 is OFF")
        # Call a function to stop processing camera 1 feed

def toggle_camera2():
    st.session_state.camera2_on = not st.session_state.camera2_on
    if st.session_state.camera2_on:
        st.write("Camera 2 is ON")
        # Call a function to start processing camera 2 feed
        process_camera_feed(camera_id=2)
    else:
        st.write("Camera 2 is OFF")
        # Call a function to stop processing camera 2 feed



# Main function that encapsulates the app logic
def main():

   st.title('Object Detection and Tracking')

    # Place the toggles in columns at the beginning of your main function
    col1, col2 = st.columns(2)
    with col1:
        st.toggle('Activate Camera 1', key='camera1_on', on_change=toggle_camera1)
    with col2:
        st.toggle('Activate Camera 2', key='camera2_on', on_change=toggle_camera2)

    cam1_on, cam2_on, cam1_placeholder, cam2_placeholder = setup_ui()

    # Initialize camera feeds
    cam1_feed = cv2.VideoCapture(CAM1_URL)
    cam2_feed = cv2.VideoCapture(CAM2_URL)

    # Main loop for processing the feeds
    while True:
        if cam1_on:
            ret, frame = cam1_feed.read()
            if not ret:
                st.warning('CAM1 - End of Video.')
                continue
            processed_frame = process_frame(frame)
            draw_plots(cam1_placeholder, processed_frame)

        if cam2_on:
            ret, frame = cam2_feed.read()
            if not ret:
                st.warning('CAM2 - End of Video.')
                continue
            processed_frame = process_frame(frame)
            draw_plots(cam2_placeholder, processed_frame)

        # Break the loop if neither camera is on
        if not cam1_on and not cam2_on:
            break

    # Release resources when not processing
    stop_processing(cam1_feed, cam2_feed)

if __name__ == "__main__":
    main()
