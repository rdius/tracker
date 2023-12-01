import datetime
import time
from ultralytics import YOLO
import cv2
from helper import create_video_writer
from deep_sort_realtime.deepsort_tracker import DeepSort
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict
import streamlit as st
import threading



# Load YOLO model and DeepSort tracker
model = YOLO("yolov8x.pt")
tracker = DeepSort(max_age=50)

CONFIDENCE_THRESHOLD = 0.8

# Placeholder for camera feeds and plots
frame_placeholder1 = st.empty()
frame_placeholder2 = st.empty()
plot_placeholder1 = st.empty()
plot_placeholder2 = st.empty()
# Define RTSP URLs for cameras
#RTSP_URL_CAMERA1 = "http://195.196.36.242/mjpg/video.mjpg"
RTSP_URL_CAMERA1 = 'http://158.58.130.148/mjpg/video.mjpg'
RTSP_URL_CAMERA2 = "http://pendelcam.kip.uni-heidelberg.de/mjpg/video.mjpg"




def update_time_series_data(class_name, class_count, class_time_series, time_list, count_list, start, obj_count):
    for class_name, count in class_count.items():
        if class_name not in class_time_series:
            class_time_series[class_name] = []
        class_time_series[class_name].append(count)

    time_list.append(start)
    count_list.append(obj_count)



def detect_objects_and_tracky(frame, class_count, model, tracker):
    # Perform detection with YOLO model
    detections = model(frame)[0]  # Assuming this returns the detections

    results = []
    #for data in detections.tolist():
    #    confidence = data[4]
    #    if float(confidence) < CONFIDENCE_THRESHOLD:
    #        continue

    for data in detections.boxes.data.tolist():
        confidence = data[4]
        if float(confidence) < CONFIDENCE_THRESHOLD:
            continue

        xmin, ymin, xmax, ymax, class_id = map(int, data[:5])
        results.append([[xmin, ymin, xmax - xmin, ymax - ymin], confidence, class_id])

        class_name = model.names[class_id]
        normalized_class_name = class_name.rstrip('s')
        class_count[normalized_class_name] += 1

    # Update tracks with the DeepSort tracker
    tracks = tracker.update_tracks(results, frame=frame)

    # Draw rectangles for visual feedback
    for track in tracks:
        if not track.is_confirmed():
            continue
        xmin, ymin, xmax, ymax = track.to_tlbr()  # Assuming DeepSort provides to_tlbr method
        cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)

    return results


def detect_objects_and_track(frame, class_count, model, tracker, CONFIDENCE_THRESHOLD=0.8):
    # Perform detection with YOLO model
    detections = model(frame)[0]  # Assuming this returns the detections

    results = []
    for data in detections.boxes.data.tolist():
        confidence = data[4]
        if float(confidence) < CONFIDENCE_THRESHOLD:
            continue

        xmin, ymin, xmax, ymax, class_id = map(int, data[:5])
        results.append([[xmin, ymin, xmax - xmin, ymax - ymin], confidence, class_id])

        class_name = model.names[class_id]
        normalized_class_name = class_name.rstrip('s')
        class_count[normalized_class_name] += 1

    # Update tracks with the DeepSort tracker
    tracks = tracker.update_tracks(results, frame=frame)

    # Draw rectangles for visual feedback
    for track in tracks:
        if not track.is_confirmed():
            continue
        xmin, ymin, xmax, ymax = track.to_tlbr()
        cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)

    # Calculate total object count
    obj_count = sum(class_count.values())

    return detections, obj_count



def draw_plots(ax, time_list, count_list, class_time_series, colors):
    ax[0].cla()
    ax[0].plot(time_list, count_list)
    ax[0].set_title("Object Count Over Time")
    ax[0].set_xlabel("Time")
    ax[0].set_ylabel("Count")

    ax[1].cla()
    for idx, (label, timeseries) in enumerate(class_time_series.items()):
        ax[1].plot(timeseries, label=label, color=colors[idx % len(colors)])
    ax[1].legend()
    ax[1].set_title("Object Count By Class Type Over Time")
    ax[1].set_xlabel("Frame")
    ax[1].set_ylabel("Count")


def display_frame(frame_placeholder, frame):
    frame_placeholder.image(frame, channels="BGR", use_column_width=True)


def display_plots(plot_placeholder, img):
    plot_placeholder.image(img, channels="BGR", use_column_width=True)


# Define the process_camera_feed function
def process_camera_feed_(video_cap, frame_placeholder, plot_placeholder, model, tracker):
    CONFIDENCE_THRESHOLD = 0.8
    class_count = defaultdict(int)
    class_time_series = {}
    time_list, count_list = [], []
    colors = [(np.random.rand(), np.random.rand(), np.random.rand()) for _ in range(1000)]

    plt.ion()
    fig, ax = plt.subplots(2, 1, figsize=(10, 8))
    plt.subplots_adjust(hspace=0.4)

    writer = create_video_writer(video_cap, "output.mp4")
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter('plot_video.mp4', fourcc, 1, (fig.canvas.get_width_height()[0], fig.canvas.get_width_height()[1]))

    while True:
        start = datetime.datetime.now()
        ret, frame = video_cap.read()
        if not ret:
            st.warning('End of Video.')
            break

        detections = model(frame)[0]

        # Detect objects and track
        detect_objects_and_track(frame, class_count, detections, tracker)

        # Update time series data
        obj_count = sum(class_count.values())
        update_time_series_data(class_count, class_time_series, time_list, count_list, start, obj_count)

        # Draw plots and display frame
        draw_plots(ax, time_list, count_list, class_time_series, colors)
        display_frame(frame_placeholder, frame)

        # Draw and display plots
        fig.canvas.draw()
        img = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        img = img.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        display_plots(plot_placeholder, img)

        # Write video and plot
        writer.write(frame)
        video_writer.write(img)

        end = datetime.datetime.now()
        fps = f"FPS: {1 / (end - start).total_seconds():.2f}"
        cv2.putText(frame, fps, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 2)

        if cv2.waitKey(1) == ord("q"):
            break

    video_cap.release()
    writer.release()
    video_writer.release()
    cv2.destroyAllWindows()

def update_class_count(detections, class_count, model, CONFIDENCE_THRESHOLD=0.8):
    """
    Update the count of each detected class.

    Args:
    - detections: The detections made by the YOLO model.
    - class_count: A dictionary to keep track of the count of each class.
    - model: The YOLO model used for detection.
    - CONFIDENCE_THRESHOLD: The threshold for considering a detection valid.

    Returns:
    - A dictionary with updated class counts.
    """
    for data in detections.boxes.data.tolist():
        confidence = data[4]
        if float(confidence) < CONFIDENCE_THRESHOLD:
            continue
        class_id = int(data[5])
        class_name = model.names[class_id]

        # Normalize class name to handle singular and plural forms (e.g., 'person' vs. 'persons')
        normalized_class_name = class_name.rstrip('s')

        class_count[normalized_class_name] += 1  # Increment count for the detected class

    return class_count


def process_camera_feed4(rtsp_url, model, tracker):
    CONFIDENCE_THRESHOLD = 0.8
    class_count = defaultdict(int)
    class_time_series = {}
    time_list, count_list = [], []
    colors = [(np.random.rand(), np.random.rand(), np.random.rand()) for _ in range(1000)]

    video_cap = cv2.VideoCapture(rtsp_url)
    plt.ion()
    fig, ax = plt.subplots(2, 1, figsize=(10, 8))
    plt.subplots_adjust(hspace=0.4)

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter('plot_video.mp4', fourcc, video_cap.get(cv2.CAP_PROP_FPS), (int(video_cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(video_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))))

    while video_cap.isOpened():
        ret, frame = video_cap.read()
        if not ret:
            break

        start = datetime.datetime.now()
        detections = model(frame)[0]
        obj_count = 0

        for data in detections.boxes.data.tolist():
            confidence = data[4]
            if float(confidence) < CONFIDENCE_THRESHOLD:
                continue

            xmin, ymin, xmax, ymax = map(int, data[:4])
            class_id = int(data[5])
            class_name = model.names[class_id]
            normalized_class_name = class_name.rstrip('s')

            class_count[normalized_class_name] += 1
            obj_count += 1

            cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
            cv2.putText(frame, f"{class_name} {confidence:.2f}", (xmin, ymin - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        update_time_series_data(class_count, class_time_series, time_list, count_list, start, obj_count)
        draw_plots(ax, time_list, count_list, class_time_series, colors)

        fig.canvas.draw()
        plot_img = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        plot_img = plot_img.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        video_writer.write(plot_img)

        # Display frame and plot in Streamlit (if using Streamlit)
        # frame_placeholder.image(frame, channels="BGR", use_column_width=True)
        # plot_placeholder.image(plot_img, channels="BGR", use_column_width=True)

        end = datetime.datetime.now()
        fps = f"FPS: {1 / (end - start).total_seconds():.2f}"
        cv2.putText(frame, fps, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 2)

    video_cap.release()
    video_writer.release()
    plt.close()


def process_camera_feed(rtsp_url, model, tracker, frame_placeholder, plot_placeholder):
    # Initialize necessary variables and objects for processing the camera feed
    CONFIDENCE_THRESHOLD = 0.8
    class_count = defaultdict(int)
    class_time_series = {}
    time_list, count_list = [], []
    colors = [(np.random.rand(), np.random.rand(), np.random.rand()) for _ in range(1000)]

    plt.ion()
    fig, ax = plt.subplots(2, 1, figsize=(10, 8))
    plt.subplots_adjust(hspace=0.4)

    # Open the camera feed
    video_cap = cv2.VideoCapture(rtsp_url)
    writer = create_video_writer(video_cap, "output.mp4")
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter('plot_video.mp4', fourcc, 1, (fig.canvas.get_width_height()[0], fig.canvas.get_width_height()[1]))

    # Process the camera feed
    while True:
        ret, frame = video_cap.read()
        if not ret:
            break

        # Object detection and tracking
        #detections, obj_count = detect_objects_and_track(frame, model, tracker)
        detections, obj_count = detect_objects_and_track(frame, class_count, model, tracker)
        #class_count = update_class_count(detections, class_count)
        class_count = update_class_count(detections, class_count, model, CONFIDENCE_THRESHOLD=0.8)

        # Update time series data
        update_time_series_data(class_count, class_time_series, time_list, count_list, datetime.datetime.now(), obj_count)

        # Draw plots and display frame
        plot_img = draw_plots(ax, time_list, count_list, class_time_series,colors)
        display_frame(frame_placeholder, frame)
        display_plots(plot_placeholder, plot_img)

    # Release resources
    video_cap.release()
    writer.release()
    video_writer.release()


def update_time_series_data(class_count, class_time_series, time_list, count_list, start, obj_count):
    for class_name, count in class_count.items():
        if class_name not in class_time_series:
            class_time_series[class_name] = []
        class_time_series[class_name].append(count)

    time_list.append(start)
    count_list.append(obj_count)

def draw_plots(ax, time_list, count_list, class_time_series, colors):
    ax[0].cla()
    ax[0].plot(time_list, count_list)
    ax[0].set_title("Object Count Over Time")
    ax[0].set_xlabel("Time")
    ax[0].set_ylabel("Count")

    ax[1].cla()
    for idx, (label, timeseries) in enumerate(class_time_series.items()):
        ax[1].plot(timeseries, label=label, color=colors[idx % len(colors)])

    ax[1].legend()
    ax[1].set_title("Object Count By Class Type Over Time")
    ax[1].set_xlabel("Frame")
    ax[1].set_ylabel("Count")
    plt.tight_layout()


def get_video_capture(rtsp_url):
    """
    This function creates a video capture object for the given RTSP URL.

    Args:
    rtsp_url (str): The RTSP URL of the camera.

    Returns:
    cv2.VideoCapture: The video capture object for the given URL.
    """
    video_cap = cv2.VideoCapture(rtsp_url)

    if not video_cap.isOpened():
        st.error(f"Failed to open camera with URL: {rtsp_url}")
    return video_cap





def main():
    # Initialize session state for toggles if not already present
    if 'cam1_on' not in st.session_state:
        st.session_state['cam1_on'] = False
    if 'cam2_on' not in st.session_state:
        st.session_state['cam2_on'] = False

    # Layout for toggle buttons
    col1, col2 = st.columns(2)
    with col1:
        cam1_on = st.toggle('Activate Camera 1')
        if cam1_on:
            st.session_state['cam1_on'] = True
            st.write('Camera 1 activated!')
        else:
            st.session_state['cam1_on'] = False
            st.write('Camera 1 deactivated.')

    with col2:
        cam2_on = st.toggle('Activate Camera 2')
        if cam2_on:
            st.session_state['cam2_on'] = True
            st.write('Camera 2 activated!')
        else:
            st.session_state['cam2_on'] = False
            st.write('Camera 2 deactivated.')

    # Camera 1 processing
    if st.session_state['cam1_on']:
        st.write("Processing Camera 1 Feed...")
        frame_placeholder1, plot_placeholder1 = st.columns(2)
        process_camera_feed(RTSP_URL_CAMERA1, model, tracker, frame_placeholder1, plot_placeholder1)

    # Camera 2 processing
    if st.session_state['cam2_on']:
        st.write("Processing Camera 2 Feed...")
        frame_placeholder2, plot_placeholder2 = st.columns(2)
        process_camera_feed(RTSP_URL_CAMERA2, model, tracker, frame_placeholder2, plot_placeholder2)

if __name__ == "__main__":
    main()
