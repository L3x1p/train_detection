import cv2 as cv
import threading
import pandas as pd
from detector import YOLODetector
from rpi_relays import RaspberryRelayLogic
import queue
import time

def get_camera_streams(csv_path):
    """Read camera streams, coordinates, and pin numbers from a CSV file."""
    camera_df = pd.read_csv(csv_path, header=None)
    camera_streams = camera_df.iloc[:, 0].tolist()
    camera_cords = [None] * len(camera_streams)
    camera_pins = []

    if camera_df.shape[1] > 1:
        for i in range(len(camera_streams)):
            cords = camera_df.iloc[i, 1]
            if pd.notna(cords):
                camera_cords[i] = tuple(map(int, str(cords).split(',')))

    if camera_df.shape[1] > 2:
        camera_pins = [tuple(map(int, str(pins).split(','))) for pins in camera_df.iloc[:, 2]]

    return camera_streams, camera_cords, camera_pins

class CameraHandler:

    def __init__(self, stream_url, relay_logic, frame_queue, processed_frame_queue, interest_area=None):
        self.stream_url = stream_url
        self.relay_logic = relay_logic
        self.frame_queue = frame_queue
        self.processed_frame_queue = processed_frame_queue
        self.yolo_detector = YOLODetector()
        self.interest_area = interest_area
        self.frame_lock = threading.Lock()

    def capture_video(self):
        max_reconnect_attempts = 5
        reconnect_delay = 10  # Delay in seconds

        while True:
            cap = cv.VideoCapture(self.stream_url)
            if not cap.isOpened():
                print(f"Failed to open stream: {self.stream_url}")
                return

            reconnect_attempts = 0
            while reconnect_attempts < max_reconnect_attempts:
                ret, frame = cap.read()
                if not ret:
                    print(f"Stream lost, attempting to reconnect {self.stream_url}")
                    cap.release()
                    time.sleep(reconnect_delay)
                    cap = cv.VideoCapture(self.stream_url)
                    reconnect_attempts += 1
                    continue

                with self.frame_lock:
                    if self.frame_queue.empty():  # Only put the frame if the queue is empty
                        self.frame_queue.put((self.stream_url, frame))
                reconnect_attempts = 0  # Reset reconnect attempts after a successful frame read

            if reconnect_attempts >= max_reconnect_attempts:
                print(f"Failed to reconnect after {max_reconnect_attempts} attempts: {self.stream_url}")
                break

            cap.release()



    def process_video(self):
        while True:
            _, frame = self.frame_queue.get()
            if self.interest_area:
                cv.rectangle(frame, (self.interest_area[0], self.interest_area[1]),
                             (self.interest_area[0] + self.interest_area[2],
                              self.interest_area[1] + self.interest_area[3]), (0, 255, 0), 2)

            processed_frame, intersection, count = self.yolo_detector.process_frame(frame, self.interest_area)
            self.relay_logic.update_relay_status(intersection, count)
            self.processed_frame_queue.put((self.stream_url, processed_frame))


def display_frames(processed_frame_queues, window_names):
    while True:
        for stream_url, processed_frame_queue in processed_frame_queues.items():
            if not processed_frame_queue.empty():
                _, frame = processed_frame_queue.get()
                window_name = window_names[stream_url]
                cv.imshow(window_name, frame)
        if cv.waitKey(1) & 0xFF == ord('q'):
            break
    cv.destroyAllWindows()



def main():
    csv_path = 'cam_list.csv'
    camera_streams, camera_cords, camera_pins = get_camera_streams(csv_path)

    frame_queues = {stream_url: queue.Queue() for stream_url in camera_streams}
    processed_frame_queues = {stream_url: queue.Queue() for stream_url in camera_streams}
    window_names = {stream_url: f"Camera {index}" for index, stream_url in enumerate(camera_streams)}

    for index, (stream_url, cords, pins) in enumerate(zip(camera_streams, camera_cords, camera_pins)):
        relay_logic = RaspberryRelayLogic(*pins)  # Initialize appropriately

        if cords is None:
            print(f"No coordinates found for stream: {stream_url}")
            continue

        camera_handler = CameraHandler(stream_url, relay_logic, frame_queues[stream_url],
                                       processed_frame_queues[stream_url], interest_area=cords)
        threading.Thread(target=camera_handler.capture_video).start()
        threading.Thread(target=camera_handler.process_video).start()

    display_frames(processed_frame_queues, window_names)

if __name__ == "__main__":
    main()