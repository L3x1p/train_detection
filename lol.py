import cv2 as cv
import threading
import pandas as pd
from detector import YOLODetector  # Replace with your actual import statement
from rpi_relays import RaspberryRelayLogic  # Import the RaspberryRelayLogic class
import time
def get_camera_streams(csv_path):
    """Read camera streams and pin numbers from a CSV file."""
    camera_df = pd.read_csv(csv_path, header=None)
    camera_streams = camera_df.iloc[:, 0].tolist()
    camera_cords = [None] * len(camera_streams)  # Initialize with None for each camera
    camera_pins = []

    if camera_df.shape[1] > 1:
        # Parse coordinates if the second column exists
        for i in range(len(camera_streams)):
            cords = camera_df.iloc[i, 1]
            if pd.notna(cords):  # Check if cords is not NaN
                camera_cords[i] = tuple(map(int, str(cords).split(',')))

    if camera_df.shape[1] > 2:
        # Parse pin numbers if the third column exists
        camera_pins = [tuple(map(int, str(pins).split(','))) for pins in camera_df.iloc[:, 2]]

    return camera_streams, camera_cords, camera_pins


class CameraHandler:
    def __init__(self, stream_url, window_name, relay_logic):
        self.stream_url = stream_url
        self.window_name = window_name
        self.latest_frame = None
        self.frame_lock = threading.Lock()
        self.interest_area_defined = False
        self.interest_area = (0, 0, 0, 0)
        self.yolo_detector = YOLODetector()
        self.ix, self.iy = -1, -1
        self.drawing = False
        self.relay_logic = relay_logic  # Add this line


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
                    self.latest_frame = frame
                reconnect_attempts = 0  # Reset reconnect attempts after a successful frame read

            if reconnect_attempts >= max_reconnect_attempts:
                print(f"Failed to reconnect after {max_reconnect_attempts} attempts: {self.stream_url}")
                break

            cap.release()

    def draw_on_frame(self, frame):
        if self.interest_area_defined:
            x, y, w, h = self.interest_area
            cv.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    def draw_rectangle(self, event, x, y, flags, param):
        # status,cords=param
        # print(status,cords)
        # if status==True:
            # self.interest_area=
        if event == cv.EVENT_LBUTTONDOWN:
            self.drawing = True
            self.ix, self.iy = x, y
        elif event == cv.EVENT_MOUSEMOVE and self.drawing:
            temp_frame = self.latest_frame.copy()
            cv.rectangle(temp_frame, (self.ix, self.iy), (x, y), (0, 255, 0), 2)
            cv.imshow(self.window_name, temp_frame)
        elif event == cv.EVENT_LBUTTONUP:
            self.drawing = False
            self.interest_area_defined = True
            self.interest_area = (min(self.ix, x), min(self.iy, y), abs(self.ix - x), abs(self.iy - y))
            # cv.destroyWindow(self.window_name)  # Close the preview window once the area is defined

    def define_interest_area(self, predefined_area):
        if predefined_area:
            self.interest_area = predefined_area
            self.interest_area_defined = True
            return
        # print(status)
        # Capture a single frame for preview
        cap = cv.VideoCapture(self.stream_url)
        ret, frame = cap.read()
        if not ret:
            print(f"Failed to grab a frame from stream: {self.stream_url}")
            cap.release()
            return

        cap.release()  # Release the capture immediately after grabbing the preview frame

        # Display the preview frame
        cv.namedWindow(self.window_name)
        cv.imshow(self.window_name, frame)
        cv.setMouseCallback(self.window_name, self.draw_rectangle)

        # Wait until the user has defined the interest area
        while not self.interest_area_defined:
            key = cv.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                update_csv_with_aoi('cam_list.csv', self.stream_url, self.interest_area)

        # After defining the interest area, destroy the preview window
        cv.destroyWindow(self.window_name)

    def process_video(self):
        cv.namedWindow(self.window_name)  # Use the same window name for processing
        while True:
            key = cv.waitKey(1) & 0xFF
            with self.frame_lock:
                frame = self.latest_frame.copy() if self.latest_frame is not None else None
            if frame is not None:
                self.draw_on_frame(frame)
                processed_frame, intersection, count = self.yolo_detector.process_frame(frame, self.interest_area)
                # print(intersection, count)
                cv.imshow(self.window_name, processed_frame)
                self.relay_logic.update_relay_status(intersection, count)  # Add this line

            if key == ord('q'):
                break
            elif key == ord('s'):
                update_csv_with_aoi('cam_list.csv', self.stream_url, self.interest_area)
        cv.destroyAllWindows()


def main():
    cam_list_csv_path = 'cam_list.csv'
    camera_streams, camera_cords, camera_pins = get_camera_streams(cam_list_csv_path)

    handlers = []
    for index, (stream_url, cords, pins) in enumerate(zip(camera_streams, camera_cords, camera_pins)):
        relay_logic = RaspberryRelayLogic(*pins)  # Initialize relay logic for each camera
        handler = CameraHandler(stream_url, f'Camera {index}', relay_logic)
        handlers.append(handler)
        threading.Thread(target=handler.capture_video, daemon=True).start()

    for handler, cords in zip(handlers, camera_cords):
        handler.define_interest_area(cords)

    for handler in handlers:
        threading.Thread(target=handler.process_video).start()






def update_csv_with_aoi(csv_path, stream_url, new_aoi):
    camera_df = pd.read_csv(csv_path, header=None)
    new_aoi_str = ",".join(map(str, new_aoi))  # Convert new_aoi to a comma-separated string

    for index, row in camera_df.iterrows():
        if row[0] == stream_url:
            if camera_df.shape[1] < 2:  # Check if there is a column to store the new_aoi
                # If not, append a new column
                camera_df[1] = pd.NA
            camera_df.at[index, 1] = new_aoi_str  # Store the new_aoi string
            break

    camera_df.to_csv(csv_path, header=None, index=False)
    print('Saved to CSV')




if __name__ == '__main__':
    main()