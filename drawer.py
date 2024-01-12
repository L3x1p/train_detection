import cv2 as cv
import pandas as pd

def get_camera_streams(csv_path):
    """Read camera streams and pin numbers from a CSV file."""
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

class CameraAOI:
    def __init__(self, stream_url, window_name):
        self.stream_url = stream_url
        self.window_name = window_name
        self.interest_area_defined = False
        self.interest_area = (0, 0, 0, 0)
        self.ix, self.iy = -1, -1
        self.drawing = False
        self.latest_frame = None

    def update_latest_frame(self, frame):
        self.latest_frame = frame

    def draw_rectangle(self, event, x, y, flags, param):
        if self.latest_frame is None:
            return

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

def define_interest_areas(csv_path):
    camera_streams, _, _ = get_camera_streams(csv_path)

    for stream_url in camera_streams:
        camera_aoi = CameraAOI(stream_url, f"Camera AOI: {stream_url}")

        cap = cv.VideoCapture(stream_url)
        if not cap.isOpened():
            print(f"Failed to open stream: {stream_url}")
            continue

        while True:
            ret, frame = cap.read()
            if not ret:
                print(f"Failed to grab a frame from stream: {stream_url}")
                break

            camera_aoi.update_latest_frame(frame)
            cv.imshow(camera_aoi.window_name, frame)
            cv.setMouseCallback(camera_aoi.window_name, camera_aoi.draw_rectangle)

            key = cv.waitKey(1) & 0xFF
            if key == ord('s'):
                # Save the interest area to the CSV
                update_csv_with_aoi(csv_path, stream_url, camera_aoi.interest_area)
                print(f"Saved interest area to CSV for stream: {stream_url}")
                break
            elif key == ord('q'):
                # Restart drawing
                camera_aoi.interest_area_defined = False

        cap.release()
        cv.destroyAllWindows()

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
    print(f'Saved interest area to CSV for stream: {stream_url}')

if __name__ == '__main__':
    csv_path = 'cam_list.csv'  # Replace with the path to your CSV file
    define_interest_areas(csv_path)
