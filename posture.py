import cv2
import time
import math as m
import mediapipe as mp
import winsound
import time


# Calculate distance
def findDistance(point1, point2):
    x1, y1 = point1
    x2, y2 = point2
    dist = m.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
    return dist


# Calculate angle.
def findAngle(point1, point2):
    x1, y1 = point1
    x2, y2 = point2
    theta = m.acos((y2 - y1) * (-y1) / (m.sqrt(
        (x2 - x1) ** 2 + (y2 - y1) ** 2) * y1))
    degree = int(180 / m.pi) * theta
    return degree


"""
Function to send alert. Use this function to send alert when bad posture detected.
Feel free to get creative and customize as per your convenience.
"""


def sendWarning(x):
    global start
    if start == 0:
        start = time.time()
        winsound.Beep(frequency, duration)
        print(x)
    end = time.time()
    if (end - start) > max_dur:
        start = 0
    
    

# =============================CONSTANTS and INITIALIZATIONS=====================================#

start = 0
max_dur = 10.0

frequency = 2500  # Set Frequency To 2500 Hertz
duration = 1000  # Set Duration To 1000 ms == 1 second

# max angles
torso_inclination_max = 30
neck_inclination_max = 11
orientation = "left"
    
# Initilize frame counters.
good_frames = 0
bad_frames = 0

# Font type.
font = cv2.FONT_HERSHEY_SIMPLEX

# Colors.
blue = (255, 127, 0)
red = (50, 50, 255)
green = (127, 255, 0)
dark_blue = (127, 20, 0)
light_green = (127, 233, 100)
yellow = (0, 255, 255)
pink = (255, 0, 255)

# Initialize mediapipe pose class.
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(model_complexity=2)
# ===============================================================================================#


if __name__ == "__main__":
    # For webcam input replace file name with 0.
    file_name = 'input.mp4'
    
    cap = cv2.VideoCapture(0)

    # Meta.
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_size = (width, height)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
   

    # Video writer.
    video_output = cv2.VideoWriter('output.mp4', fourcc, fps, frame_size)

    while cap.isOpened():
        # Capture frames.
        success, image = cap.read()
        if not success:
            print("Null.Frames")
            break
        # Get fps.
        fps = cap.get(cv2.CAP_PROP_FPS)
        # Get height and width.
        h, w = image.shape[:2]

        # Convert the BGR image to RGB.
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Process the image.
        keypoints = pose.process(image)

        # Convert the image back to BGR.
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        # Use lm and lmPose as representative of the following methods.
        lm = keypoints.pose_landmarks
        lmPose = mp_pose.PoseLandmark # Enum codes for body parts
        
        coords = {}
        skip_frame = False
        
        # Acquire the landmark coordinates.
        # Once aligned properly, left or right should not be a concern.      
        body_parts = {"left shoulder": lmPose.LEFT_SHOULDER,
                      "right shoulder":lmPose.RIGHT_SHOULDER,
                      "left ear":      lmPose.LEFT_EAR,
                      "right ear":     lmPose.RIGHT_EAR,
                      "left hip":      lmPose.LEFT_HIP,
                      "right hip":     lmPose.RIGHT_HIP,
                      "right elbow":   lmPose.RIGHT_ELBOW,
                      "left elbow":   lmPose.LEFT_ELBOW,
                }
        for name in body_parts.keys():
            try:
                coords[name] = (int(lm.landmark[body_parts[name]].x * w),
                                int(lm.landmark[body_parts[name]].y * h))
            except AttributeError:
                print("{name} not found".format(name=name))
                skip_frame = True

        if skip_frame:
            continue
        # Calculate distance between left shoulder and right shoulder points.
        offset = findDistance(coords["left shoulder"], coords["right shoulder"])

        # Assist to align the camera to point at the side view of the person.
        # Offset threshold 30 is based on results obtained from analysis over 100 samples.
        if offset < 100:
            cv2.putText(image, str(int(offset)) + ' Aligned', (w - 150, 30), font, 0.9, green, 2)
        else:
            cv2.putText(image, str(int(offset)) + ' Not Aligned', (w - 150, 30), font, 0.9, red, 2)

        # Calculate angles.
        neck_inclination = findAngle(coords["right shoulder"], coords["right ear"])
        torso_inclination = findAngle(coords["right hip"], coords["right shoulder"])

        # Draw landmarks.
        cv2.circle(image, coords["right shoulder"], 7, yellow, -1)
        cv2.circle(image, coords["right ear"], 7, yellow, -1)
        cv2.circle(image, coords["right elbow"], 7, pink, -1)

        # Let's take y - coordinate of P3 100px above x1,  for display elegance.
        # Although we are taking y = 0 while calculating angle between P1,P2,P3.
        r_shldr_x, r_shldr_y = coords["right shoulder"]
        cv2.circle(image, (r_shldr_x, r_shldr_y - 100) , 7, dark_blue, -1)
        cv2.circle(image, coords["left shoulder"], 7, pink, -1)
        cv2.circle(image, coords["right hip"], 7, yellow, -1)

        # Similarly, here we are taking y - coordinate 100px above x1. Note that
        # you can take any value for y, not necessarily 100 or 200 pixels.
        r_hip_x, r_hip_y = coords["right hip"]
        cv2.circle(image, (r_hip_x, r_hip_y - 100), 7, dark_blue, -1)

        # Put text, Posture and angle inclination.
        # Text string for display.
        angle_text_string = 'Neck : ' + str(int(neck_inclination)) + '  Torso : ' + str(int(torso_inclination))

        # Determine whether good posture or bad posture.
        # The threshold angles have been set based on intuition.
        if neck_inclination < neck_inclination_max and torso_inclination < torso_inclination_max:
            bad_frames = 0
            good_frames += 1
            
            cv2.putText(image, angle_text_string, (10, 30), font, 0.9, light_green, 2)
            cv2.putText(image, str(int(neck_inclination)), (r_shldr_x + 10, r_shldr_y), font, 0.9, light_green, 2)
            cv2.putText(image, str(int(torso_inclination)), (r_hip_x + 10, r_hip_y), font, 0.9, light_green, 2)

            # Join landmarks.
            cv2.line(image, coords["right shoulder"], coords["right ear"], green, 4)
            cv2.line(image, coords["right shoulder"], (r_shldr_x, r_shldr_y - 100), green, 4)
            cv2.line(image, coords["right hip"], coords["right shoulder"], green, 4)
            cv2.line(image, coords["right hip"], (r_hip_x, r_hip_y - 100), green, 4)

        else:
            good_frames = 0
            bad_frames += 1

            cv2.putText(image, angle_text_string, (10, 30), font, 0.9, red, 2)
            cv2.putText(image, str(int(neck_inclination)), (r_shldr_x + 10, r_shldr_y), font, 0.9, red, 2)
            cv2.putText(image, str(int(torso_inclination)), (r_hip_x + 10, r_hip_y), font, 0.9, red, 2)

            # Join landmarks.
            cv2.line(image, coords["right shoulder"], coords["right ear"], red, 4)
            cv2.line(image, coords["right shoulder"], (r_shldr_x, r_shldr_y - 100), red, 4)
            cv2.line(image, coords["right hip"], coords["right shoulder"], red, 4)
            cv2.line(image, coords["right hip"], (r_hip_x, r_hip_y - 100), red, 4)

        # Calculate the time of remaining in a particular posture.
        good_time = (1 / fps) * good_frames
        bad_time =  (1 / fps) * bad_frames

        # Pose time.
        if good_time > 0:
            time_string_good = 'Good Posture Time : ' + str(round(good_time, 1)) + 's'
            cv2.putText(image, time_string_good, (10, h - 20), font, 0.9, green, 2)
        else:
            time_string_bad = 'Bad Posture Time : ' + str(round(bad_time, 1)) + 's'
            cv2.putText(image, time_string_bad, (10, h - 20), font, 0.9, red, 2)

        # If you stay in bad posture for more than 3 minutes (180s) send an alert.
        if bad_time > 2:
            sendWarning("Sending Warning")
        # Write frames.
        video_output.write(image)

        # Display.
        cv2.imshow('MediaPipe Pose', image)
        if cv2.waitKey(5) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()