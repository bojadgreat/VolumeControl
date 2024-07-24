import math
import time
from ctypes import cast, POINTER
import cv2
import mediapipe as mp
from comtypes import CLSCTX_ALL
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume


# Function to set system volume gradually
def set_system_volume(volume_target, current_volume, step=0.01, delay=0.01):
    while current_volume != volume_target:
        if current_volume < volume_target:
            current_volume = min(volume_target, current_volume + step)
        else:
            current_volume = max(volume_target, current_volume - step)
        devices = AudioUtilities.GetSpeakers()
        interface = devices.Activate(
            IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
        volume_interface = cast(interface, POINTER(IAudioEndpointVolume))
        volume_interface.SetMasterVolumeLevelScalar(current_volume, None)
        time.sleep(delay)
    return current_volume


cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
mpHands = mp.solutions.hands
hands = mpHands.Hands()
mpDraw = mp.solutions.drawing_utils


# Function to calculate angle between two points
def calculate_angle(a, b):
    radians = math.atan2(b[1] - a[1], b[0] - a[0])
    angle = math.degrees(radians)
    return angle


# Function to calculate centroid of five landmarks
def calculate_centroid(thumb, index, middle, ring, pinky):
    x = (thumb[0] + index[0] + middle[0] + ring[0] + pinky[0]) / 5
    y = (thumb[1] + index[1] + middle[1] + ring[1] + pinky[1]) / 5
    return x, y


previous_angle = 0
movement_threshold = 3
volume = 0.5

while True:
    status, img = cap.read()
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(imgRGB)
    multiRes = results.multi_hand_landmarks

    if (multiRes):
        indexPoint = ()
        thumbPoint = ()
        midPoint = ()
        ringPoint = ()
        pinkyPoint = ()

        for handLms in multiRes:
            mpDraw.draw_landmarks(img, handLms, mpHands.HAND_CONNECTIONS)

            for idHand, lm in enumerate(handLms.landmark):
                h, w, c = img.shape
                cx, cy = int(lm.x * w), int(lm.y * h)

                if idHand == 4:
                    thumbPoint = (cx, cy)
                if idHand == 8:
                    indexPoint = (cx, cy)
                if idHand == 12:
                    midPoint = (cx, cy)
                if idHand == 16:
                    ringPoint = (cx, cy)
                if idHand == 20:
                    pinkyPoint = (cx, cy)

        cv2.circle(img, thumbPoint, 15, (255, 255, 0), cv2.FILLED)
        cv2.circle(img, indexPoint, 15, (255, 255, 0), cv2.FILLED)
        cv2.circle(img, midPoint, 15, (255, 255, 0), cv2.FILLED)
        cv2.circle(img, ringPoint, 15, (255, 255, 0), cv2.FILLED)
        cv2.circle(img, pinkyPoint, 15, (255, 255, 0), cv2.FILLED)

        cv2.line(img, thumbPoint, indexPoint, (255, 255, 0), 3)
        cv2.line(img, indexPoint, midPoint, (255, 255, 0), 3)
        cv2.line(img, midPoint, ringPoint, (255, 255, 0), 3)
        cv2.line(img, ringPoint, pinkyPoint, (255, 255, 0), 3)

        # Calculate centroid of five landmarks
        centroid = calculate_centroid(thumbPoint, indexPoint, midPoint, ringPoint, pinkyPoint)

        # Calculate angles between centroid and each fingertip
        thumb_angle = calculate_angle(centroid, thumbPoint)
        index_angle = calculate_angle(centroid, indexPoint)
        middle_angle = calculate_angle(centroid, midPoint)
        ring_angle = calculate_angle(centroid, ringPoint)
        pinky_angle = calculate_angle(centroid, pinkyPoint)

        # Calculate average angle
        avg_angle = (thumb_angle + index_angle + middle_angle + ring_angle + pinky_angle) / 5

        # Calculate the difference between current angle and previous angle
        angle_diff = avg_angle - previous_angle

        # Determine gesture direction
        if abs(angle_diff) > movement_threshold:
            if angle_diff > 0:
                print("Clockwise gesture detected")
                # Increase volume gradually
                volume = set_system_volume(min(1.0, volume + 0.05), volume)
            elif angle_diff < 0:
                print("Counterclockwise gesture detected")
                # Decrease volume gradually
                volume = set_system_volume(max(0.0, volume - 0.05), volume)
        else:
            print("No movement detected, maintaining current volume")

        previous_angle = avg_angle

    cv2.imshow("Volume Control", img)
    cv2.waitKey(1)

    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

hands.close()
cap.release()
cv2.destroyAllWindows()
