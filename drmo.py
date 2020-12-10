#!/usr/bin/env python3
import cv2
import PySimpleGUI as sg

# Define constants for colors
RED = (0, 0, 255)
GREEN = (0, 255, 0)

# TODO: Implement CNN
def predict(eye):
    """Returns False if the input eye image is closed. True if open."""

    return True

def render(frame, face_detector, open_eyes_detector, left_eye_detector, right_eye_detector, frame_x, frame_y):
    """Returns a rendered frame of video to be displayed. Takes input from a video capture,
    uses OpenCV classifiers and draws box around certain features."""

    # Apply some filters to normalize the image
    frame = cv2.resize(frame, (frame_x, frame_y), fx=0.6, fy=0.6) # resize
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) # extract greyscale
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) # extract rgb
    
    # Detect faces
    faces = face_detector.detectMultiScale(
        gray,
        scaleFactor=1.2,
        minNeighbors=5,
        minSize=(50, 50),
        flags=cv2.CASCADE_SCALE_IMAGE
    )

    # Iterate coordinates of each detected face
    for (x, y, w, h) in faces:

        # Extract just the face from the frame
        face = frame[y:y+h, x:x+w]
        gray_face = gray[y:y+h, x:x+w]

        # Separate the face into left and right sides
        left_face = frame[y:y+h, x+int(w/2):x+w]
        left_face_gray = gray[y:y+h, x+int(w/2):x+w]

        right_face = frame[y:y+h, x:x+int(w/2)]
        right_face_gray = gray[y:y+h, x:x+int(w/2)]

        # Detect the left eye
        left_eye = left_eye_detector.detectMultiScale(
            left_face_gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30),
            flags = cv2.CASCADE_SCALE_IMAGE
        )

        # Detect the right eye
        right_eye = right_eye_detector.detectMultiScale(
            right_face_gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30),
            flags = cv2.CASCADE_SCALE_IMAGE
        )

        # Check if each eye is closed or not and draw a rectangle of the proper color
        for (ex, ey, ew, eh) in right_eye:
            color = GREEN
            pred = predict(right_face[ey:ey+eh, ex:ex+ew])
            if not pred: color = RED

            cv2.rectangle(right_face, (ex, ey), (ex+ew, ey+eh), color, 2)

        for (ex, ey, ew, eh) in left_eye:
            color = GREEN
            pred = predict(left_face[ey:ey+eh, ex:ex+ew])
            if not pred: color = RED

            cv2.rectangle(left_face, (ex, ey), (ex+ew, ey+eh), color, 2)
    
        # Draw a rectangle around the face
        cv2.rectangle(frame, (x, y), (x+w, y+h), GREEN, 2)

    return frame

if __name__ == '__main__':

    # Open the video capture
    cap = cv2.VideoCapture(0)
    if cap.isOpened() == False:
        print('error: cannot open video capture')
        exit(1)

    # Set up the output file 
    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))
    out = cv2.VideoWriter('out.mp4', cv2.VideoWriter_fourcc(*'mp4v'), 10, (frame_width, frame_height))

    # Set up the classifiers
    face_detector = cv2.CascadeClassifier('./data/haarcascade_frontalface_default.xml')
    open_eyes_detector = cv2.CascadeClassifier('./data/haarcascade_eye_tree_eyeglasses.xml')
    right_eye_detector = cv2.CascadeClassifier('./data/haarcascade_righteye_2splits.xml')
    left_eye_detector = cv2.CascadeClassifier('./data/haarcascade_lefteye_2splits.xml')

    # Set up the PySimpleGUI window
    sg.theme('Material1')
    window = sg.Window('Driver Monitor',
        [
            [sg.Text('Driver Monitor')],
            [sg.Image(filename='', key='image')],
            [sg.Text('Created by Theo Henson')],
        ],
        location=(0, 0),
        grab_anywhere=False,
        resizable=True
    )


    while window(timeout=20)[0] != sg.WIN_CLOSED:

        # Read from the video capture and render the frame
        ret, frame = cap.read()
        if not ret: break
        frame = render(frame, face_detector, open_eyes_detector, left_eye_detector, right_eye_detector, frame_width, frame_height)

        # Update the GUI and write to a file
        window['image'](data=cv2.imencode('.png', frame)[1].tobytes())
        out.write(frame)

    # Clean up the display window and write the file
    cap.release()
    out.release()
    window.close()

