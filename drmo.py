#!/usr/bin/env python3
import cv2
import PySimpleGUI as sg
from features import Eye, Face, Frame

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

    past = None
    while window(timeout=20)[0] != sg.WIN_CLOSED:

        # Read from the video capture and instantiate the frame
        ret, data = cap.read()
        if not ret: break
        if past != None: frame = Frame(data, (frame_width, frame_height), past)
        else: frame = Frame(data, (frame_width, frame_height), [])

        # Detect faces and render the rectangles on the frame
        frame.detect(face_detector, open_eyes_detector, left_eye_detector, right_eye_detector)
        past = frame.past
        draw = frame.render()

        # Update the GUI and write to a file
        window['image'](data=cv2.imencode('.png', draw)[1].tobytes())
        out.write(draw)

    # Clean up the display window and write the file
    cap.release()
    out.release()
    window.close()

