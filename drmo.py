#!/usr/bin/env python3
import os
import cv2
import util
import PySimpleGUI as sg
from datetime import datetime
from features import Frame, HIST

is_pi = False
led = None
try:
    import RPi.GPIO as gpio
    is_pi = True
except:
    pass

LAMP_CONTROL = 17

if __name__ == '__main__':

    fm = util.FileManager()
    if not fm.fully_installed:
        fm.install()

    # Turn on IR lamp
    if is_pi:
        gpio.setmode(gpio.BCM)
        gpio.setup(LAMP_CONTROL, gpio.OUT)
        gpio.output(LAMP_CONTROL, True)
        print('IR lamp on')

    # Handle cli args
    save = ''
    # if len(sys.argv) > 1:
    #     save = sys.argv[1]

    # Open the video capture
    cap = cv2.VideoCapture(0)
    if cap.isOpened() == False:
        print('Cannot open video capture!')
        exit(1)

    # Set up the output file 
    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))
    out = cv2.VideoWriter(os.path.join(util.data_path, 'out.mp4'), cv2.VideoWriter_fourcc(*'mp4v'), 10, (frame_width, frame_height))

    # Set up the classifiers
    face_detector = cv2.CascadeClassifier(os.path.join(util.data_path, 'haarcascade_frontalface_default.xml'))
    open_eyes_detector = cv2.CascadeClassifier(os.path.join(util.data_path, 'haarcascade_eye_tree_eyeglasses.xml'))
    right_eye_detector = cv2.CascadeClassifier(os.path.join(util.data_path, 'haarcascade_righteye_2splits.xml'))
    left_eye_detector = cv2.CascadeClassifier(os.path.join(util.data_path, 'haarcascade_lefteye_2splits.xml'))

    # Calculate new file names
    q = 0
    if save != '':
        for name in os.listdir(os.path.join(util.data_path, save)):
            n = int(name.split('.')[0])
            if n > q: q = n

    # Set up the PySimpleGUI window
    sg.theme('Material1')
    window = sg.Window('Driver Monitor',
        [
            [sg.Text('Driver Monitor')],
            [sg.Image(filename='', key='image')],
            [sg.MLine(size=(64,16), key='status', font=('mono', 12))],
            [sg.Text('Diagnostic data: resolution: {}x{}, torch device: {}{}'.format(
                str(frame_width),
                str(frame_height),
                util.device,
                ', saving eyes to '+os.path.join(util.data_path, save) if save != '' else '',
            ))],
            [sg.Text('Created by Theo Henson')],
        ],
        location=(0, 0),
        grab_anywhere=False,
        resizable=True
    )
    sg.cprint_set_output_destination(window, 'status')

    past = None
    last_status = -1

    while window(timeout=20)[0] != sg.WIN_CLOSED:

        # Read from the video capture and instantiate the frame
        ret, data = cap.read()
        if not ret: break
        if past != None: frame = Frame(data, (frame_width, frame_height), past)
        else: frame = Frame(data, (frame_width, frame_height), [])

        # Detect faces and render the rectangles on the frame
        frame.detect(face_detector, open_eyes_detector, left_eye_detector, right_eye_detector)
        past = frame.past
        draw = cv2.flip(frame.render(), 1)

        # Save eyes for dataset
        if save != '':
            for i in frame.carve_eyes():
                for j in i:
                    if len(j) == 0: continue
                    cv2.imwrite(os.path.join(util.data_path, save, '{}.jpg'.format(str(q))), j)
                    q += 1

        # Update the display video and write to a file
        window['image'](data=cv2.imencode('.png', draw)[1].tobytes())
        out.write(draw)

        # Skip status if not enough frames have been accumulated
        if len(past) < (HIST / 10):
            continue

        # Calculate the current status of the driver
        status = 0
        no_face = 0
        open_eyes = 0
        closed_eyes = 0
        for faces in past:
            if len(faces) == 0: no_face += 1
            for face in faces:
                for eye in face.eyes:
                    if eye == None: continue
                    if not eye.opened: closed_eyes += 1
                    else: open_eyes += 1

        if no_face > (len(past) * 0.3):
            status = 2
        elif closed_eyes > (open_eyes * 0.8):
            status = 1

        # Print the current status
        if status == last_status:
            pass
        elif status == 0:
            sg.cprint(datetime.now().strftime("%H:%M:%S -- DRIVER IS FOCUSED   "),
                background_color="green", text_color="white"
            )
        elif status == 1:
            sg.cprint(datetime.now().strftime("%H:%M:%S -- DRIVER IS UNFOCUSED "),
                background_color="red", text_color="white"
            )
        elif status == 2:
            sg.cprint(datetime.now().strftime("%H:%M:%S -- NO DRIVER DETECTED  "),
                background_color="yellow"
            )
        last_status = status


    # Clean up the display window and write the file
    cap.release()
    out.release()
    window.close()
    if is_pi:
        gpio.cleanup()

