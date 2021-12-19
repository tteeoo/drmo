#!/usr/bin/env python3
import cv2
import os
import sys
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
LED_YES = 23
LED_NONE = 22
LED_NO = 27

if __name__ == '__main__':

    fm = util.FileManager()
    if not fm.fully_installed:
        fm.install()

    # Turn on IR lamp and set up LEDs
    if is_pi:
        gpio.setmode(gpio.BCM)
        gpio.setwarnings(False)

        gpio.setup(LAMP_CONTROL, gpio.OUT)
        gpio.setup(LED_YES, gpio.OUT)
        gpio.setup(LED_NONE, gpio.OUT)
        gpio.setup(LED_NO, gpio.OUT)

        gpio.output(LAMP_CONTROL, True)
        gpio.output(LED_NONE, True)
        print('IR lamp on')

    # Handle cli args
    save = ''
    is_gui = False
    if len(sys.argv) > 1:
         is_gui = True

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
    if is_gui:
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


    # Handle signals
    sk = util.SignalKiller()

    while (not is_gui or window(timeout=20)[0] != sg.WIN_CLOSED) and not sk.kill_now:

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
        if is_gui:
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
            if is_gui:
                    sg.cprint(datetime.now().strftime("%H:%M:%S -- DRIVER IS FOCUSED   "),
                        background_color="green", text_color="white"
                    )
            if is_pi:
                gpio.output(LED_NO, False)
                gpio.output(LED_NONE, False)
                gpio.output(LED_YES, True)
        elif status == 1:
            if is_gui:
                sg.cprint(datetime.now().strftime("%H:%M:%S -- DRIVER IS UNFOCUSED "),
                    background_color="red", text_color="white"
                )
            if is_pi:
                gpio.output(LED_NO, True)
                gpio.output(LED_NONE, False)
                gpio.output(LED_YES, False)
        elif status == 2:
            if is_gui:
                sg.cprint(datetime.now().strftime("%H:%M:%S -- NO DRIVER DETECTED  "),
                    background_color="yellow"
                )
            if is_pi:
                gpio.output(LED_NO, False)
                gpio.output(LED_NONE, True)
                gpio.output(LED_YES, False)
        last_status = status

    print("Quitting")

    # Clean up the display window and write the file
    cap.release()
    out.release()
    if is_gui:
        window.close()
    if is_pi:
        gpio.cleanup()

