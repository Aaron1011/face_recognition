#!/usr/bin/python
"""
This program is demonstration for face and object detection using haar-like features.
The program finds faces in a camera image or video stream and displays a red box around them.

Original C implementation by:  ?
Python implementation by: Roman Stanchak, James Bowman
"""
import sys
import cv2.cv as cv
import cv2
from optparse import OptionParser
import usb.core
import usb.util
import time
from pyfaces.pyfaces import PyFaces
import threading
import Queue
import os
import pyttsx
ENGINE = pyttsx.init()

#ENGINE.say("Program started")
#ENGINE.runAndWait()

num = 0


PEOPLE = {'/home/aaron/Downloads/photos/1.jpeg': 'Aaron Hill',
          '/home/aaron/Downloads/photos/2.jpeg': 'Chase Adams',
          '/home/aaron/Downloads/photos/4.jpeg': 'Hunter Adams',
          '/home/aaron/Downloads/photos/3.jpeg': 'A Gibby Baby'}

class FaceRecognizer(threading.Thread):
    def __init__(self, imgdir, queue):
        threading.Thread.__init__(self)
        self.imgdir = imgdir
        self.queue = queue
        self.kill = False

    def run(self):
        global ENGINE
        global PEOPLE
        while True:
            if self.kill:
                break
            filename = self.queue.get()
            match = PyFaces(filename, self.imgdir, 20, 100)
            print match.matchfile
            ENGINE.say(PEOPLE[match.matchfile] + " detected!")
            ENGINE.runAndWait()
            os.remove(filename)
            self.queue.task_done()




# Parameters for haar detection
# From the API:
# The default parameters (scale_factor=2, min_neighbors=3, flags=0) are tuned
# for accurate yet slow object detection. For a faster operation on real video
# images the settings are:
# scale_factor=1.2, min_neighbors=2, flags=CV_HAAR_DO_CANNY_PRUNING,
# min_size=<minimum possible face size

min_size = (20, 20)
image_scale = 2
haar_scale = 1.2
min_neighbors = 2
haar_flags = 0

last_x = -1000
last_y = -1000


def detect_and_draw(img, cascade):
    global num
    global last_x
    global last_y
    # allocate temporary images
    gray = cv.CreateImage((img.width,img.height), 8, 1)
    small_img = cv.CreateImage((cv.Round(img.width / image_scale),
			       cv.Round (img.height / image_scale)), 8, 1)

    # convert color input image to grayscale
    cv.CvtColor(img, gray, cv.CV_BGR2GRAY)

    # scale input image for faster processing
    cv.Resize(gray, small_img, cv.CV_INTER_LINEAR)

    cv.EqualizeHist(small_img, small_img)

    if(cascade):
        t = cv.GetTickCount()
        faces = cv.HaarDetectObjects(small_img, cascade, cv.CreateMemStorage(0),
                                     haar_scale, min_neighbors, haar_flags, min_size)
        t = cv.GetTickCount() - t
        #print "detection time = %gms" % (t/(cv.GetTickFrequency()*1000.))
        if faces:
            for ((x, y, w, h), n) in faces:
                # the input to cv.HaarDetectObjects was resized, so scale the
                # bounding box of each face and convert it to two CvPoints
                pt1 = (int(x * image_scale), int(y * image_scale))
                pt2 = (int((x + w) * image_scale), int((y + h) * image_scale))
                x1 = pt1[0]
                x2 = pt2[0]
                y1 = pt1[1]
                y2 = pt2[1]
                centerx = x1+((x2-x1)/2)
                centery = y1+((y2-y1)/2)
                centery = int(y) + cv.Round(h/2)
                if x1 in range(last_x-50, last_x+50) and y1 in range(last_y-50, last_y+51):
                    return
                print x1, y1
                print last_x, last_y
                last_x = x1
                last_y = y1

                cv.SetImageROI(img, (x1, y1, x2-x1, y2-y1))
                cv.ShowImage("result", img)
                filename = "photos/" +str(num) + ".jpeg"
                thumbnail = cv.CreateMat(250, 250, cv.CV_8UC3)
                cv.Resize(img, thumbnail)
                cv.SaveImage(filename, thumbnail)
                q.put(filename)
                num += 1

                cv.ResetImageROI(img)
                #cv.Rectangle(img, pt1, pt2, cv.RGB(255, 0, 0), 3, 8, 0)

    #cv.ShowImage("result", img)

if __name__ == '__main__':

    parser = OptionParser(usage = "usage: %prog [options] [filename|camera_index]")
    parser.add_option("-c", "--cascade", action="store", dest="cascade", type="str", help="Haar cascade file, default %default", default = "../data/haarcascades/haarcascade_frontalface_alt.xml")
    (options, args) = parser.parse_args()

    cascade = cv.Load(options.cascade)


    if len(args) != 1:
        parser.print_help()
        sys.exit(1)

    input_name = args[0]
    if input_name.isdigit():
        capture = cv.CreateCameraCapture(int(input_name))
        cv.SetCaptureProperty(capture, cv.CV_CAP_PROP_FRAME_WIDTH, 250)
        cv.SetCaptureProperty(capture, cv.CV_CAP_PROP_FRAME_HEIGHT, 250)

    else:
        capture = None

    q = Queue.Queue()
    a = FaceRecognizer('/home/aaron/Downloads/photos', q)
    a.daemon = True
    a.start()


    cv.NamedWindow("result", 1)

    if capture:
        frame_copy = None
        while True:
            frame = cv.QueryFrame(capture)
            if not frame:
                cv.WaitKey(0)
                break
            if not frame_copy:
                frame_copy = cv.CreateImage((frame.width,frame.height),
                                            cv.IPL_DEPTH_8U, frame.nChannels)
            if frame.origin == cv.IPL_ORIGIN_TL:
                cv.Copy(frame, frame_copy)
            else:
                cv.Flip(frame, frame_copy, 0)

            detect_and_draw(frame_copy, cascade)

            if cv.WaitKey(10) >= 0:
                break
    else:
        image = cv.LoadImage(input_name, 1)
        detect_and_draw(image, cascade)
        cv.WaitKey(0)

    a.kill = True
    cv.DestroyWindow("result")
