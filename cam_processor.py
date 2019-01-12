import cv2
import queue
import threading
import time
import numpy as np
#from imutils.video import pivideostream as VideoStream 
from imutils.video import VideoStream
#import imutilsx.video.videostream.py
#from lib import 

class cam_processor:

    def __init__(self,
                 out_queue,
                 cap_frame_width = 640,
                 cap_frame_height = 480,
                 cap_framerate = 32,
                 out_queue_full_sleep = .1,
                 out_queue_max_wait = 0.01,
                 input_source = ''):
        self._out_queue_full_sleep = out_queue_full_sleep
        self._out_queue_max_wait = out_queue_max_wait

        self._pause_mode = False

        self._ready = False
        self._error = ''
        self._working = False
        self._stop_flag = False
        self._cap_frame_width = cap_frame_width
        self._cap_frame_height = cap_frame_height
        self._cap_framerate = cap_framerate

        #Get the camera
        #vs = VideoStream(usePiCamera=True).start()
        #vs.resolution( cap_frame_width, cap_frame_height)
        #self._video_stream = vs
        #self._video_stream.start()
        if input_source == '':
            print ('Using Pi Camera')
            self._video_stream = VideoStream(usePiCamera=True, 
                                        resolution=(self._cap_frame_width, self._cap_frame_height),
                                        framerate = self._cap_framerate).start()
        else:
            print ('Using Input Source: ', input_source)
            self._video_stream = VideoStream(input_source, usePiCamera=False, 
                                        resolution=(self._cap_frame_width, self._cap_frame_height),
                                        framerate = self._cap_framerate).start()
        
        
        time.sleep(2)
        self._ready = True

        # TODO Create own capture class that doesn't use imutils, but uses picamera
        # TODO Then, we could capture at different resolutions via splitter
        # See: https://picamera.readthedocs.io/en/release-1.13/recipes2.html
        # 4.12. Recording at multiple resolutions

        # Another option is circular video record
        # and grab a frame and put it on the output queue any time it is empty
        # because when the image is analyzed, it is taken off the queue
        # this would ensure it's processing the most recent, not the image from
        # a few seconds ago
        # circular video recording is FAF (fast as f...)
        # So this would, perhaps, make the NCS processing thread the bottle neck
        #self._video_stream.resolution( cap_frame_width, cap_frame_height)

        self._out_queue = out_queue
        self._worker_thread = threading.Thread(target=self._do_work, args=())

        self._ready = True

    def start_processing(self):
        self._stop_flag = False
        self._worker_thread.start()

    def stop_processing(self):
        self._stop_flag = True
        self._worker_thread.join()  
        self._worker_thread = None  

    def _do_work(self):
        if (self._video_stream is None):
            self._ready = False
            self._working = False
            self._stop_flag = True
            return
        print ('Cam processor starting')  
        frame = self._video_stream.read()
        (h, w) = frame.shape[:2]
        print('CAPTURING AT ',w, ' by ',h)      
        while not self._stop_flag:
            try:
                while (not self._stop_flag):
                    # TODO Test perormance here with a pass if self._out_queue.full() is true
                    # Why grab a frame if we can't put it on the stack?
                    # if (self_out_queue.full()): pass
                    # other option is to while (not self._out_queue.full()):
                    frame = self._video_stream.read()
                    # (h, w) = frame.shape[:2]
                    # print('h,w ',h,w)
                    #frame = cv2.resize(frame, (self._cap_frame_width, self._cap_frame_height))
                    # self._out_queue.put(frame, True, self._out_queue_full_sleep)
                   
                    self._out_queue.put_nowait(frame)
                    #print ('frame to queue - length: ', self._out_queue.__len__) 
                    #print ('frame ')
            except:
                time.sleep(self._out_queue_full_sleep)
                pass