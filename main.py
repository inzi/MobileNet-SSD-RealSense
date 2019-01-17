import sys, os
if sys.version_info.major < 3 or sys.version_info.minor < 4:
    print("Please using python3.4 or greater!")
    sys.exit(1)
import numpy as np
import cv2, io, time, argparse, re
from os import system
from os.path import isfile, join
from time import sleep
import multiprocessing as mp
from openvino.inference_engine import IENetwork, IEPlugin
import heapq
import threading
from imutils.video import VideoStream

#Django connect
api_path = '/home/pi/workspace/bbw3.2/bb' #os.path.abspath('~','workspace','bbw3.2','bb', 'bb')
# print(api_path)  # home/foo/work

sys.path.append(api_path) 

# print(sys.path)

import django
from django.conf import settings
from bb import settings as bbsettings
#os.environ['DJANGO_SETTINGS_MODULE']='bb.settings'


settings.configure(DATABASES=bbsettings.DATABASES, DEBUG=True)

django.setup()

#from captures.models import Capture
print ('starting... {}'.format(__name__))



pipeline = None
lastresults = None
threads = []
processes = []
frameBuffer = None
results = None
fps = ""
detectfps = ""
framecount = 0
detectframecount = 0
frametimestamps = []
inferredframesfordisplay = []
time1 = 0
time2 = 0
cam = None
camera_mode = 0
camera_width = 320
camera_height = 240
window_name = ""
background_transparent_mode = 0
ssd_detection_mode = 1
face_detection_mode = 0
elapsedtime = 0.0
background_img = None
depth_sensor = None
depth_scale = 1.0
align_to = None
align = None

LABELS = [['background',
           'aeroplane', 'bicycle', 'bird', 'boat',
           'bottle', 'bus', 'car', 'cat', 'chair',
           'cow', 'diningtable', 'dog', 'horse',
           'motorbike', 'person', 'pottedplant',
           'sheep', 'sofa', 'train', 'tvmonitor'],
          ['background', 'face']]



LABELS = [['background',
           'vehicle','label2','label3'],['background','ar2-1','ar2-2']]
          

def camThread(LABELS, results, frameBuffer, camera_mode, camera_width, camera_height, background_transparent_mode, background_img, vidfps):
    global fps
    global detectfps
    global lastresults
    global framecount
    global detectframecount
    global time1
    global time2
    global cam
    global window_name
    global depth_scale
    global align_to
    global align

    global frametimestamps
    global inferredframesfordisplay

    # Configure depth and color streams
    #  Or
    # Open USB Camera streams
    # # if camera_mode == 0:
    # #     pipeline = rs.pipeline()
    # #     config = rs.config()
    # #     config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, vidfps)
    # #     config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, vidfps)
    # #     profile = pipeline.start(config)
    # #     depth_sensor = profile.get_device().first_depth_sensor()
    # #     depth_scale = depth_sensor.get_depth_scale()
    # #     align_to = rs.stream.color
    # #     align = rs.align(align_to)
    # #     window_name = "RealSense"
    # # elif camera_mode == 1:
    # #     cam = cv2.VideoCapture(0)
    # #     if cam.isOpened() != True:
    # #         print("USB Camera Open Error!!!")
    # #         sys.exit(0)
    # #     cam.set(cv2.CAP_PROP_FPS, vidfps)
    # #     cam.set(cv2.CAP_PROP_FRAME_WIDTH, camera_width)
    # #     cam.set(cv2.CAP_PROP_FRAME_HEIGHT, camera_height)
    # #     window_name = "USB Camera"


    # 1296x972
    # 1296, 730
    # 800, 600 - 60 (12+ fps) (GPU upscaled 640x480)
    # 800,600 - 20 10+
    # 1024, 768 - 42 (6+ fps)
    # 1024, 768 - 20 (6+)
    # 640, 480 - 90 (17+ fps)
    # cam =  VideoStream(usePiCamera=True, 
    #                                     resolution=(640, 480),
    #                                     framerate = 90).start()
    window_name = "picam"

    use_file = True
    
    print ('opening...')
    if use_file:
        cam = cv2.VideoCapture('file:///home/pi/Videos/traffic1-xvid.avi')

    else:
        cam =  VideoStream(usePiCamera=True, 
                                        resolution=(640, 480),
                                        framerate = 90).start()

    # cam = VideoStream('~/Videos/traffic1.mp4', resolution=(640,480), framerate=30).start()
    # cam = cv2.VideoCapture('~/Videos/traffic1.mpg')
    # cam = cv2.VideoCapture('file:///home/pi/Videos/traffic1-xvid.avi')
    print ('opened...')
    cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)

    inference_engine_responded = False
    print ('warming up...')
    thisframe_timestamp = 0 
    last_frame_shown = 0
    while True:
        t1 = time.perf_counter()

        # 0:= RealSense Mode
        # 1:= USB Camera Mode

        if not frameBuffer.full():
            # continue
            #frameBuffer.get()
            if use_file:
                _success, color_image = cam.read()
                color_image = cv2.resize(color_image, (640,480))
            else:
                # USB Camera Stream Read
                color_image = cam.read()

            frames = color_image

            # height = color_image.shape[0]
            # width = color_image.shape[1]
            width = 640 #int(cam.get(3)) #int(cam.get(cv2.CV_CAP_PROP_FRAME_WIDTH) )  # float
            height = 480 #int(cam.get(4)) #int(cam.get(cv2.CV_CAP_PROP_FRAME_HEIGHT) )# float

            thisframe_timestamp = int(round(time.time() * 1000))
            # thisframe_timestamp += 1
            # print ('Generated: {}                             g'.format(thisframe_timestamp))
            frameBuffer.put([color_image.copy(), thisframe_timestamp])
            heapq.heappush(frametimestamps, thisframe_timestamp)
            # print (' -- frametimestamps len: {}'.format(len(frametimestamps)))
        res = None

        if not results.empty():
            orig_img, res, inf_frame_timestamp = results.get(False)

            #print ('{} Results'.format(len(res)))
            # if inference_engine_responded == False:
            #     #clear frametimestamps
            #     frametimestamps = []
            #     heapq.heappush(frametimestamps, inf_frame_timestamp)
            #     heapq.heappush
            #     inference_engine_responded = True
            detectframecount += 1
            imdraw = overlay_on_image(orig_img, inf_frame_timestamp,res, LABELS, camera_mode, background_transparent_mode,
                                      background_img, depth_scale=depth_scale, align=align, )
            lastresults = res

            heapq.heappush(inferredframesfordisplay, (inf_frame_timestamp, imdraw))
            _infts, _imgdraw = inferredframesfordisplay[0]

            # print ('                                   showing {}'.format(_infts))
            if len(frametimestamps)>0:
                if _infts == frametimestamps[0]:
                    # print ('expecting {} got {}  last {}    showing {}'.format(frametimestamps[0], inf_frame_timestamp, last_frame_shown, _infts))
                    heapq.heappop(frametimestamps)  
                    cv2.imshow(window_name, cv2.resize(imdraw, (width, height)))
                    heapq.heappop(inferredframesfordisplay)
                    last_frame_shown = _infts


            # heapq.heappush(inferredframesfordisplay, (inf_frame_timestamp, imdraw))
            # play_to = last_frame_shown
            # while len(inferredframesfordisplay)>0 and inferredframesfordisplay[0][:1][0]<=play_to:
            #             _infts, _imgdraw = heapq.heappop(inferredframesfordisplay)
            #             if len(frametimestamps)>0:
            #                 _lowts = frametimestamps[0]
            #             else:
            #                 _lowts = _infts

            #             print ('                                   showing {}'.format(_infts))

            #             cv2.imshow(window_name, cv2.resize(imdraw, (width, height)))
            #             if len(frametimestamps)>1:
            #                 print ('popping {}                               p'.format(_lowts))
            #                 heapq.heappop(frametimestamps)
            #             last_frame_shown = _infts

            # catchup = True
            # while catchup:
            #     if len(inferredframesfordisplay)>0:
            #         _infts, _imgdraw = inferredframesfordisplay[0]
            #         if len(frametimestamps)>0:
            #             _lowts = frametimestamps[0]
            #         else:
            #             _lowts = _infts

            #         # print ('Testing {} against lowest {}'.format(_infts, _lowts))
            #         if _infts < _lowts:
            #             print ('                                   showing {}'.format(_infts))
            #             cv2.imshow(window_name, cv2.resize(imdraw, (width, height)))
            #             heapq.heappop(inferredframesfordisplay)

            #         if _infts == _lowts:
            #             if len(frametimestamps)>1:
            #                 # print ('popping {}                               p'.format(_lowts))
            #                 heapq.heappop(frametimestamps)
            #             if len(frametimestamps)>0:
            #                 print ('new low: {}'.format(frametimestamps[0]))
            #             # print ('lowts {}   :::  lowinfts {}'.format(len(frametimestamps), len(inferredframesfordisplay)))
            #             catchup = False
                    
            #         if _infts > _lowts:
            #             print ('ooo frame {} - lowts {}   :::  lowinfts {}'.format(_infts, len(frametimestamps), len(inferredframesfordisplay)))
            #             catchup = False
            #     else:
            #         # print ('lowts {}   ---  lowinfts none'.format(len(frametimestamps)))
            #         catchup = False
                        

            # lowts = frametimestamps[0]
            # print ('lowts: {}  inf_frame_timestamp: {}'.format(lowts, inf_frame_timestamp))
            # if inf_frame_timestamp == lowts:
            #     heapq.heappop(frametimestamps)

            #     cv2.imshow(window_name, cv2.resize(imdraw, (width, height)))
            #     # catch up any frames

            # lowts = frametimestamps[0]
            # waiting_frame_ts = lowts
            # while True:
            #     _waiting_frame_ts, _i = inferredframesfordisplay[0]
            #     if _waiting_frame_ts <= lowts:
            #         _inf, _imgdraw = heapq.heappop(inferredframesfordisplay)
            #         cv2.imshow(window_name, cv2.resize(imdraw, (width, height)))
            #     if _waiting_frame_ts > lowts:
            #         waiting_frame_ts = _waiting_frame_ts
            #         break
            # heapq.heappush(frametimestamps, waiting_frame_ts)


            # while 
            #     waiting_frame_ts = inferredframesfordisplay[0]
            #     if waiting_frame_ts <= lowts:
            #         _inf, _imgdraw = heapq.heappop(inferredframesfordisplay)
            #         cv2.imshow(window_name, cv2.resize(imdraw, (width, height)))
                


        else:
            # print ('waiting on results')
            continue
            # imdraw = overlay_on_image(frames, lastresults, LABELS, camera_mode, background_transparent_mode,
            #                           background_img, depth_scale=depth_scale, align=align)

        # cv2.imshow(window_name, cv2.resize(imdraw, (width, height)))

        if cv2.waitKey(1)&0xFF == ord('q'):
            # Stop streaming
            if pipeline != None:
                pipeline.stop()
            sys.exit(0)

        ## Print FPS
        framecount += 1
        if framecount >= 15:
            fps       = "(Playback) {:.1f} FPS".format(time1/15)
            detectfps = "(Detection) {:.1f} FPS".format(detectframecount/time2)
            framecount = 0
            detectframecount = 0
            time1 = 0
            time2 = 0
        t2 = time.perf_counter()
        elapsedTime = t2-t1
        time1 += 1/elapsedTime
        time2 += elapsedTime


# l = Search list
# x = Search target value
def searchlist(l, x, notfoundvalue=-1):
    if x in l:
        return l.index(x)
    else:
        return notfoundvalue


def async_infer(ncsworker):

    #ncsworker.skip_frame_measurement()

    while True:
        ncsworker.predict_async()


class NcsWorker(object):

    def __init__(self, devid, frameBuffer, results, camera_mode, camera_width, camera_height, number_of_ncs, vidfps, skpfrm):
        self.devid = devid
        self.frameBuffer = frameBuffer
        # self.model_xml = "./lrmodel/MobileNetSSD/MobileNetSSD_deploy.xml"
        # self.model_bin = "./lrmodel/MobileNetSSD/MobileNetSSD_deploy.bin"
        self.model_xml = "./lrmodel/MobileNetSSD/vehicle-detection-adas-0002.xml"
        self.model_bin = "./lrmodel/MobileNetSSD/vehicle-detection-adas-0002.bin"
        # self.model_xml = "./lrmodel/MobileNetSSD/vehicle-license-plate-detection-barrier-0106.xml"
        # self.model_bin = "./lrmodel/MobileNetSSD/vehicle-license-plate-detection-barrier-0106.bin"


        self.camera_width = camera_width
        self.camera_height = camera_height
        self.num_requests = 4
        self.inferred_request = [0] * self.num_requests
        self.orig_images = [0] * self.num_requests
        self.lowest_reqnum = 1
        self.heap_request = []
        self.inferred_cnt = 0
        self.plugin = IEPlugin(device="MYRIAD")
        self.net = IENetwork(model=self.model_xml, weights=self.model_bin)
        self.input_blob = next(iter(self.net.inputs))
        self.exec_net = self.plugin.load(network=self.net, num_requests=self.num_requests)
        self.results = results
        self.camera_mode = camera_mode
        self.number_of_ncs = number_of_ncs
        if self.camera_mode == 0:
            self.skip_frame = skpfrm
        else:
            self.skip_frame = 0
        self.roop_frame = 0
        self.vidfps = vidfps

    def image_preprocessing(self, color_image):

        # prepimg = cv2.resize(color_image, (300, 300))
        prepimg = cv2.resize(color_image, (672, 384)) #vehicle-detection-adas-0002
        prepimg = prepimg[np.newaxis, :, :, :]     # Batch size axis add
        prepimg = prepimg.transpose((0, 3, 1, 2))  # NHWC to NCHW
       
        # prepimg = cv2.resize(color_image, (300,300)) #vehicle-license-plate-detection-barrier-0106
        # prepimg = prepimg - 127.5
        # prepimg = prepimg * 0.007843
        # prepimg = prepimg[np.newaxis, :, :, :]     # Batch size axis add
        # prepimg = prepimg.transpose((0, 3, 1, 2))  # NHWC to NCHW (new axis, color, height width)

        return prepimg


    def predict_async(self):
        try:

            if self.frameBuffer.empty():
                #print ('waiting on cam')
                return

            # self.roop_frame += 1
            # if self.roop_frame <= self.skip_frame:
            #    self.frameBuffer.get()
            #    return
            # self.roop_frame = 0

            # orig_img, frame_timestamp = self.frameBuffer.get()
            # print ('pulled {}                       <'.format(frame_timestamp))
            # prepimg = self.image_preprocessing(orig_img)
            reqnum = searchlist(self.inferred_request, 0)

            if reqnum > -1:
                orig_img, frame_timestamp = self.frameBuffer.get()
                # print ('pulled {}                       <'.format(frame_timestamp))
                prepimg = self.image_preprocessing(orig_img)
                # reqnum = searchlist(self.inferred_request, 0)
                self.exec_net.start_async(request_id=reqnum, inputs={self.input_blob: prepimg})
                self.inferred_request[reqnum] = 1
                self.orig_images[reqnum] = orig_img
                self.inferred_cnt += 1
                # print ('started {} - inferred_cnt {}'.format(reqnum, self.inferred_cnt))
                if self.inferred_cnt == sys.maxsize: #This basically says if the # is the max # the OS can support, go back to zero.
                    self.inferred_request = [0] * self.num_requests
                    self.heap_request = []
                    self.inferred_cnt = 0
                heapq.heappush(self.heap_request, (self.inferred_cnt, reqnum, frame_timestamp))
            # else:
            #     print('Passing {}              !!!!!!!!!!!!!!!!'.format(frame_timestamp))
            # to access lowest heap item without pop heap[0]
            # cnt, dev = self.heap_request[0]

            # print (self.heap_request[0])
            if len(self.heap_request)>0:
                cnt, dev, inf_frame_timestamp = heapq.heappop(self.heap_request)
                #print ('heappop cnt {} dev {}'.format(cnt, dev))

                dev_wait_state = self.exec_net.requests[dev].wait(0)
                if dev_wait_state == 0:
                    #done with this request
                    self.exec_net.requests[dev].wait(-1)

                    out = self.exec_net.requests[dev].outputs["detection_out"].flatten()
                    # out = self.exec_net.requests[dev].outputs["DetectionOutput_"].flatten() #vehicle-license-plate-detection-barrier-0106
                    # print ('Completed {}               >'.format(inf_frame_timestamp))
                    self.results.put([self.orig_images[dev],[out], inf_frame_timestamp])
                    self.inferred_request[dev] = 0
                    self.orig_images[dev] = None
                else:
                    # destroy self.
                    # print ('repush - dev wait state {}'.format(dev_wait_state))
                    # print ('Passed {}                               ?'.format(inf_frame_timestamp))
                    heapq.heappush(self.heap_request, (cnt, dev, inf_frame_timestamp))

        except:
            import traceback
            traceback.print_exc()


def inferencer(results, frameBuffer, ssd_detection_mode, face_detection_mode, camera_mode, camera_width, camera_height, number_of_ncs, vidfps, skpfrm):

    # Init infer threads
    threads = []
    for devid in range(number_of_ncs):
        thworker = threading.Thread(target=async_infer, args=(NcsWorker(devid, frameBuffer, results, camera_mode, camera_width, camera_height, number_of_ncs, vidfps, skpfrm),))
        thworker.start()
        threads.append(thworker)

    for th in threads:
        th.join()


def overlay_on_image(frames, inf_frame_timestamp, object_infos, LABELS, camera_mode, background_transparent_mode, background_img, depth_scale=1.0, align=None):

    try:

        # 0:=RealSense Mode, 1:=USB Camera Mode

        color_image = frames

        if isinstance(object_infos, type(None)):
            # 0:= No background transparent, 1:= Background transparent
            if background_transparent_mode == 0:
                return color_image
            elif background_transparent_mode == 1:
                return background_img

        # Show images
        height = color_image.shape[0]
        width = color_image.shape[1]
        entire_pixel = height * width
        occupancy_threshold = 0.9

        if background_transparent_mode == 0:
            img_cp = color_image.copy()
        elif background_transparent_mode == 1:
            img_cp = background_img.copy()

        # for (object_info, LABEL) in zip(object_infos, LABELS):
        for object_info in object_infos:
            

            drawing_initial_flag = True

            for box_index in range(100):
                if object_info[box_index + 1] == 0.0:
                    break
                base_index = box_index * 7
                if (not np.isfinite(object_info[base_index]) or
                    not np.isfinite(object_info[base_index + 1]) or
                    not np.isfinite(object_info[base_index + 2]) or
                    not np.isfinite(object_info[base_index + 3]) or
                    not np.isfinite(object_info[base_index + 4]) or
                    not np.isfinite(object_info[base_index + 5]) or
                    not np.isfinite(object_info[base_index + 6])):
                    continue

                x1 = max(0, int(object_info[base_index + 3] * height))
                y1 = max(0, int(object_info[base_index + 4] * width))
                x2 = min(height, int(object_info[base_index + 5] * height))
                y2 = min(width, int(object_info[base_index + 6] * width))

                object_info_overlay = object_info[base_index:base_index + 7]

                # 0:= No background transparent, 1:= Background transparent
                # if background_transparent_mode == 0:
                #     min_score_percent = 30
                # elif background_transparent_mode == 1:
                #     min_score_percent = 20

                min_score_percent = 10

                source_image_width = width
                source_image_height = height

                base_index = 0
                class_id = object_info_overlay[base_index + 1]
                # print ('class_id: ',class_id)
                LABEL = LABELS[int(class_id)]
                percentage = int(object_info_overlay[base_index + 2] * 100)
                if (percentage <= min_score_percent):
                    continue

                box_left = int(object_info_overlay[base_index + 3] * source_image_width)
                box_top = int(object_info_overlay[base_index + 4] * source_image_height)
                box_right = int(object_info_overlay[base_index + 5] * source_image_width)
                box_bottom = int(object_info_overlay[base_index + 6] * source_image_height)

                # 0:=RealSense Mode, 1:=USB Camera Mode
                label_text = LABEL[int(class_id)] + " (" + str(percentage) + "%)"

                box_color = (255, 128, 0)
                box_thickness = 1
                cv2.rectangle(img_cp, (box_left, box_top), (box_right, box_bottom), box_color, box_thickness)
                label_background_color = (125, 175, 75)
                label_text_color = (255, 255, 255)
                label_size = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
                label_left = box_left
                label_top = box_top - label_size[1]
                if (label_top < 1):
                    label_top = 1
                label_right = label_left + label_size[0]
                label_bottom = label_top + label_size[1]
                cv2.rectangle(img_cp, (label_left - 1, label_top - 1), (label_right + 1, label_bottom + 1), label_background_color, -1)
                cv2.putText(img_cp, label_text, (label_left, label_bottom), cv2.FONT_HERSHEY_SIMPLEX, 0.5, label_text_color, 1)
                frame_text_color = (200,100,50)
                cv2.putText(img_cp, str(inf_frame_timestamp), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, frame_text_color, 1)

      
        cv2.putText(img_cp, fps,       (width-170,15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (38,0,255), 1, cv2.LINE_AA)
        cv2.putText(img_cp, detectfps, (width-170,30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (38,0,255), 1, cv2.LINE_AA)
        return img_cp

    except:
        import traceback
        traceback.print_exc()


if __name__ == '__main__':

    print ('__main__')
    parser = argparse.ArgumentParser()
    parser.add_argument('-mod','--mode',dest='camera_mode',type=int,default=0,help='Camera Mode. (0:=RealSense Mode, 1:=USB Camera Mode. Defalut=0)')
    parser.add_argument('-wd','--width',dest='camera_width',type=int,default=320,help='Width of the frames in the video stream. (USB Camera Mode Only. Default=320)')
    parser.add_argument('-ht','--height',dest='camera_height',type=int,default=240,help='Height of the frames in the video stream. (USB Camera Mode Only. Default=240)')
    parser.add_argument('-tp','--transparent',dest='background_transparent_mode',type=int,default=0,help='TransparentMode. (RealSense Mode Only. 0:=No background transparent, 1:=Background transparent)')
    parser.add_argument('-sd','--ssddetection',dest='ssd_detection_mode',type=int,default=1,help='[Future functions] SSDDetectionMode. (0:=Disabled, 1:=Enabled Default=1)')
    parser.add_argument('-fd','--facedetection',dest='face_detection_mode',type=int,default=0,help='[Future functions] FaceDetectionMode. (0:=Disabled, 1:=Full, 2:=Short Default=0)')
    parser.add_argument('-numncs','--numberofncs',dest='number_of_ncs',type=int,default=1,help='Number of NCS. (Default=1)')
    parser.add_argument('-vidfps','--fpsofvideo',dest='fps_of_video',type=int,default=30,help='FPS of Video. (USB Camera Mode Only. Default=30)')
    parser.add_argument('-skpfrm','--skipframe',dest='number_of_frame_skip',type=int,default=7,help='Number of frame skip. (RealSense Mode Only. Default=7)')

    args = parser.parse_args()
    print ('past args')
    camera_mode   = args.camera_mode
    camera_width  = args.camera_width
    camera_height = args.camera_height
    background_transparent_mode = args.background_transparent_mode
    ssd_detection_mode = args.ssd_detection_mode
    face_detection_mode = args.face_detection_mode
    number_of_ncs = args.number_of_ncs
    vidfps = args.fps_of_video
    skpfrm = args.number_of_frame_skip

    # 0:=RealSense Mode, 1:=USB Camera Mode
    if camera_mode != 0 and camera_mode != 1:
        print("Camera Mode Error!! " + str(camera_mode))
        sys.exit(0)

    if camera_mode != 0 and background_transparent_mode == 1:
        background_transparent_mode = 0

    if background_transparent_mode == 1:
        background_img = np.zeros((camera_height, camera_width, 3), dtype=np.uint8)

        if face_detection_mode != 0:
            ssd_detection_mode = 0

    if ssd_detection_mode == 0 and face_detection_mode != 0:
        del(LABELS[0])

    try:

        mp.set_start_method('forkserver')
        frameBuffer = mp.Queue(10)
        results = mp.Queue()

        # Start streaming
        p = mp.Process(target=camThread,
                       args=(LABELS, results, frameBuffer, camera_mode, camera_width, camera_height, background_transparent_mode, background_img, vidfps),
                       daemon=True)
        p.start()
        processes.append(p)

        # Start detection MultiStick
        # Activation of inferencer
        p = mp.Process(target=inferencer,
                       args=(results, frameBuffer, ssd_detection_mode, face_detection_mode, camera_mode, camera_width, camera_height, number_of_ncs, vidfps, skpfrm),
                       daemon=True)
        p.start()
        processes.append(p)

        while True:
            sleep(1)

    except:
        import traceback
        traceback.print_exc()
    finally:
        for p in range(len(processes)):
            processes[p].terminate()

        print("\n\nFinished\n\n")
