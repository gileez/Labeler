# USAGE
# python motion_detector.py
# python motion_detector.py --video videos/example_01.mp4

# import the necessary packages
import sys, os
import YOLO
import argparse
import datetime
import imutils
import time
import cv2
import numpy as np
import copy
from imutils.object_detection import non_max_suppression

# to make sure windows open when in debug mode
#if __debug__:
#    cv2.startWindowThread()

# GLOBALS
ALLOWED_GAP = 2 # allowed tracking gap
HARD_THRESHOLD = 0.8
SOFT_THRESHOLD = 0.5
PROB_THRESH = 20 # a detection above this prob value will initialize a tracker
MAX_TRACK = 60 # lifetime of a tracker in frames
DRAW_TRACKING = True # when saving new frames, draw detection on frame. should only be used for debugging
FULL_FRAME = [0,0,448,448]
CLASSES = ["aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow", "diningtable", "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor"]
TARGET_CLASS = CLASSES.index('person') # class index we are trying to generate data for
term_crit = ( cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 1 ) # termination criteria for openCV CAMShift algorithem
# end GLOBALS

#=============== HELPER FUNCTIONS=====================
def toPoints(bb):
    # returns [X1,Y1,X2,Y2]
    return [bb[0], bb[1], bb[0]+bb[2], bb[1]+bb[3]]

def toRect(bbPoints):
    #returns [X1,Y1,W,H]
    return [bbPoints[0],\
            bbPoints[1],\
            bbPoints[2]-bbPoints[0],\
            bbPoints[3]-bbPoints [1]]
    
def rectArea(bb):
    if not bb:
        return 0
    return bb[2] * bb[3]

def rectUnion(bb1,bb2):
    bb1Points = toPoints(bb1)
    bb2Points = toPoints(bb2)
    rect = toRect(    [min(bb1Points[0],bb2Points[0]),
                    min(bb1Points[1],bb2Points[1]),
                    max(bb1Points[2],bb2Points[2]),
                    max(bb1Points[3],bb2Points[3])])
    return rect
    
def rectOverlap(bb1,bb2,threshhold=0.5):
    # convert to 4 points
    bb1Points = toPoints(bb1)
    bb2Points = toPoints(bb2)
    overlapbb = toRect([max(bb1Points[0], bb2Points[0]),
                        max(bb1Points[1], bb2Points[1]),
                        min(bb1Points[2], bb2Points[2]),
                        min(bb1Points[3], bb2Points[3])])
    if overlapbb[2] <= 0 or overlapbb[3] <= 0 :
        # width or height are negative. no overlap here
        return None
    if threshhold:
        if float(rectArea(overlapbb))/max(rectArea(bb1),rectArea(bb2)) > threshhold:
            return overlapbb
        else:
            return None
    return overlapbb

#==============================END HELPER============================

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-v", "--video", help="path to the video file")
ap.add_argument("-a", "--min-area", type=int, default=200, help="minimum area size")
args = vars(ap.parse_args())

# if the video argument is None, then we are reading from webcam
# if args.get("video", None) is None:
#     camera = cv2.VideoCapture(0)
#     time.sleep(0.25)
# 
# # otherwise, we are reading from a video file
# else:
#     camera = cv2.VideoCapture(args["video"])
vidFile = '/home/gili/dev/YOLO/darknet/1.mp4'
camera = cv2.VideoCapture(vidFile)
USE_PERSON_DETECTOR = 1
if USE_PERSON_DETECTOR:
    hog = cv2.HOGDescriptor()
    hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
# initialize the first frame in the video stream
firstFrame = None
frame_num = 0
tot_new_frames = 0
tracks = {}
results = []

#init network
HOMEDIR = os.path.join(os.path.dirname(__file__),'../')
myYolo=YOLO.YOLO(HOMEDIR+'cfg/yolo-small.cfg', HOMEDIR+'yolo-small.weights')
# loop over the frames of the video
while True:
    # clean up tracks if necessary
    trackDel = []
    for t in tracks.keys():
        for i in range(len(tracks[t])):
            if tracks[t][i]['frames'][-1]['frame_num'] + ALLOWED_GAP < frame_num:
                trackDel.append((t,i))
            elif len(tracks[t][i]['frames']) > MAX_TRACK:
                trackDel.append((t,i))
        if not tracks[t]:
            trackDel.append((t,-1)) #erase the entire track
    for t,i in trackDel:
        if i==-1:
            del(tracks[t])
        else:
            del(tracks[t][i])
    
    # grab the current frame
    (grabbed, frame) = camera.read()
    # if the frame could not be grabbed, then we have reached the end
    # of the video
    if not grabbed:
        print "frame not grabbed. ending..."
        break
    else:
        frame_num+=1
        print "working on frame: %d" %frame_num
    #TODO: figure out if any additional resizing is necessary
    #frame = imutils.resize(frame, height=448)
    # feed forward to network
    yoloRes = myYolo.test(frame)
    yoloRects = {}
    # TODO: add support for multiple occurences
    for y in yoloRes:
        if y.index == TARGET_CLASS and y.prob > PROB_THRESH:
            if TARGET_CLASS in yoloRects:
                raise Exception("attempting to add multiple detections of same class. not yet supported")
            yoloRects[TARGET_CLASS] = [y.x, y.y, y.w, y.h]
            cv2.rectangle(frame, (y.x, y.y), (y.x + y.w, y.y + y.h), (0, 255, 0), 2)
            print "found a %s" %CLASSES[TARGET_CLASS]
    #        
    if not (tracks or yoloRects):
        #theres no reason to perform tracking on this frame...go to next one
        #display current frame 
        cv2.imshow("DETECTION",frame)
        key = cv2.waitKey(1) & 0xFF
    else:
        # check if detection matches a track
        for c in yoloRects.keys():
            matched = False
            if c in tracks:
                # detection class exists in the current frame detections
                for i in range(len(tracks[c])):
                    #TODO: take this track 1 step forward to make comparison better
                    
                    if rectOverlap(tracks[c][i]['frames'][-1]['rect'], yoloRects[c], threshhold=0.5 if tracks[c][i]['frames'][-1].get('SOURCE')=='PERSON_DETECTOR' else 0.7):
                        #got a match between an exising track and a detection. write to results!
                        if 'yoloRect' not in tracks[c][i]['frames'][-1]:
                            tot_new_frames += len(tracks[c][i]['frames'])-2
                            print "======================MATCHED %d frames========================" %(len(tracks[c][i]['frames'])-2)
                            results.append(copy.deepcopy(tracks[c][i]))
                        # reset track history 
                        tracks[c][i]={'frames':[{
                                                    'rect': yoloRects[c],
                                                    'class': c,
                                                    'frame_num': frame_num,
                                                    'yoloRect': yoloRects[c]
                        }]}
                        # remove yoloDetection as it was just used
                        del(yoloRects[c])
                        matched = True
                        break
                    else:
                        print "pass"
                if not matched:
                    # a track with this class exists however no overlap with detections.
                    # should start new tracker in this class
                    tracks[c].append({'frames':[{
                                        'rect': yoloRects[c],
                                        'class': c,
                                        'frame_num': frame_num,
                                        'yoloRect': yoloRects[c]
                    }]})
            else:
                #no such class in tracker. time to make one
                tracks[c] = [{'frames':[{
                                        'rect': yoloRects[c],
                                        'class': c,
                                        'frame_num': frame_num,
                                        'yoloRect': yoloRects[c]
                }]}]
        # prep this frame for tracking
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, np.array((0., 60.,32.)), np.array((180.,255.,255.)))
        # prep for opencv person detector
        boxes=None
        personRects=[]
        if USE_PERSON_DETECTOR:
            # detect people in the image
            (rects, weights) = hog.detectMultiScale(frame, winStride=(4, 4),padding=(8, 8), scale=1.05)
            if rects!=[]:
                for i in range(len(rects)):
                    rects[i] = rectOverlap(rects[i], FULL_FRAME, threshhold=None) # making sure our rects are within the frame
                    if rects[i] is None:
                        raise Exception("you've managed to generate an openCV person detector that is out of bounds. nice one...")
                rects = np.array([[x, y, x + w, y + h] for (x, y, w, h) in rects])
                boxes = non_max_suppression(rects, probs=None, overlapThresh=0.1)
                # draw boxes
                for (xA, yA, xB, yB) in boxes:
                    personRects.append([ xA, yA, xB-xA, yB-yA ])
                    cv2.rectangle(frame, (xA, yA), (xB, yB), (255, 0, 0), 2)
        # step forward on all trackers that don't have a detection on this frame
        for c in tracks:
            for t in tracks[c]:
                if t['frames'][-1]['frame_num']<frame_num: # validates this isn't a YOLO detection
                    matches = []
                    for p in personRects:
                        # best fit rect will be selected by highest overlapping area
                        overlapped = rectOverlap(t['frames'][-1]['rect'], p, threshhold=None)
                        if overlapped:
                            # got a match
                            matches.append({'rect':p,
                                            'overlap':rectArea(overlapped),
                                            'class':c,
                                            'frame_num':frame_num,
                                            'SOURCE':'PERSON_DETECTOR'})
                    # tracking preps
                    track_window = tuple(t['frames'][-1]['rect'])
                    hsv_roi = hsv[int(track_window[1]):int(track_window[1]+track_window[3]),int(track_window[0]):int(track_window[0]+track_window[2])]
                    #hsv_roi =  cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
                    mask_roi = cv2.inRange(hsv_roi, np.array((0., 60.,32.)), np.array((180.,255.,255.)))
                    roi_hist = cv2.calcHist([hsv_roi],[0],mask_roi,[180],[0,180])
                    roi_hist = cv2.normalize(roi_hist,roi_hist,0,255,cv2.NORM_MINMAX)
                    #roi_hist = roi_hist.reshape(-1) #from provided examples. kills the tracker
                    dst = cv2.calcBackProject([hsv],[0],roi_hist,[0,180],1)
                    # apply meanshift to get the new location
                    dst = dst & mask # from provided examples. not explained. not performed in an other examples found
                    ret, track_window = cv2.CamShift(dst, track_window, term_crit)
                    # visual purposes only
                    pts = np.int0(cv2.cv.BoxPoints(ret))
                    cv2.polylines(frame, [pts], True, (0, 0, 255), 2)
                    track_window = rectOverlap(track_window, FULL_FRAME, threshhold=None) # cropping track_window
                    x,y,w,h = track_window
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 0), 2)
                    key = cv2.waitKey(1) & 0xFF
                    overlapped = rectArea(rectOverlap(t['frames'][-1]['rect'], track_window, threshhold=None))
                    matches.append({'rect': track_window,
                                        'class': c,
                                        'frame_num': frame_num,
                                        'SOURCE':'TRACKER',
                                        'overlap':0 if overlapped is None else overlapped})
                    # sort the possible matches and append the one with maximum overlap.
                    # TODO: consult if this is a good approach
                    t['frames'].append(sorted(matches,key=lambda x: x['overlap'], reverse=True)[0])
    
#         # draw the text and timestamp on the frame
#         cv2.putText(frame, "Room Status: {}".format(text), (10, 20),
#             cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
#         cv2.putText(frame, datetime.datetime.now().strftime("%A %d %B %Y %I:%M:%S%p"),
#             (10, frame.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 255), 1)
    
        # show the frame and record if the user presses a key
        cv2.imshow("DETECTION",frame)
        key = cv2.waitKey(1) & 0xFF
    
        # if the `q` key is pressed, break from the lop
        if key == ord("q"):
            break

# cleanup the camera and close any open windows
cv2.destroyAllWindows()
print "got %d results holding %d new frames" %(len(results), tot_new_frames)
# Generating images from results
# prep folder
res_dir="/home/gili/dev/YOLO/LabelerRes"
if not os.path.exists(res_dir):
    os.mkdir(res_dir)
maxFolder=0
dirs = next(os.walk(res_dir))[1]
for d in dirs:
    try:
        if int(d) > maxFolder:
            maxFolder=int(d)
    except:
        pass
res_dir+="/%d" %(maxFolder+1)
os.mkdir(res_dir)
os.mkdir(res_dir+'/labels')
os.mkdir(res_dir+'/images')
# traverse results
image_counter=0
for r in results:
    frame_counter_offset = r['frames'][0]['frame_num']
    c = r['frames'][0]['class']
    #TODO: make img size dynamic
    imWidth = 448.0
    imHeight = 448.0
    for i in range(1,len(r['frames'])-1): # first and last frames are the original detections
        image_counter+=1
        # get frame from original video file
        camera.set(cv2.cv.CV_CAP_PROP_POS_FRAMES,frame_counter_offset + i)
        (grabbed, track_frame) = camera.read()
        # save rect data to txt file of same name
        x,y,w,h = r['frames'][i]['rect']
        if DRAW_TRACKING:
            # draw detection rect to file for debugging purposes
            cv2.rectangle(track_frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        xc = (x + w) / 2.0 / imWidth
        yc = (y + h) / 2.0 / imHeight
        w /= imWidth
        h /= imHeight
        # TODO: we may have more than one detection per image
        rect_label = "%s %f %f %f %f" %(c, xc, yc, w, h)
        txt_path = '%s/labels/%d.txt' %(res_dir,image_counter)
        text_file = open(txt_path, "w")
        text_file.write(rect_label)
        text_file.close()
        # save frame to destination
        cv2.imwrite("%s/images/%d.jpg" %(res_dir, image_counter), track_frame)
camera.release()