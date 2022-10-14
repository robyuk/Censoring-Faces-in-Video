#! /usr/bin/env /usr/bin/python3
#
# Gets each frame in a video file and blurs human faces, then stores the new video as {outvideofile} 
#
# Set the invideofile variable to integer 0 or 1 to capture from a connected webcam (This won't work in repl - you will have to run the code locally on your PC) and then you may want to view the video in real time and limit the number of frames captured as shown in lines 41 to 43)
#
# The faces.xml file contains the rules for looking for faces in an image or frame
#
# Tinker with the detectMultiScale parameters in line 36 to tune the face detection

import cv2

#invideofile='smile.mp4'
invideofile='video-sample.mp4'
outvideofile='blurfaces2.avi'
blurness=(50,50)

faceCascade=cv2.CascadeClassifier('faces.xml')

video=cv2.VideoCapture(invideofile)
#print(type(video))
#
# Video.read() returns a tuple containing a boolean=True and the next frame as a Nimpy array.  If there is no next frame the boolean is False and the frame is None
success, frame=video.read()
width=int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
height=int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
numframes=int(video.get(cv2.CAP_PROP_FRAME_COUNT))
fps=int(video.get(cv2.CAP_PROP_FPS))
outvid=cv2.VideoWriter(outvideofile,cv2.VideoWriter_fourcc(*'DIVX'),fps, (width,height))
#print(width,height,numframes,fps)
#print(frame)

while success:
  faces=faceCascade.detectMultiScale(frame,1.1, 4)  # Detect faces in frame
  for (x,y,w,h) in faces:
     frame[y:y++h,x:x+h]=cv2.blur(frame[y:y+h,x:x+w],blurness)
    
  outvid.write(frame)
  # Uncomment next 3 lines to show the video in real Time  (does not work on Repl - works if you are running o a local machine with a display)
  #cv2.imshow("Recording",frame)
  #if cv2.waitKey(1)==ord('q'):   # Press q to break
  #  break
  success, frame=video.read()  # Get the next frame
  
outvid.release()
video.release()
cv2.destroyAllWindows
