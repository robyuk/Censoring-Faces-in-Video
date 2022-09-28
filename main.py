#! /usr/bin/env /usr/bin/python3
#
# Gets each frame in a video file and merges a cat face into human faces, then stores the new video as {outvideofile} 
#
# Set the invideofile variable to integer 0 or 1 to capture from a connected webcam (This won't work in repl - you will have to run the code locally on your PC) and then you may want to view the video in real time and limit the number of frames captured as shown in lines 41 to 43)
#
# The faces.xml file contains the rules for looking for faces in an image or frame
#
# Tinker with the detectMultiScale parameters in line 36 to tune the face detection

import cv2

invideofile='smile.mp4'
outvideofile='catfaces.avi'
catfacefile='CatFace3.png'
imageDir='images'
faceCascade=cv2.CascadeClassifier('faces.xml')
boxcolour=(255,255,255)  # (r,g,b) colour of box around faces
boxthick=4  # Thicknesss of the box line around faces

def calculateDims(scalePercent, width, height):
  newHeight=int(height*scalePercent/100)
  newWidth=int(width*scalePercent/100)
  return (newWidth, newHeight)

def resize(image, scalePercent, resizedPath):
  print(f'Original Dimensions: {image.shape}')

def imblend(im1,x,y,im2,w,h):
  # Resize image. im2 to w,h and blend it into im1 at x,y
  resizedim2=cv2.resize(im2, (w,h) )  # Resize image im2 to fit w,h
  # im1Place is the portion of the image where im2 will be blended.
  im1Place=im1[y:y+w,x:x+h]
  # create a new image that is the blend of the portion and im2
  #blend=cv2.addWeighted(im1Place,0.7, resizedim2,0.3, 0)
  sum=im1Place+resizedim2
  im1[y:y+w,x:x+h]=sum  # Put the blend in im1
  return im1

catface=cv2.imread(catfacefile)  # Get the cat face image
catfacedim=catface.shape
#print(f'{catfacefile}: Original Dimensions: {catface.shape}')

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
  faces=faceCascade.detectMultiScale(frame,1.4, 4)  # Detect faces in frame
  for (x,y,w,h) in faces:
     newframe=imblend(frame,x,y,catface,w,h)
    
  outvid.write(newframe)
  # Uncomment next 3 lines to show the video in real Time  (does not work on Repl - works if you are running o a local machine with a display)
  #cv2.imshow("Recording",frame)
  #if cv2.waitKey(1)==ord('q'):   # Press q to break
  #  break
  success, frame=video.read()  # Get the next frame
  
outvid.release()
video.release()
cv2.destroyAllWindows
