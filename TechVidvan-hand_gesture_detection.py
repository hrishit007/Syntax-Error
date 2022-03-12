# TechVidvan hand Gesture Recognizer

# import necessary packages

import cv2
import numpy as np
import mediapipe as mp
import tensorflow as tf
import screen_brightness_control as sbc
import time
import pyautogui

from tensorflow.keras.models import load_model
from math import hypot
from ctypes import cast, POINTER
from comtypes import CLSCTX_ALL
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume
#count for opening teams only once
count=0
# initialize mediapipe
mpHands = mp.solutions.hands
hands = mpHands.Hands(max_num_hands=1, min_detection_confidence=0.7)
mpDraw = mp.solutions.drawing_utils

# Load the gesture recognizer model
model = load_model('mp_hand_gesture')

# Load class names
f = open('gesture.names', 'r')
classNames = f.read().split('\n')
f.close()
print(classNames)


# Initialize the webcam
cap = cv2.VideoCapture(0)

while True:
    # Read each frame from the webcam
    _, frame = cap.read()

    x, y, c = frame.shape

    # Flip the frame vertically
    frame = cv2.flip(frame, 1)
    framergb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Get hand landmark prediction
    result = hands.process(framergb)

    # print(result)
    
    className = ''

    # post process the result
    if result.multi_hand_landmarks:
        landmarks = []
        for handslms in result.multi_hand_landmarks:
            for lm in handslms.landmark:
                # print(id, lm)
                lmx = int(lm.x * x)
                lmy = int(lm.y * y)

                landmarks.append([lmx, lmy])

            # Drawing landmarks on frames
            mpDraw.draw_landmarks(frame, handslms, mpHands.HAND_CONNECTIONS)
            # var1=landmarks[4][0]
            # print("var1=",var1)
            # var2=landmarks[4][1]
            # print("var2=",var2)
            # Predict gesture
            prediction = model.predict([landmarks])
            #print(prediction)
            classID = np.argmax(prediction)
            className = classNames[classID]
            # print(className);

    # show the prediction on the frame
    cv2.putText(frame, className, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 
                   1, (0,255,0), 2, cv2.LINE_AA)
    if(className=="thumbs up"):
        print("Working");
        devices = AudioUtilities.GetSpeakers()
        interface = devices.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
        volume = cast(interface, POINTER(IAudioEndpointVolume))
        volbar=400
        volper=0
 
        volMin,volMax = volume.GetVolumeRange()[:2]
 
        while True:
            success,img = cap.read() #If camera works capture an image
            imgRGB = cv2.cvtColor(img,cv2.COLOR_BGR2RGB) #Convert to rgb
    
            #Collection of gesture information
            results = hands.process(imgRGB) #completes the image processing.
 
            lmList = [] #empty list
            if results.multi_hand_landmarks: #list of all hands detected.
                #By accessing the list, we can get the information of each hand's corresponding flag bit
                for handlandmark in results.multi_hand_landmarks:
                    for id,lm in enumerate(handlandmark.landmark): #adding counter and returning it
                        # Get finger joint points
                        h,w,_ = img.shape
                        cx,cy = int(lm.x*w),int(lm.y*h)
                        lmList.append([id,cx,cy]) #adding to the empty list 'lmList'
                    mpDraw.draw_landmarks(img,handlandmark,mpHands.HAND_CONNECTIONS)
    
            if lmList != []:
                #getting the value at a point
                                #x      #y
                x1,y1 = lmList[4][1],lmList[4][2]  #thumb
                x2,y2 = lmList[8][1],lmList[8][2]  #index finger
                #creating circle at the tips of thumb and index finger
                cv2.circle(img,(x1,y1),13,(255,0,0),cv2.FILLED) #image #fingers #radius #rgb
                cv2.circle(img,(x2,y2),13,(255,0,0),cv2.FILLED) #image #fingers #radius #rgb
                cv2.line(img,(x1,y1),(x2,y2),(255,0,0),3)  #create a line b/w tips of index finger and thumb
 
                length = hypot(x2-x1,y2-y1) #distance b/w tips using hypotenuse
        # from numpy we find our length,by converting hand range in terms of volume range ie b/w -63.5 to 0
                vol = np.interp(length,[30,350],[volMin,volMax]) 
                volbar=np.interp(length,[30,350],[400,150])
                volper=np.interp(length,[30,350],[0,100])
        
        
                # print(vol,int(length))
                volume.SetMasterVolumeLevel(vol, None)
        
                # Hand range 30 - 350
                # Volume range -63.5 - 0.0
                #creating volume bar for volume level 
                cv2.rectangle(img,(50,150),(85,400),(0,0,255),4) # vid ,initial position ,ending position ,rgb ,thickness
                cv2.rectangle(img,(50,int(volbar)),(85,400),(0,0,255),cv2.FILLED)
                cv2.putText(img,f"{int(volper)}%",(10,40),cv2.FONT_ITALIC,1,(0, 255, 98),3)
                #tell the volume percentage ,location,font of text,length,rgb color,thickness
            cv2.imshow('Image',img) #Show the video 
            if cv2.waitKey(1) & 0xff==ord(' '): #By using spacebar delay will stop
                break
        
        # cap.release()     #stop cam       
        # cv2.destroyAllWindows() #close window
    if(className=="peace"):
        while True:
            success,img = cap.read()
            imgRGB = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
            results = hands.process(imgRGB)
 
            lmList = []
            if results.multi_hand_landmarks:
                for handlandmark in results.multi_hand_landmarks:
                    for id,lm in enumerate(handlandmark.landmark):
                        h,w,_ = img.shape
                        cx,cy = int(lm.x*w),int(lm.y*h)
                        lmList.append([id,cx,cy])
                    mpDraw.draw_landmarks(img,handlandmark,mpHands.HAND_CONNECTIONS)
    
            if lmList != []:
                x1,y1 = lmList[4][1],lmList[4][2]
                x2,y2 = lmList[8][1],lmList[8][2]

                cv2.circle(img,(x1,y1),4,(255,0,0),cv2.FILLED)
                cv2.circle(img,(x2,y2),4,(255,0,0),cv2.FILLED)
                cv2.line(img,(x1,y1),(x2,y2),(255,0,0),3)
 
                length = hypot(x2-x1,y2-y1)
 
                bright = np.interp(length,[15,220],[0,100])
                # print(bright,length)
                sbc.set_brightness(int(bright))
        
                # Hand range 15 - 220
                # Brightness range 0 - 100
 
            cv2.imshow('Image',img)
            if cv2.waitKey(1) & 0xff==ord('q'):
                break
    if(className == 'okay'):
        im = pyautogui.screenshot()
        im.save("SS1.jpg")
    if(className=="thumbs down"):
        cv2.putText(frame, "thumbs down-exit", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 
                   1, (0,255,0), 2, cv2.LINE_AA)
        cv2.imshow("Output", frame) 

        time.sleep(2)
        break
    if(className=="smile"):
        cap.release()
        import subprocess
        subprocess.run('start microsoft.windows.camera:', shell=True)
    if(className=="call me"):
        # import subprocess
        # subprocess.Popen("C:\Users\Hrishit\Downloads\Teams_windows_x64.exe")
        if(count==0):
            import os
            os.startfile("C:/Users/Hrishit/Downloads/Teams_windows_x64.exe")
            count+=1
        
    if(className=="rock"):
        import os
        os.system("Notepad")
    # Show the final output
    
    cv2.imshow("Output", frame) 
    
    if cv2.waitKey(1) == ord('q'):
        break

# release the webcam and destroy all active windows
cap.release()

cv2.destroyAllWindows()
# import os
# os.system("microsoft.windows.camera")

