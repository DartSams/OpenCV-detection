import cv2 as cv
import numpy as np
import random
import pygame
import mediapipe as mp #opensource machine learning library of detected datasets

camera = cv.VideoCapture(0) #gets the device default camera

class Game:
    def __init__(self,camera):
        self.camera = camera
        # self.face_cascade = cv.CascadeClassifier('facial recognition\haarcascades\haarcascade_finger.xml') #for custom haarcascades
        self.face_cascade = cv.CascadeClassifier(cv.data.haarcascades + 'haarcascade_frontalface_default.xml')
        self.points = []
        self.mphands = mp.solutions.hands
        self.hands = self.mphands.Hands(max_num_hands=1)
        self.mp_drawing = mp.solutions.drawing_utils
        

    
    def detect(self,frame,grayscale,object):
        if object == "face":
            faces=self.face_cascade.detectMultiScale(grayscale,1.3,5) #detect the face on a grayscale image
            for x,y,w,h in faces:
                cv.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),5) #draws a rectangle on the detected face boundary
                self.face_center_x,self.face_center_y = int(x+w/2),int(y+h/2)
                # print(int(face_center_x),int(face_center_y))
                self.points.append((int(self.face_center_x),int(self.face_center_y)))
                # cv.circle(frame,(self.face_center_x,self.face_center_y),10,(0,0,255),-1) #draw a circle at the face center point


    def draw_circle(self,frame,pos,thickness,color):
        return cv.circle(frame,pos,thickness,color,-1)
    
    def find_distance(self,obj1_pos,obj2_pos): #distance formula sqrt((x2-x1)**2+(y2-y1)**2)
        try:
            dist = pygame.math.Vector2(obj1_pos[0],obj1_pos[1]).distance_to((obj2_pos[0],obj2_pos[1]))
        except:
            dist = pygame.math.Vector2(obj1_pos.x,obj1_pos.y).distance_to((obj2_pos.x,obj2_pos.y))

        finally:
            return dist

    def draw_contours(self,gray):
        r,thresh = cv.threshold(gray,127,255,0)
        contours, hierarchy = cv.findContours(thresh, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
        cv.drawContours(self.frame, contours, -1, (0,255,0), 3,hierarchy=hierarchy)
        return contours
    
    def draw_hand_skeleton(self):
        found_hands = self.hands.process(self.frame)
        if found_hands.multi_hand_landmarks:
            for hand_landmarks in found_hands.multi_hand_landmarks:
                # print(hand_landmarks)
                self.mp_drawing.draw_landmarks(self.frame,hand_landmarks,self.mphands.HAND_CONNECTIONS) #draws the hand skeleton of all hands viewable
                hand_landmarks = found_hands.multi_hand_landmarks[0] #selects the first hand
                thumbtip = hand_landmarks.landmark[4]
                indextip = hand_landmarks.landmark[8]
                middletip = hand_landmarks.landmark[12]
                ringip = hand_landmarks.landmark[16]
                pinkietip = hand_landmarks.landmark[20]
                finger_distance = self.find_distance(indextip,middletip)
                print(finger_distance)
                # print(thumbtip.x * self.w,thumbtip.y * self.h) #have to multiply the initial (x,y) coords by the window shape width and height to get proper coordinates if not the coords are on a 0-1 scale 
                
                if finger_distance < 0.1: #if two fingers are close together then can start drawing 
                    self.points.append((int(indextip.x*self.w),int(indextip.y*self.h))) #adds localtion of index finger (x,y) coords to a list to draw circles later giving the effect of a pencil


            # hand_landmarks = found_hands.multi_hand_landmarks[0]
            # thumbtip = hand_landmarks.landmark[4]
            # indextip = hand_landmarks.landmark[8]
            # x = self.find_distance(thumbtip,indextip)
            # print(thumbtip.x)
            # cv.line(self.frame,(int(thumbtip.x),int(thumbtip.y)),(int(indextip.x),int(indextip.y)),(255,255,50),3)
        return found_hands

    def loop(self):
        while True:
            self.ret,self.frame=camera.read()
            self.h,self.w,self.c = self.frame.shape
            self.gray=cv.cvtColor(self.frame,cv.COLOR_BGR2RGB)
            # self.detect(self.frame,self.gray,"face")
            # x1,y1 = 10,50
            # x2,y2 = 90,80

            # object1 = self.draw_circle(self.frame,(x1,y1),10,(0,255,255))
            # object2 = self.draw_circle(self.frame,(x2,y2),10,(255,255,0))
            # cv.line(self.frame,(x1,y1),(self.face_center_x,self.face_center_y),(255,255,50),3)
            # print(self.find_distance([x1,y1],[self.face_center_x,self.face_center_y]))
            # self.draw_contours(self.gray)
            for p in self.points:
                cv.circle(self.gray,(p[0],p[1]),10,(0,0,255),-1) #draw a circle from a list of coords (x,y)

            self.draw_hand_skeleton()
            
            try:
                cv.imshow('Gray Window',self.gray)
                cv.imshow('Window',self.frame)
                if cv.waitKey(1)==ord('q'):
                    break
            
            except:
                print('Camera error')
                break

        self.camera.release()
        cv.destroyAllWindows()



x = Game(camera)
x.loop()


###Useful links
#https://www.youtube.com/watch?v=Ye-lTW68pZc&t=1s
#https://www.youtube.com/watch?v=NZde8Xt78Iw
#https://www.youtube.com/watch?v=dZ4itBvIjVY
#https://chromewebstore.google.com/detail/download-all-images/ifipmflagepipjokmbdecpmjbibjnakm?hl=en