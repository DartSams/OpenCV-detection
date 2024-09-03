import cv2 as cv
import numpy as np
import random
import pygame
import mediapipe as mp #opensource machine learning library of detected datasets

camera = cv.VideoCapture(0) #gets the device default camera

class Game:
    def __init__(self,camera,haarcascade_path=""):
        self.camera = camera
        self.haarcascade_path = haarcascade_path
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
                cv.circle(frame,(self.face_center_x,self.face_center_y),10,(0,0,255),-1) #draw a circle at the face center point


    def draw_circle(self,frame,pos,thickness,color):
        return cv.circle(frame,pos,thickness,color,-1)
    
    def find_distance(self,obj1_pos,obj2_pos):
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
                # print(finger_distance)
                # print(thumbtip.x * self.w,thumbtip.y * self.h) #have to multiply the initial (x,y) coords by the window shape width and height to get proper coordinates if not the coords are on a 0-1 scale 
                
                pos = (int(indextip.x * self.w), int(indextip.y * self.h))

                if finger_distance > 0.19 and self.drawing_mode: #if only holding up 1 finger while the others are down 
                    self.points.append(pos) #adds localtion of index finger (x,y) coords to a list to draw circles later giving the effect of a pencil

                if finger_distance < 0.1: #logic to erase drawing points
                    cv.circle(self.frame2,pos,self.eraser_size,(255,255,255),1) #displays a visual effect to show user is erasing points
                    self.points = [p for p in self.points if self.find_distance(p, pos) > self.eraser_size] #only keeps the pencil drawings from the list if they are greater than the eraser size



        return found_hands

    def loop(self):
        while True:
            self.ret,self.frame=camera.read()
            self.h,self.w,self.c = self.frame.shape
            self.gray=cv.cvtColor(self.frame,cv.COLOR_BGR2RGB)
            for p in self.points:
                cv.circle(self.gray,(p[0],p[1]),10,(0,0,255),-1) #draw a circle at the face center point
            self.detect(self.frame,self.gray,"face")
            self.draw_hand_skeleton()
            
            try:
                #flip the frame (easier to visually understand)
                self.frame = cv.flip(self.frame,1)
                self.frame2 = cv.flip(self.frame2,1)


                cv.imshow('Gray Window',self.frame2)
                cv.imshow('Window',self.frame)
                if cv.waitKey(1)==ord('q'):
                    break

                if cv.waitKey(1)==ord('1'):
                    print("clearing all")
                    self.points.clear()

                if cv.waitKey(1)==ord('2'): 
                    print("you can now draw")
                    self.drawing_mode = True

                if cv.waitKey(1)==ord('3'): 
                    print("disabling drawing mode")
                    self.drawing_mode = False
            
            except:
                print('Camera error')
                break

        self.camera.release()
        cv.destroyAllWindows()



x = Game(camera)
x.loop()