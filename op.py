import cv2
import numpy as np
import dlib
import random,math
import time

def euclidian_distance(p1,p2):
    x1 = p1[0]
    y1 = p1[1]
    x2 = p2[0]
    y2 = p2[1]
    return ((((x2 - x1 )**2) + ((y2-y1)**2) )**0.5)

def is_near(p1, p2,min_dist):
    result = euclidian_distance(p1,p2)
    return (result < min_dist)

cap = cv2.VideoCapture(0)

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

'''
x 80 from box
y 90 from box line
'''

generateRandomPadding = True
(game_point_x, game_point_y) = (-50,-50)
game_over = False
start = time.time()

color = (0,0,255)
text = 'Place your nose on the dot'
while True:
    _, frame = cap.read()
    gray = cv2.cvtColor(frame,
    cv2.COLOR_BGR2GRAY)

    (old_game_point_x,old_game_point_y) = (game_point_x,game_point_y)
    cv2.putText(frame,  
                text,  
                (50, 50),  
                cv2.FONT_HERSHEY_SIMPLEX , 1,  
                color,  
                2,  
                cv2.LINE_4) 
    faces = detector(gray)
    for face in faces:
        x1 = face.left()
        y1 = face.top()
        x2 = face.right()
        y2 = face.bottom()
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        print(f"{x1},{y1},{x2},{y2}")
        
        landmarks = predictor(gray, face)

        # for n in range(0, 68):
        #     x = landmarks.part(n).x
        #     y = landmarks.part(n).y
        #     cv2.circle(frame, (x, y), 2, (255, 0, 0), -1)

        x = landmarks.part(30).x # noseTip-x
        y = landmarks.part(30).y # noseTip-y
        cv2.circle(frame, (x, y), 2, (255, 0, 0), -1)

        nose_point = (x,y)
        box_ref_point = (x1,y)
        dist = euclidian_distance(box_ref_point,nose_point) #distance between box and nose
        perc = 0.70
        
        x_start = int(perc * dist)
        x_end = int(x_start + (2 * (1-perc)))


        # xx =int((1-perc) * dist)
        # cv2.rectangle(frame, (x-xx, y-15), (x+xx, y-5), (0, 255, 0), 2)


        if(generateRandomPadding):
            x_padding = random.randint(x_start,x_end)
            y_padding = random.randint(5,15)
            generateRandomPadding = False
        
        game_point_x = box_ref_point[0] + x_padding
        game_point_y = box_ref_point[1] - y_padding
        
        # GAME POINT Stability
        stability_value = abs( (((x1-x2) * (y1-y2)) * 15 )/((443 - 294) * (394 - 244)) )
        if(stability_value > 25):
            stability_value = 25
        print(f"sv --> {stability_value}")
        # stability_value = 0
        if(is_near((old_game_point_x,old_game_point_y),(game_point_x,game_point_y),stability_value)):
            game_point_x = old_game_point_x
            game_point_y = old_game_point_y



        # cv2.circle(frame, (x , y), 2, (200, 200, 0), -1)
        cv2.circle(frame, (game_point_x, game_point_y), 6, (200, 200, 0), -1)
        p1 = (x,y)
        p2 = (game_point_x, game_point_y)
        
        if(time.time() - start > 3):
            # print(f"[p1:={p1} || p2:={p2}]")
            if(is_near(p1,p2,5)):
                game_over = True
                break

    if(game_over):
        # break
        color = (0,255,0)
        game_over = False
        text = "Good Job"
    else:
        color = (0,0,255)
        text = 'Place your nose on the dot'

    cv2.imshow("Frame", frame)

    key = cv2.waitKey(1)
    if key == 13:
        break
 