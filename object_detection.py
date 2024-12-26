import numpy as np
import cv2
import mediapipe as mp
import matplotlib.pyplot as plt


mp_obj=mp.solutions.objectron
mp_draw=mp.solutions.drawing_utils

img=cv2.imread(r'C:\Users\Admin.DESKTOP-NUPAOG1\DataScience\Mediapipe\images\cup1.jpeg')

img=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)

obj=mp_obj.Objectron(static_image_mode=True,
    max_num_objects=5,
    min_detection_confidence=0.2,
    model_name='Cup')

res=obj.process(img)

img=cv2.cvtColor(img,cv2.COLOR_RGB2BGR)

if not res.detected_objects:
    print("No object is found")
    

cimg=img.copy()
#landmark_spec = mp_draw.DrawingSpec(color=(0, 255, 0), thickness=1, circle_radius=1)  # Green, thin
#connection_spec = mp_draw.DrawingSpec(color=(255, 0, 0), thickness=1)  # Red, thin


for dec_obj in res.detected_objects:
    mp_draw.draw_landmarks(cimg,  
                           dec_obj.landmarks_2d,
                           mp_obj.BOX_CONNECTIONS,
                           #landmark_drawing_spec=landmark_spec,
                           #connection_drawing_spec=connection_spec
                           )

    mp_draw.draw_axis(cimg,
                      dec_obj.rotation,
                      dec_obj.translation)

#cv2.imshow('Cup',cimg)

#cv2.waitKey(0)

fig, ax = plt.subplots(figsize=(10, 10))
ax.imshow(cimg)
ax.axis('off')
plt.show()

