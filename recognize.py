import cv2
import numpy as np
import keras.utils as image
def nothing(x):
    pass

image_x, image_y = 132,132

from keras.models import load_model
classifier = load_model('model10.h5')
names=['Call me','Hi','I Love You','No','OK','Peace','Pinky','Thumbs Down','Thumbs up','Yes']
def predictor():
       import numpy as np
       test_image = image.load_img('1.png', target_size=(224,224))
       test_image = image.img_to_array(test_image)
       test_image = np.expand_dims(test_image, axis = 0)
       result = classifier.predict(test_image)
       return names[np.argmax(result[0])]
       
cam = cv2.VideoCapture(0)
cv2.namedWindow("test")
img_counter = 0
img_text = ''
while True:
    ret, frame = cam.read()
    frame = cv2.flip(frame,1)
    img = cv2.rectangle(frame, (425,100),(625,300), (0,255,0), thickness=2, lineType=8, shift=0)
    imcrop = img[102:298, 427:623]
    cv2.putText(frame, img_text, (30, 400), cv2.FONT_HERSHEY_TRIPLEX, 1.5, (0, 255, 0))
    cv2.imshow("test", frame)
    #if cv2.waitKey(1) == ord('c'): 
    img_name = "1.png"
    save_img = cv2.resize(imcrop, (image_x, image_y))
    cv2.imwrite(img_name, save_img)
    print("{} written!".format(img_name))
    img_text = predictor()  
    if cv2.waitKey(1) == 27:
        break
cam.release()
cv2.destroyAllWindows()