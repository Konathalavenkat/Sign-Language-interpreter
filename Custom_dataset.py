import cv2
import time
import numpy as np
import os
def nothing(x):
    pass
image_x, image_y = 224,224
path="./dl_dataset"
def create_folder(folder_name):
    if not os.path.exists(path+'/train/' + folder_name):
        os.mkdir(path+'/train/' + folder_name)
    if not os.path.exists(path+'/test/' + folder_name):
        os.mkdir(path+'/test/' + folder_name)     
def capture_images(ges_name):
    create_folder(str(ges_name))
    cam = cv2.VideoCapture(0)
    cv2.namedWindow("test")
    img_counter = 0
    t_counter = 1
    training_set_image_name = 1
    test_set_image_name = 1
    listImage = [1,2,3,4,5]
    for loop in listImage:
        while True:
            ret, frame = cam.read()
            frame = cv2.flip(frame, 1)
            img = cv2.rectangle(frame, (425, 100), (625, 300), (0, 255, 0), thickness=2, lineType=8, shift=0)
            rgb = img[102:298, 427:623]
            cv2.putText(frame, str(img_counter), (30, 400), cv2.FONT_HERSHEY_TRIPLEX, 1.5, (127, 127, 255))
            cv2.imshow("frame", frame)
            if cv2.waitKey(1) == ord('c'):
                if t_counter <= 3:
                    img_name = path+"/train/" + str(ges_name) + "/{}.png".format(training_set_image_name)
                    save_img = cv2.resize(rgb, (image_x, image_y))
                    cv2.imwrite(img_name, save_img)
                    print("{} written!".format(img_name))
                    training_set_image_name += 1
                if t_counter == 4 :
                    img_name = path+"/test/" + str(ges_name) + "/{}.png".format(test_set_image_name)
                    save_img = cv2.resize(rgb, (image_x, image_y))
                    cv2.imwrite(img_name, save_img)
                    print("{} written!".format(img_name))
                    test_set_image_name += 1
                    if test_set_image_name > 250:
                        break
                t_counter += 1
                if t_counter == 5:
                    t_counter = 1
                img_counter += 1
            elif cv2.waitKey(1) == 27:
                break
        if test_set_image_name > 250:
            break
    cam.release()
    cv2.destroyAllWindows()
ges_name = input("Enter class name: ")
capture_images(ges_name)