import cv2
import time
import numpy as np
import os


def nothing(x):
    pass


image_x, image_y = 64, 64

def create_folder(folder_name):
    if not os.path.exists('./mydata/training_set2/' + folder_name):
        os.mkdir('./mydata/training_set2/' + folder_name)
    if not os.path.exists('./mydata/test_set2/' + folder_name):
        os.mkdir('./mydata/test_set2/' + folder_name)
    
        

        
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


            img = cv2.rectangle(frame, (195, 255), (465, 405), (0, 0, 0), thickness=2, lineType=8, shift=0)

            imcrop = img[257:403, 197:463]
            hsv = cv2.cvtColor(imcrop, cv2.COLOR_BGR2HSV)
            #mask = cv2.inRange(hsv, lower_blue, upper_blue)
            lower_green = np.array([34, 177, 76]) 
            upper_green = np.array([255, 255, 255])
            masking = cv2.inRange(hsv, lower_green, upper_green)
            result = cv2.bitwise_and(imcrop, imcrop, mask=masking)

            cv2.putText(frame, str(img_counter), (30, 400), cv2.FONT_HERSHEY_TRIPLEX, 1.5, (127, 127, 255))
            cv2.imshow("test", frame)
            cv2.imshow("mask", masking)
            cv2.imshow("result", result)

            if cv2.waitKey(1) == ord('c'):

                if t_counter <= 350:
                    img_name = "./mydata/training_set2/" + str(ges_name) + "/{}.png".format(training_set_image_name)
                    save_img = cv2.resize(masking, (image_x, image_y))
                    cv2.imwrite(img_name, save_img)
                    print("{} written!".format(img_name))
                    training_set_image_name += 1


                if t_counter > 350 and t_counter <= 400:
                    img_name = "./mydata/test_set2/" + str(ges_name) + "/{}.png".format(test_set_image_name)
                    save_img = cv2.resize(masking, (image_x, image_y))
                    cv2.imwrite(img_name, save_img)
                    print("{} written!".format(img_name))
                    test_set_image_name += 1
                    if test_set_image_name > 250:
                        break


                t_counter += 1
                if t_counter == 401:
                    t_counter = 1
                img_counter += 1


            elif cv2.waitKey(1) == 27:
                break

        if test_set_image_name > 250:
            break


    cam.release()
    cv2.destroyAllWindows()
    
ges_name = input("Enter gesture name: ")
capture_images(ges_name)
