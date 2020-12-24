import cv2
import numpy as np

def function():
    image = cv2.imread('C:\\Users\\Vibuthi mishra\\Documents\\projects\\DIP_Project\\demo.jpeg') 
    # We use cvtColor, to convert to grayscale 
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) 
    
    cv2.imshow('Grayscale', gray_image) 
    cv2.waitKey(0)   
    
    # window shown waits for any key pressing event 
    cv2.destroyAllWindows() 
    preprocess(gray_image)

def preprocess(bgr_img):#gray image   
    img = bgr_img[:]
    cv2.imshow('1', img)
    cv2.waitKey()   

    blur = cv2.GaussianBlur(img,(5,5),0)
    cv2.imshow('blur', blur) 
    cv2.waitKey(0) 

    # thesholding and binarization 
    ret,th_img = cv2.threshold(blur,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU) #converts black to white and inverse
    print(ret)
    cv2.imshow('th_img', th_img) 
    cv2.waitKey(0)  

    # Noise reduction
    clean = cv2.fastNlMeansDenoising(th_img)
    cv2.imshow('clean image', clean) 
    cv2.waitKey(0) 

    # Skew Detection and Correction
    #Applicable only if image is not horizontal

    


function()