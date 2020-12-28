import cv2
import numpy as np
from matplotlib import pyplot as plt 
from skimage.morphology import skeletonize

def function():
    image = cv2.imread('./demo.jpeg') 
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
    #print(ret)
    cv2.imshow('th_img', th_img) 
    cv2.waitKey(0)  

    # Noise reduction
    clean = cv2.fastNlMeansDenoising(th_img)
    print(clean)
    cv2.imshow('clean image', clean) 
    cv2.waitKey(0) 

    ####Line Segmentation
    histogram = line_segmentation(clean)
    index_line = indexing_line(clean,histogram)
    bound = addLineBoundingBox(clean,index_line)
    cv2.imshow('Bounding boxes', bound) 
    cv2.waitKey(0) 

    ### Cropping Lines
    size= clean.shape
    image = list(clean)
    final_lines=[]
    start=0
    for i in range(len(index_line)):
        temp=[]
        temp=image[start:index_line[i][0]]
        final_lines.append(np.uint8(temp))
        start= index_line[i][0]+1

    ### Word Segmentation
    histogram = word_segmentation(final_lines[0])
    index_word = indexing_word(histogram)
    bound = addWordBoundingBox(final_lines[0],index_word)
    cv2.imshow('Bounding boxes', bound) 
    cv2.waitKey(0) 

    print(final_lines[0])
    print(index_word)
    ###Character segmentation
    final_words = []
    for i in range(len(index_word)-1):
        start = index_word[i][1]
        end = index_word[i+1][0]
        final_words.append(np.uint8([x[start:end] for x in final_lines[0]]))
    
    character_segmentation(final_words[0])

# ploting histogram  
def addLineBoundingBox(image,index):
    index = np.array(index)
    index = index.flatten()
    for i in index:
        for j in range(len(image[i])):
            image[i][j] = 255
    
    return image

def addWordBoundingBox(image,index):
    index = np.array(index)
    index = index.flatten()
    new_image = image.copy()
    for j in index:
        for i in range(len(image)):
            new_image[i][j] = 255    
    return new_image


def plot_histogram(histogram,orientation):
    # this is for plotting purpose
    index = np.arange(len(histogram))
    if orientation == "horizontal":
        plt.barh(index, histogram)
    else:
        plt.bar(index,histogram)

    plt.xlabel('count', fontsize=5)
    plt.ylabel('row_count', fontsize=5)
    plt.title('Image Histogram')
    plt.show()
    
# segmenting lines
def line_segmentation(clean):
    size= clean.shape
    #print(size)
    histogram=[]
    for i in range(size[0]):
        count=0
        for j in range(size[1]):
            if(clean[i][j]!=0):
                count+=1
        histogram.append(count)

    plot_histogram(histogram,orientation="horizontal") 
    return histogram

def word_segmentation(image):
    
    cv2.imshow('Line',image)
    cv2.waitKey(0)
    
    histogram=[]
    for i in range(image.shape[1]):
        count=0
        for j in range(len(image)):
           if(image[j][i]!=0):
               count+=1
        histogram.append(count)
    plot_histogram(histogram,orientation="vertical") 
    return histogram
    
def character_segmentation(image):
    temp = np.uint8(image) / 255

    image = np.uint8(skeletonize(temp)) * 255
    cv2.imshow("Skeletonized",image)
    cv2.waitKey(0)

    indexToWhiten = []
    for j in range(image.shape[1]):
        count=0
        for i in range(image.shape[0]):
            if image[i][j] == 255:
                count+=1
        if count == 1:
            indexToWhiten.append(j)
    
    for j in indexToWhiten:
        for i in range(image.shape[0]):
            temp[i][j] = 255
    
    cv2.imshow("Character Segmentation",temp)
    cv2.waitKey(0)

def indexing_line(clean,histogram):
    print(histogram)
    size= clean.shape
    lines = 1
    temp=0
    i=0
    index=[]
    while(i<size[0]):
        if(histogram[i]==0):
            temp=i
            count=0
            while(histogram[i]==0):
                count+=1
                i+=1
            if(count>=5):
                t=[]
                t.append(temp)
                t.append(temp+count)
                index.append(t)
                lines+=1
        else:
            i+=1
    return index

def indexing_word(histogram):
    spaces = []
    i=0
    while(i<len(histogram)):
        count = 0
        if histogram[i] == 0:
            start = i
            while histogram[i] == 0:
                count+=1
                i+=1
                if i == len(histogram):
                    break

            if count >= 5:
                spaces.append([start,i-1])
        i+=1
    return spaces

function()