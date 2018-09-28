import cv2
import numpy as np
from PIL import Image

Prewitt_x = np.array([[-1, 0, 1],[-1, 0, 1],[-1, 0, 1]])
Prewitt_y = np.array([[-1, -1, -1],[0, 0, 0],[1, 1, 1]])
Prewitt_45 = np.array([[0, 1, 1],[-1, 0, 1],[-1, -1, 0]])
Prewitt_135 = np.array([[-1, -1, 0],[-1, 0, 1],[0, 1, 1]])

def task1(FILENAME):
    '''
        Task1
        Read and save image. 
        An image specified by FILENAME is read and save it as “01original.jpg” in the current directory.
    '''
    img = cv2.imread(FILENAME)
    cv2.imwrite("01original.jpg",img)

def task2():
    '''
        Task2
        Compute a binary edge image by Prewitt operator given a fixed threshold value. 
        The corresponding binary edge image is computed using function myprewittedge. 
        The binary edge image is computed with 
        threshold T = (maximum intensity value of the input image) * 0.2, 
        direction = ‘all’ and save it as “02binary1.jpg”
    '''
    image = Image.open("fig.tif")
    image_array = np.array(image)
    image_array = binary(image_array, t=0)

    image_x = imconv(image_array, Prewitt_x)
    image_y = imconv(image_array, Prewitt_y)
    image_45 = imconv(image_array, Prewitt_45)
    image_135 = imconv(image_array, Prewitt_135)

    image_xy = find_max_image(image_x, image_y, image_45, image_135)

    cv2.imwrite("02binary1.jpg", image_xy)
    #plt.imshow(image_xy, cmap=cm.gray)
    #plt.axis("off")
    #plt.show()

def imconv(image_array, Prewitt):

    image = image_array.copy()
    dim1, dim2 = image.shape
    for i in range(1, dim1 - 1):
        for j in range(1, dim2 - 1):
            image[i, j] = abs((image_array[(i - 1):(i + 2), (j - 1):(j + 2)] * Prewitt).sum())
    image = image * (255.0 / image.max())
    return image

def task3():
    '''
    Compute a binary edge image by Prewitt operator with a self-adapted threshold value. 
    This task is similar to TASK 2, 
    except the corresponding binary edge image is computed using a self-adapted threshold value. 
    The self-adapted threshold value should be computed within the 
    myprewittedge function according to the algorithm described below when the argument T takes the default value,
    i.e. T = [] (for python, T = None). 
    This binary edge image is computed with T = [] (for python, T = None), direction = ‘all’. 
    Please save the binary edge image as “03binary2.jpg”.
    '''
    image = Image.open("fig.tif").convert("L")
    image_array = np.array(image)

    t = find_threshold(image_array)
    image_array = binary(image_array,t)

    image_x = imconv(image_array, Prewitt_x)
    image_y = imconv(image_array, Prewitt_y)
    image_45 = imconv(image_array, Prewitt_45)
    image_135 = imconv(image_array, Prewitt_135)

    image_xy = find_max_image(image_x,image_y,image_45,image_135)

    cv2.imwrite("03binary2.jpg", image_xy)
    #plt.imshow(image_xy, cmap=cm.gray)
    #plt.axis("off")
    #plt.show()

def find_threshold(image_array):
    max = 0
    min = 255
    image = image_array.copy()
    dim1, dim2 = image.shape
    for i in range(1, dim1):
        for j in range(1, dim2):
            if image[i, j] > max:
                max = image[i, j]
            if image[i, j] < min:
                min = image[i, j]
    t = (int(max) + int(min))/2
    for time in range(10):
        G1 = []
        G2 = []
        for i in range(1, dim1):
            for j in range(1, dim2):
                if image[i, j] > t:
                    G1.append(image[i, j])
                else:
                    G2.append(image[i, j])
        t = 0.5*(average(G1) + average(G2))
    return t

def average(G):
    sum = 0
    for i in range(len(G)):
        sum += G[i]
    return sum / len(G)

def binary(image_array, t):
    if t == 0:
        image = image_array.copy()
        dim1, dim2 = image.shape
        max = 0
        for i in range(1, dim1):
            for j in range(1, dim2):
                if image[i, j] > max:
                    max = image[i, j]
        for i in range(1, dim1):
            for j in range(1, dim2):
                if image[i, j] > 0.5 * max:
                    image[i, j] = 255
                else:
                    image[i, j] = 0
        return image
    else:
        image = image_array.copy()
        dim1, dim2 = image.shape
        for i in range(1, dim1):
            for j in range(1, dim2):
                if image[i, j] > t:
                    image[i, j] = 255
                else:
                    image[i, j] = 0
        return image

def find_max_image(image1, image2, image3, image4):
    image = image1.copy()
    dim1, dim2 = image.shape
    for i in range(1, dim1):
        for j in range(1, dim2):
            image[i,j] = max(image1[i,j],image2[i,j],image3[i,j],image4[i,j])
    return image

def abs_image(image_array):
    image = image_array.copy()
    dim1, dim2 = image.shape
    for i in range(1, dim1):
        for j in range(1, dim2):
            image[i,j] = abs(image[i,j])
    return image

def task4():
    '''
    Find the longest line segment in the binary edge image based on Hough transform using mylineextraction(f)
    where f is the output from TASK 3 (“03binary2.jpg”). 
    Then, draw the longest line segment (as a blue line) on the input image and saved it as “04longestline.jpg”.
    For this task, hough, houghpeaks, houghlines functions in MATLAB 
    and related python functions in skimage.transform package can be used.
    '''
    img1 = cv2.imread("01original.jpg")
    img = cv2.imread("03binary2.jpg",0)
    img[0] = 0
    img[1] = 0
    img[2] = 0
    img[3] = 0
    img[4] = 0
    for col in img:
        col[0] = 0
        col[1] = 0
        col[2] = 0
        col[3] = 0
        col[4] = 0
    minLineLength = 320
    maxLineGap = 200
    lines1 = cv2.HoughLinesP(img, 1, np.pi/180,minLineLength,maxLineGap)
    point_set = []
    for line in lines1:
        x1 = line[0][0]
        y1 = line[0][1]
        x2 = line[0][2]
        y2 = line[0][3]
        if (x1 and x2) < 302 and (x1 and x2) > 291 and (y1-y2) > 5:
            point_set.append(line[0])
    x1 = point_set[0][2]
    y1 = point_set[0][3]
    x2 = point_set[1][0]
    y2 = point_set[1][1]
    cv2.circle(img1,(x1,y1),4,(0, 0, 255),2)
    cv2.circle(img1, (x2, y2), 4, (0, 0, 255), 2)
    cv2.line(img1, (x1, y1), (x2, y2), (127, 255, 0), 2)
    cv2.imwrite('04longestline.jpg',img1)
    #cv2.imshow("fig", img1)
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()

def task5():
    '''
    Image alignment using SIFT. You are given three ordinary images,
    i.e., “image1.png”, “image2.png” and “image3.png”, as well as one QR code image “QR-Code.png”.
    Your task is to find the ordinary image that matches “QR-Code.png” best among the three images,
    draw the line for each matching and save the output as “05QR_img1.png”,
    “06QR_img2.png” and “07QR_img3.png” respectively. 
    '''
    img1 = cv2.imread("image1.png")
    img2 = cv2.imread("image2.png")
    img3 = cv2.imread("image3.png")
    qr = cv2.imread("QR-Code.png")
    sift = cv2.xfeatures2d.SIFT_create()

    kp1, des1 = sift.detectAndCompute(img1, None)
    kp2, des2 = sift.detectAndCompute(img2, None)
    kp3, des3 = sift.detectAndCompute(img3, None)
    kp_qr, des_qr = sift.detectAndCompute(qr, None)
    bf = cv2.BFMatcher(cv2.NORM_L2)

    matches1 = bf.knnMatch(des_qr, des1, k=2)
    matches2 = bf.knnMatch(des_qr, des2, k=2)
    matches3 = bf.knnMatch(des_qr, des3, k=2)

    good1 = []
    good2 = []
    good3 = []

    for m, n in matches1:
        if m.distance < 0.75 * n.distance:
            good1.append(m)
    for m, n in matches2:
        if m.distance < 0.5 * n.distance:
            good2.append(m)
    for m, n in matches3:
        if m.distance < 0.75 * n.distance:
            good3.append(m)

    drawMatchesKnn_cv2(qr, kp_qr,img1, kp1, good1[:10],'05QR_img1.png')
    drawMatchesKnn_cv2(qr, kp_qr, img2, kp2, good2[:],'06QR_img2.png')
    drawMatchesKnn_cv2(qr, kp_qr, img3, kp3, good3[:10],'07QR_img3.png')

def drawMatchesKnn_cv2(img1_gray, kp1, img2_gray, kp2, goodMatch, IMGNAME):
    h1, w1 = img1_gray.shape[:2]
    h2, w2 = img2_gray.shape[:2]
    length = min(len(kp1), len(kp2))
    vis = np.zeros((max(h1, h2), w1 + w2, 3), np.uint8)
    vis[:h1, :w1] = img1_gray
    vis[:h2, w1:w1 + w2] = img2_gray

    for key_point_pair in goodMatch:
        point1_idx = key_point_pair.queryIdx
        point2_idx = key_point_pair.trainIdx
        if point1_idx < length and point2_idx < length:
            (x1, y1) = np.int32(kp1[point1_idx].pt)
            (x2, y2) = np.int32(kp2[point2_idx].pt)
            x2 = x2 + w1
            cv2.line(vis, (x1, y1), (x2, y2), (0, 0, 255))
            cv2.putText(vis, str(point1_idx), (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 0))
            cv2.putText(vis, str(point2_idx), (x2, y2), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0))
    cv2.imwrite(IMGNAME, vis)
    #cv2.imshow("match", vis)
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()


if __name__ == "__main__":
    # task1("fig.tif")
    # task2()
    # task3()
    # task4()
    task5()
    # pass
