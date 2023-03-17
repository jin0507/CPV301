import cv2
import numpy as np
import matplotlib.pyplot as plt

def stitching_build(path0, path1):

    img1 = cv2.imread(path0)
    img2 = cv2.imread(path1)
    
    img1_gray = cv2.cvtColor(img1, cv2.COLOR_RGB2GRAY)
    img2_gray = cv2.cvtColor(img2, cv2.COLOR_RGB2GRAY)

    sift = cv2.SIFT_create()
    kp1, des1 = sift.detectAndCompute(img1_gray, None)
    kp2, des2 = sift.detectAndCompute(img2_gray, None)

    bf = cv2.BFMatcher()
    matches = bf.knnMatch(des1,des2, k=2) 
    ratio = 0.75

    good = []
    for m in matches:
        if m[0].distance < 0.5*m[1].distance:         
            good.append(m)
    matches = np.asarray(good)

    if len(matches[:,0]) >= 4:
        src = np.float32([ kp1[m.queryIdx].pt for m in matches[:,0] ]).reshape(-1,1,2)
        dst = np.float32([ kp2[m.trainIdx].pt for m in matches[:,0] ]).reshape(-1,1,2)

        H, masked = cv2.findHomography(src, dst, cv2.RANSAC, 5.0)
        print(H)
    else:
        raise AssertionError("Can't find enough keypoints.")  	
    
    dst = cv2.warpPerspective(img1,H,(img2.shape[1] + img1.shape[1], img2.shape[0]))     	
    plt.subplot(122),plt.imshow(dst),plt.title('Warped Image')
    plt.show()
    plt.figure()
    dst[0:img2.shape[0], 0:img2.shape[1]] = img2
    cv2.imwrite('resultant_stitched_panorama.jpg',dst)
    plt.imshow(dst)
    plt.show()
    cv2.imwrite('resultant_stitched_panorama.jpg',dst)
    
stitching_build("img0.jpg", "img1.jpg")
