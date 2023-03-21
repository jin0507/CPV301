import cv2
import numpy as np

# Load the image
img1 = cv2.imread('tuan.jpg')
img2 = cv2.imread('7.jpg')
# Convert the image from BGR to YCrCb color space
img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2YCR_CB)
img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2YCR_CB)

#Initiate ORB 
orb = cv2.ORB_create(50)

kp1, des1 = orb.detectAndCompute(img1, None)
kp2, des2 = orb.detectAndCompute(img2, None)

matcher = cv2.DescriptorMatcher_create(cv2.DESCRIPTOR_MATCHER_BRUTEFORECE_HAMMING)

#Math descriptors

matches = matcher.match(des1, des2, None)

matches = sorted(matches, key = lambda x:x)

points1 = np.zeros((len(matches), 2), dtype=np.float32)
points2 = np.zeros((len(matches), 2), dtype=np.float32)

for i, match in enumerate(matches):
    points1[i] = kp1[match.queryIdx].pt

h, mask = cv2.findHomography(points1, points2, cv2.RANSAC)

#Use homography

height, width, channels = img2.shape

im1Reg = cv2.warPerspective(img1, h, (width, height))

img3 = cv2.darwMatches(img1, kp1, img2, kp2, matches[:10],None)
cv2.imshow("Keypoint matches",)
cv2.imshow("Registered image", im1Reg)
cv2.waitKey(0)