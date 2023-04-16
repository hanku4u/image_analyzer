import cv2
import numpy as np

# Load the image:
img = cv2.imread('chip2.png')

# Convert the image to grayscale:
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Define the pattern you are looking for and extract its keypoints and descriptors:
pattern = cv2.imread('patternSample.png',0)
detector = cv2.ORB_create()
kp_pattern, des_pattern = detector.detectAndCompute(pattern, None)

# Use a feature-based matching algorithm to find the keypoints and descriptors in the image that match the pattern's keypoints and descriptors:
matcher = cv2.DescriptorMatcher_create(cv2.DESCRIPTOR_MATCHER_BRUTEFORCE_HAMMING)
kp_image, des_image = detector.detectAndCompute(gray, None)
matches = matcher.match(des_pattern, des_image)

# Sort the matches by their distance and select the best ones:
matches = sorted(matches, key=lambda x:x.distance)
good_matches = matches[:10]

# Calculate the homography matrix that maps the pattern keypoints to the image keypoints:
src_pts = np.float32([kp_pattern[m.queryIdx].pt for m in good_matches]).reshape(-1,1,2)
dst_pts = np.float32([kp_image[m.trainIdx].pt for m in good_matches]).reshape(-1,1,2)
M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC,5.0)

# Use the homography matrix to warp the pattern image to the image coordinates:
h,w = pattern.shape
pts = np.float32([[0,0],[0,h-1],[w-1,h-1],[w-1,0]]).reshape(-1,1,2)
dst = cv2.perspectiveTransform(pts,M)

# Draw a polygon around the pattern in the image:
cv2.polylines(img,[np.int32(dst)],True,(0,0,255),3, cv2.LINE_AA)

# Display the resulting image:
cv2.imshow('Detected',img)
cv2.waitKey(0)
cv2.destroyAllWindows()

