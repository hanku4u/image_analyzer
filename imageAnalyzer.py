
import cv2
import numpy as np

# Load the image
img = cv2.imread('chip3.png')

# Convert to grayscale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Define the pattern you are looking for and create a template:
pattern = cv2.imread('patternSample.png',0)

# identify height and width of pattern
h, w = pattern.shape[::]

# Use the matchTemplate() function to find the pattern in the image:
res = cv2.matchTemplate(gray,pattern, cv2.TM_SQDIFF)

# Set a threshold value to determine whether the pattern is present in the image or not:
# threshold = 0.8
# loc = np.where(res >= threshold)

min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)

# Draw a rectangle around the pattern in the image:
top_left = min_loc
bottom_right = (top_left[0] + w, top_left[1] + h)
cv2.rectangle(img,top_left, bottom_right, 255, 2)

# Display the resulting image:
cv2.imshow('Detected',img)
cv2.waitKey()
cv2.destroyAllWindows()
