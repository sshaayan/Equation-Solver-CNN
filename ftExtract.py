# Goes through each image in the dataset and extracts its features
# Input: Each image file in the dataset
# Output: CSV file for each image

import cv2
import glob
import numpy as np
import pandas as pd

folders = ["dataset/-", "dataset/+", "dataset/0", "dataset/1", "dataset/2",
			"dataset/3", "dataset/4", "dataset/5", "dataset/6", "dataset/7",
			"dataset/8", "dataset/9", "dataset/times"]
nameToLabel = {"t": 12, "+": 11, "-": 10, "9": 9, "8": 8, "7": 7, "6": 6, "5": 5,
				"4": 4, "3": 3, "2": 2, "1": 1, "0": 0}

fullData = []
for file in folders:
	currData = []
	for imgName in glob.glob(file + "/*.jpg"):
		img = cv2.imread(imgName, cv2.IMREAD_GRAYSCALE) # Get grayscale image

		# Convert to binary and then invert image
		bw_img = cv2.bitwise_not(img)
		bw_img = cv2.threshold(bw_img, 127, 255, cv2.THRESH_BINARY)[1]
		
		# Find largest contour
		contours = cv2.findContours(bw_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)[0]
		maxVals = [0, 0, 0, 0]
		for contour in contours:
			x, y, w, h = cv2.boundingRect(contour)
			if w * h > maxVals[2] * maxVals[3]:
				maxVals[0] = x
				maxVals[1] = y
				maxVals[2] = w
				maxVals[3] = h

		# Crop the image based on the contour, resize, 
		# and then reshape into a column with 784 features
		croppedImg = bw_img[maxVals[1]:maxVals[1] + maxVals[3], 
						 maxVals[0]:maxVals[0] + maxVals[2]]
		resizedImg = cv2.resize(croppedImg, (28, 28))
		reshapedImg = np.reshape(resizedImg, (784, 1))

		# Add final image to currData with the corresponding label column
		currData.append(np.append(reshapedImg, [nameToLabel[imgName[8]]]))

	# Append currData to fullData
	if len(fullData) == 0:
		fullData = currData
	else:
		fullData = np.concatenate((fullData, currData))

# Export the refined dataset to a CSV file
df = pd.DataFrame(fullData, index=None)
df.to_csv("extractedData.csv", index=False)
