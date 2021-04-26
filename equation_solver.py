# Uses the trained CNN model to evaluate handwritten equations
# Input: Jpeg images of handwritten equations (handwritten part must be in black)
# Output: Answer to each equation outputted to the terminal

import glob
import cv2
from keras.models import model_from_json
import numpy as np

labelToSymbol = {12: "*", 11: "+", 10: "-", 9: "9", 8: "8", 7: "7", 6: "6", 5: "5",
				 4: "4", 3: "3", 2: "2", 1: "1", 0: "0"}

# ----- INPUT -----
equationImages = []
for imgName in glob.glob("test_images/*.jpg"):
	finalRects = []

	# Process image, same as in ftExtract.py
	img = cv2.imread(imgName, cv2.IMREAD_GRAYSCALE)
	bw_img = cv2.bitwise_not(img)
	bw_img = cv2.threshold(bw_img, 127, 255, cv2.THRESH_BINARY)[1]
	contours = sorted(cv2.findContours(bw_img, cv2.RETR_EXTERNAL, 
							cv2.CHAIN_APPROX_NONE)[0], 
							key = lambda contour: cv2.boundingRect(contour)[0])
	for contour in contours:
		x, y, w, h = cv2.boundingRect(contour)
		croppedImg = bw_img[y:y + h, x:x + w]
		resizedImg = cv2.resize(croppedImg, (28, 28))
		finalRects.append(resizedImg)

	equationImages.append(finalRects)


# ----- PREDICT -----
# Import model
file = open("model.json", "r").read()
model = model_from_json(file)
model.load_weights("model_weights.h5")

# Use the model to predict the symbols in each sequential image in equationImages
equations = []
for curr in equationImages:
	toSolve = ""
	for img in curr:
		label = np.argmax(model.predict(img.reshape(1, 1, 28, 28)))
		toSolve += labelToSymbol[label]
	equations.append(toSolve)


# ----- SOLVE AND OUTPUT -----
for line in equations:
	try:
		print(line + " = " + str(eval(line)))
	except (ValueError, SyntaxError):
		print("ERROR: Invalid Expression")
