import cv2
from matplotlib import pyplot as plt
import pytesseract
import timeit

image = cv2.imread("tests/images/score.png",cv2.IMREAD_GRAYSCALE)

def image_to_text(image):
    text = pytesseract.image_to_string(image)
    score = text.partition(' ')[0]
    return score

starttime = timeit.default_timer()
# print("The start time is :",starttime)
score = image_to_text(image)
print("The time difference is :", timeit.default_timer() - starttime)

plt.imshow(image,'gray')
print("Score: ", score)