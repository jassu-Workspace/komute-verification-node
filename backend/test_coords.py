import cv2
import numpy as np

img = cv2.imread('template.png')
# Erase Name (DOE, JOHN)
cv2.rectangle(img, (215, 80), (350, 130), (225, 235, 215), -1)
cv2.putText(img, "SMITH", (215, 105), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
cv2.putText(img, "JANE", (215, 125), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

# Erase DL Number
cv2.rectangle(img, (275, 160), (520, 190), (235, 245, 225), -1)
cv2.putText(img, "A1234-56789-00000", (275, 185), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

# Erase DOB
cv2.rectangle(img, (85, 315), (200, 345), (220, 205, 185), -1)
cv2.putText(img, "1990/01/01", (85, 335), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)

# Erase EXP
cv2.rectangle(img, (470, 188), (560, 215), (240, 250, 230), -1)
cv2.putText(img, "2025/01/01", (470, 205), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

cv2.imwrite('template_test.jpg', img)
