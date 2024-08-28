import cv2
import numpy as np


image = cv2.imread('kivi.jpg')


gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Gaussian bulanıklığı uygulayarak gürültüyü azaltma işlemi 
blurred = cv2.GaussianBlur(gray, (5, 5), 0)

# Kenar tespiti için
edges = cv2.Canny(blurred, 50, 150)

# Kenarları genişletmek için morfolojik genişletme 
kernel = np.ones((3, 3), np.uint8)
edges_dilated = cv2.dilate(edges, kernel, iterations=2)

#  sadece kivileri içeren büyük konturları seç
contours, _ = cv2.findContours(edges_dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
mask = np.zeros_like(gray)

for contour in contours:
    if cv2.contourArea(contour) > 2500:  # Yeterince büyük konturları seç
        cv2.drawContours(mask, [contour], -1, (255), thickness=cv2.FILLED)

# beyaz arka plan oluştur
white_background = np.ones_like(image) * 255

# maskeyi uygula
result = cv2.bitwise_and(white_background, white_background, mask=mask)

# arka planı siyah yap
result[mask == 0] = 0

cv2.imshow('orjinal',image)
cv2.imshow('Kiviler Beyaz, Arka Plan Siyah', result)
cv2.waitKey(0)
cv2.destroyAllWindows()


cv2.imwrite('kivi_sonuc.jpg', result)
