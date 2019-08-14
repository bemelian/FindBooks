import numpy as np
import cv2 as cv

# Загрузка изображения, смена цвета на оттенки серого и уменьшение резкости
image = cv.imread("example.jpg")
gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
gray = cv.GaussianBlur(gray, (3, 3), 0)
cv.imwrite("gray.jpg", gray)

# Распознавание контуров
edged = cv.Canny(gray, 10, 250)
cv.imwrite("edged.jpg", edged)

# Создание и применение закрытия
kernel = cv.getStructuringElement(cv.MORPH_RECT, (7, 7))
closed = cv.morphologyEx(edged, cv.MORPH_CLOSE, kernel)
cv.imwrite("closed.jpg", closed)

# Нахождение контуров на изображении и подсчет количества книг
cnts = cv.findContours(closed.copy(), cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)[0]
total = 0

# Цикл по контурам
for c in cnts:
    # Аппроксимация контура
    peri = cv.arcLength(c, True)
    approx = cv.approxPolyDP(c, 0.02 * peri, True)

    # Если у контура 4 вершины, предполагаем, что это книга
    if len(approx) == 4:
        cv.drawContours(image, [approx], -1, (0, 255, 0), 4)
        total += 1

# Вывод результирующего изображения
print("На этой картинке {0} книг".format(total))
cv.imwrite("output.jpg", image)