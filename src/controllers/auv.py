"""
The project is aimed at creating an autonomous underwater vehicle that accurately moves along a given line. 
Using modern navigation and sensor technologies, ANPA collects data on the underwater environment, 
ensuring effective monitoring and research at minimal cost.
"""
# Import files and libraries
import time
import cv2 as cv
import numpy as np
import pymurapi as mur


# Initializing classes
class GOEM():
    # Вычисляет наклон (a) и интерсепт (b), представляющие уравнение линии y = ax + b.
    def calc_line(self, x1, y1, x2, y2) -> tuple[float | float]:
        a = float(y2 - y1) / (x2 - x1) if x2 != x1 else 0
        b = y1 - a * x1
        return a, b

    # Упорядочивает вершины прямоугольника
    def order_box(self, box):
        srt = np.argsort(box[:, 1])
        btm1 = box[srt[0]]
        btm2 = box[srt[1]]

        top1 = box[srt[2]]
        top2 = box[srt[3]]

        bc = btm1[0] < btm2[0]
        btm_l = btm1 if bc else btm2
        btm_r = btm2 if bc else btm1

        tc = top1[0] < top2[0]
        top_l = top1 if tc else top2
        top_r = top2 if tc else top1

        return np.array([top_l, top_r, btm_r, btm_l])
    
    # Вычисляет процентное смещение по горизонтали
    def get_horz_shift(self, x, w) -> float:
        hw = w / 2
        return 100 * (x - hw) / hw

    # Рассчитывает вертикальный угол между двумя точками
    def get_vert_angle(self, p1, p2, w, h) -> int:
        px1 = p1[0] - w/2
        px2 = p2[0] - w/2
        
        py1 = h - p1[1]
        py2 = h - p2[1]

        angle = 90
        if px1 != px2:
            a, b = self.calc_line(px1, py1, px2, py2)
            angle = 0
            if a != 0:
                x0 = -b/a
                y1 = 1.0
                x1 = (y1 - b) / a
                dx = x1 - x0
                tg = y1 * y1 / dx / dx
                angle = 180 * np.arctan(tg) / np.pi
                if a < 0:
                    angle = 180 - angle
        return angle

    # Вычисляет евклидово расстояние между двумя точками
    def calc_line_length(self, p1, p2) -> float:
        dx = p1[0] - p2[0]
        dy = p1[1] - p2[1]
        return (dx * dx + dy * dy) ** 0.5
    
    # Определяет вектор середины для вертикальных и горизонтальных сторон прямоугольника
    def calc_box_vector(self, box) -> tuple[tuple, tuple]:
        v_side = self.calc_line_length(box[0], box[3])
        h_side = self.calc_line_length(box[0], box[1])
        idx = [0, 1, 2, 3]
        if v_side < h_side:
            idx = [0, 3, 1, 2]
        return ((box[idx[0]][0] + box[idx[1]][0]) / 2, (box[idx[0]][1] + box[idx[1]][1]) / 2), ((box[idx[2]][0] + box[idx[3]][0]) / 2, (box[idx[2]][1]  +box[idx[3]][1]) / 2)


class ROI:

    area = 0
    vertices = None

    # Инициализирует область как многоугольник внутри кадра
    def init_roi(self, width, height) -> None:
        vertices = [(0, height), (width / 4, 3 * height / 4),(3 * width / 4, 3 * height / 4), (width, height),]
        self.vertices = np.array([vertices], np.int32)
        
        blank = np.zeros((height, width, 3), np.uint8)
        blank[:] = (255, 255, 255)
        blank_gray = cv.cvtColor(blank, cv.COLOR_BGR2GRAY)
        blank_cropped = self.crop_roi(blank_gray)
        self.area = cv.countNonZero(blank_cropped)

    # Создает маску внутри данного изображения
    def crop_roi(self, img):
        mask = np.zeros_like(img)
        match_mask_color = 255
        
        cv.fillPoly(mask, self.vertices, match_mask_color)
        
        masked_image = cv.bitwise_and(img, mask)
        
        return masked_image

    # Возвращает площадь (в пикселях) текущей ROI
    def get_area(self) -> int:
        return self.area

    # Возвращает вершины текущей ROI.
    def get_vertices(self):
        return self.vertices


class DrivingController(GOEM):
    # OCEAN — Optimized Control Engine for Auto Navigation

    # Инициализирует объекты конфигурации
    def __init__(self, config) -> None:
        self.auv = config.AUV
        # self.cap = cv.VideoCapture(0)
        self.config = config
        self.Roi = ROI()
        super().__init__()

        self.arr_right = []
        self.arr_left = []

    # Балансирует яркость и контраст изображения для оптимизации белых областей
    def __balance_pic(self, image):
        img_gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
        img_blur = cv.GaussianBlur(img_gray,(3,3),0)

        if self.config.adaptiveThreshold:
            pre_mask = cv.adaptiveThreshold(img_blur,255,cv.ADAPTIVE_THRESH_MEAN_C, cv.THRESH_BINARY,11,5)
        else:
            _, pre_mask = cv.threshold(img_gray,0,255,cv.THRESH_BINARY+cv.THRESH_OTSU)

        _, mask = cv.threshold(pre_mask,self.config.min_white,255,cv.THRESH_BINARY_INV)

        crop = self.Roi.crop_roi(mask)
        nwh = cv.countNonZero(crop)

        perc = 100 * nwh / self.Roi.get_area()

        self.config.perc = perc

        return mask


    # Подготавливает изображение, инициализируя ROI
    def __prepare_pic(self, image):
        height, width = image.shape[:2]

        if self.Roi.get_area() == 0:
            self.Roi.init_roi(width, height)

        return self.__balance_pic(image), width, height

    # Находит главный контур в данном изображении на основе наибольшей площади
    def __find_main_countour(self, image):
        cnts, _ = cv.findContours(image, cv.RETR_CCOMP, cv.CHAIN_APPROX_SIMPLE)

        C = None
        if cnts is not None and len(cnts) > 0:
            C = max(cnts, key = cv.contourArea)

        if C is None:
            return None, None

        rect = cv.minAreaRect(C)
        box = cv.boxPoints(rect)
        box = np.intp(box)

        box = self.order_box(box)

        return C, box

    # Обрабатывает изображение с камеры подводного аппарата
    def __handle_pic(self, fout = None, show = True):

        image = self.auv.get_image_bottom()
        # ret, image = self.cap.read()

        if image is None:
            raise Exception(("Video not found"))
            ...

        cropped, w, h = self.__prepare_pic(image)
        if cropped is None:
            return None, None
        cont, box = self.__find_main_countour(cropped)
        if cont is None:
            return None, None

        p1, p2 = self.calc_box_vector(box)
        p1 = list(map(int, p1))
        p2 = list(map(int, p2))

        if p1 is None: return None, None

        angle = self.get_vert_angle(p1, p2, w, h)
        shift = self.get_horz_shift(p1[0], w)

        draw = fout is not None or show

        if draw:
            cv.drawContours(image, [cont], -1, (0,0,255), 3)
            cv.drawContours(image,[box],0,(255,0,0),2)
            cv.line(image, p1, p2, (0, 255, 0), 3)
            msg_a = "Angle {0}".format(int(angle))
            msg_s = "Shift {0}".format(int(shift))

            cv.putText(image, msg_a, (10, 20), cv.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)
            cv.putText(image, msg_s, (10, 40), cv.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)

        if fout is not None:
            cv.imwrite(fout, image)

        if show:    
            cv.imshow("Image", image)
            cv.waitKey(1)

        return angle, shift

    # Выводит все настройки конфигурации
    def __display_config(self) -> None:
        for key, value in self.config.items():
            print('{} = {}'.format(key, value))

    # Получает значения угла и смещения
    def __get_vector(self):
        angle, shift = self.__handle_pic()
        return angle, shift
    
    # Пытается снова найти линию, если аппарат потерял её
    # def find_line(self, side):

    #     if side == 0:
    #         return None, None
            
    #     for _ in range(0, self.config['find_turn_attempts']):
    #         self.turn(side, self.config['find_turn_step'])
    #         angle, shift = self.get_vector()
    #         if angle is not None:
    #             return angle, shift

    #     return None, None

    # Оценивает значения угла и смещения по порогам
    def __check_shift_turn(self, angle, shift):
        turn_state = 0
        if angle < 90 - self.config.turn_angle or angle > 90 + self.config.turn_angle:
            turn_state = angle - 90

        shift_state = 0
        if abs(shift) > self.config.shift_max:
            shift_state = np.sign(-shift)

        return turn_state, shift_state
    

    def folow_line(self):

        function_speed = lambda ts, ss, z: (z * ts * self.config.rotation_coefficient) + (z * ss * self.config.shift_coefficient)
        angle, shift = self.__get_vector()
        if angle is None:
            return

        turn_state, shift_state = self.__check_shift_turn(angle, shift)
        
        # l = self.config['perc'] - self.config['last_perc']
        # print(f'{l:.2f}')
        # if abs(self.config['perc'] - self.config['last_perc']) > self.config['limit_perc']:
        #     self.config['last_perc'] = self.config['perc']
        #     return

        self.config.last_perc = self.config.perc

        speed_r = int(self.config.speed + function_speed(turn_state, shift_state, -1))
        speed_l = int(self.config.speed + function_speed(turn_state, shift_state, 1 ))

        self.arr_right.append(speed_r)
        self.arr_left.append(speed_l)

        if len(self.arr_left) == 5:
            self.auv.set_motor_power(0, sum(self.arr_right) // 5)
            self.auv.set_motor_power(1, sum(self.arr_left) // 5)

            self.arr_right = self.arr_right[1:]
            self.arr_left = self.arr_left[1:]


        print(f"Motor 1: {self.arr_right}")
        print(f"Motor 2: {self.arr_left}")

    
