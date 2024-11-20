class AUVController:
    """Основной класс управления AUV."""
    def __init__(self, draw: bool = False):
        self.auv = mur.mur_init()
        self.draw = draw
        self.image_processors = ImageProcessor()
        self.lower_bound_white = np.array([0, 0, 190])
        self.upper_bound_white = np.array([360, 20, 255])
        self.lower_bound_orange = np.array([10, 100, 100])
        self.upper_bound_orange = np.array([35, 255, 255])
        self.depth_controller = DepthController(self.auv, kp=1.7, ki=0.4, kd=1.0, target_depth=3.4)

        self.motor_controller = MotorController(
            auv=self.auv,
            image_shape=self.auv.get_image_front().shape[:2]
        )

    def find_orange_arrow(self, image: np.ndarray) -> bool:
        """Обрабатывает кадр для поиска оранжевой стрелки."""
        contours = self.image_processors.get_contours(image, self.lower_bound_orange, self.upper_bound_orange)

        if contours:
            extreme_points = self.image_processors.find_extreme_points(contours)
            if extreme_points:
                pt1, pt2 = extreme_points
                area = self.image_processors.calculate_area(pt1, pt2)
                print(f"Arrow area: {area}")
                return area
        return 0

    def go_to_rectangle(self, image: np.ndarray) -> int:
        """Следует к белому прямоугольнику."""
        contours = self.image_processors.get_contours(image, self.lower_bound_white, self.upper_bound_white)

        if contours:
            extreme_points = self.image_processors.find_extreme_points(contours)
            if extreme_points:
                pt1, pt2 = extreme_points
                area = self.image_processors.calculate_area(pt1, pt2)
                self.motor_controller.adjust_power(pt1, pt2)

                if self.draw:
                    cv2.rectangle(image, pt1, pt2, (0, 255, 255), 5)
                return area
        return 0

    def run(self):
        """Запускает основной цикл обработки."""
        area = 0
        while abs(3.4 -self.auv.get_depth()) > 0.2:
            print('---')
            current_depth = self.auv.get_depth()
            power = self.depth_controller.control_depth(current_depth)
            self.motor_controller.swim_up(-power)
        self.motor_controller.swim_up(0)
        while abs(53000 - area) > 2000:
            image = self.auv.get_image_bottom()
            area = self.find_orange_arrow(image)
            if area == 0:
                self.auv.set_motor_power(0, 20)
                self.auv.set_motor_power(1, 20)
            else:
                self.auv.set_motor_power(0, 0)
                self.auv.set_motor_power(1, 0)
                self.motor_controller.swim_up()

            if self.draw:
                cv2.imshow('Processed Image', image)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

        self.motor_controller.stop()
        area = 0
        while area < 72000:
            image = self.auv.get_image_front()
            area = self.go_to_rectangle(image)

            if self.draw:
                cv2.imshow('Processed Image', image)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break



# --- Логика навигации ---
class AUVNavigator:


# --- Основной класс ---
class AUVController:
    """Основной класс управления AUV."""
    def __init__(self, draw: bool = False):
        self.auv = mur.mur_init()
        self.config = Config()
        self.image_processor = HSVImageProcessor()
        self.depth_controller = DepthController(self.auv, kp=1.7, ki=0.4, kd=1.0, target_depth=self.config.TARGET_DEPTH)
        self.motor_controller = MotorController(self.auv, self.auv.get_image_front().shape[:2])
        self.navigator = AUVNavigator(self.motor_controller, self.depth_controller, self.config)
        self.draw = draw

    def run(self):
        self.navigator.maintain_depth()
        while True:
            image = self.auv.get_image_bottom()
            area = self.navigator.search_area(self.image_processor, image)
            if area > self.config.AREA_THRESHOLD:
                break
            if self.draw:
                cv2.imshow("Processed Image", image)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

# --- Запуск ---
if __name__ == "__main__":
    app = AUVController(draw=True)
    app.run()
