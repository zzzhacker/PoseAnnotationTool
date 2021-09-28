import mediapipe as mp
import cv2
import numpy as np


class PoseEstimation:
    """
    Examples
    --------
    >>> from PoseModel import PoseEstimation
    >>> import cv2
    >>> image = cv2.imread('input_image.png')
    >>> pose_estimator = PoseEstimation()
    >>> x,y = pose_estimator.inference(image)

    """
    def __init__(self):
        self.model_name = 'model_pose'
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(static_image_mode=True,
                                            max_num_hands=2,
                                            min_detection_confidence=0.7)

    def inference(self,image):
        image_hight, image_width, _ = image.shape
        results = self.hands.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        x = []
        y = []
        if results.multi_hand_landmarks != None:
            for hand_landmarks in results.multi_hand_landmarks:
                for landmark in hand_landmarks.landmark:
                    x.append(landmark.x*image_width)
                    y.append(landmark.y*image_hight)
                break
            return np.array([x,y]).T
        else:
            return np.full((21,2),-1.0)
        

pose_estimator = PoseEstimation()
