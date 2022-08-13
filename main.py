
import cv2
import mediapipe as mp
import numpy as np
import glob
import json
class HandDetector:

    def __init__(self, mode=False, maxHands=2, detectionCon=0.5, minTrackCon=0.5):

        self.mode = mode
        self.maxHands = maxHands
        self.detectionCon = detectionCon
        self.minTrackCon = minTrackCon

        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(static_image_mode=self.mode, max_num_hands=self.maxHands,
                                        min_detection_confidence=self.detectionCon,
                                        min_tracking_confidence=self.minTrackCon)
        self.mpDraw = mp.solutions.drawing_utils
        self.fingers = []
        self.lmList = []



    def findHands(self, img, draw=True,flipType = True):


        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        self.results = self.hands.process(imgRGB)

        allHands = []
        h, w, c = img.shape
        if self.results.multi_hand_landmarks:
            for handType, handLms in zip(self.results.multi_handedness, self.results.multi_hand_landmarks):
                myHand = {}
                ## lmList
                mylmList = []

                for id, lm in enumerate(handLms.landmark):
                    px, py, pz = int(lm.x * w), int(lm.y * h), int(lm.z * w)
                    mylmList.append([px, py])

                myHand["lmList"] = mylmList

                if flipType:
                    if handType.classification[0].label == "Right":
                        myHand["type"] = "Right"
                    else:
                        myHand["type"] = "Left"
                else:
                    myHand["type"] = handType.classification[0].label
                allHands.append(myHand)
                if draw:
                    self.mpDraw.draw_landmarks(imgRGB, handLms,
                                               self.mpHands.HAND_CONNECTIONS)

        if draw:
            return allHands, imgRGB
        else:
            return allHands

def shuffle_in_unison(a, b):
    assert len(a) == len(b)
    shuffled_a = list()
    shuffled_b = list()
    permutation = np.random.permutation(len(a))
    for old_index, new_index in enumerate(permutation):
        shuffled_a.append(a[old_index])
        shuffled_b.append(b[old_index])

    return np.array(shuffled_a), np.array(shuffled_b)
def main():
    #our dataset
    image_list = glob.glob("D:/Data_for_sign_language/Static/Alphabet/*/*.jpg")
    detector = HandDetector(detectionCon=0.8, maxHands=2)
    print(image_list)
    X = []
    y_classes = ["a","b","c","ch","e","f","g","i","ie","l","m",
                 "n","o","p","r","sz","t","th","u","v","x",
                 "y","ya"]
    y = []
    counter = 0
    keys = -1
    for i in image_list:
        # Get image frame

        image = cv2.imread(i)
        image = cv2.resize(image,(100,100))

        hands, img = detector.findHands(image)

        print(hands)
        if hands:
            # Hand 1
            hand1 = hands[0]
            print(hand1)

            lmList1 = hand1["lmList"]
            #count of every class
            if counter % 204 == 0:
                keys+=1
            counter+=1
            print(keys,lmList1)
            X.append(lmList1)
            y.append(y_classes[keys])
        else:
            counter+=1
    X,y = shuffle_in_unison(X,y)
    print(X)
    print(y)

    np.savez("C:/Users/sahar/PycharmProjects/SignLanguage_AlphabetIteration/X.npz", np.array(X))
    np.savez("C:/Users/sahar/PycharmProjects/SignLanguage_AlphabetIteration/y.npz", np.array(y))


if __name__ == "__main__":
    main()