import cv2
import mediapipe as mp

mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

cap = cv2.VideoCapture(0)

with mp_hands.Hands(max_num_hands = 1, min_detection_confidence =0.5,
                    min_tracking_confidence = 0.5) as hands:

    while cap.isOpened():
        success, image = cap.read()
        if not success:
            continue

        image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)

        result = hands.process(image)

        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        gesture_text = 'Cant found hand'

        if result.multi_hand_landmarks:
            hand_landmarks = result.multi_hand_landmarks[0]

            finger_1 = False
            finger_2 = False
            finger_3 = False
            finger_4 = False
            finger_5 = False

            if(hand_landmarks.landmark[4].y < hand_landmarks.landmark[2].y):
                finger_1 = True

            if(hand_landmarks.landmark[8].y < hand_landmarks.landmark[6].y):
                finger_2 = True

            if(hand_landmarks.landmark[12].y < hand_landmarks.landmark[10].y):
                finger_3 = True

            if(hand_landmarks.landmark[16].y < hand_landmarks.landmark[14].y):
                finger_4 = True

            if(hand_landmarks.landmark[20].y < hand_landmarks.landmark[18].y):
                finger_5 = True

            if( finger_1 and finger_2 and finger_3 and finger_4 and finger_5):
                gesture_text = 'Bo'

            elif( finger_1 and finger_2):
                gesture_text = 'Gawi'

            elif( (not finger_2) and (not finger_3) and (not finger_4)
                and (not finger_5)):
                gesture_text = 'Bawi'
            else:
                gesture_text = 'Mo Ru Get Saw Yo'

            mp_drawing.draw_landmarks(
                image, hand_landmarks, mp_hands.HAND_CONNECTIONS)

        cv2.putText( image, text='Hand shape : {}'.format(gesture_text)
                     , org=(10,30), fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                 fontScale=1,color=(0,0,255), thickness=2)

        cv2.imshow('image', image)
        
        if cv2.waitKey(1) == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()

        
            
