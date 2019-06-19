import numpy as np
import cv2
from keras.preprocessing import image
import imutils
from keras.preprocessing.image import img_to_array
from keras.models import load_model
#-----------------------------
#opencv initialization
detection_model_path = 'haarcascade_frontalface_default.xml'
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
face_detection = cv2.CascadeClassifier(detection_model_path)
cap = cv2.VideoCapture(0)
#-----------------------------
#face expression recognizer initialization==================khởi tạo nhận dạng biểu hiện khuôn mặt
from keras.models import model_from_json
model = model_from_json(open("facial_expression_model_structure.json", "r").read())
model.load_weights('emotion.h5') #load weights
# model.load_weights('C:/Users/hung phung/Desktop/doan/tensorflow-101-master/model/facial_expression_model_weights.h5') #load weig
# #-----------------------------

emotions = ('angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral')

while(True):
    ret, img = cap.read()
    # img = cv2.imread('disgust.jpg')
    #ret = cv2.imread('fear.jpg')
    img = imutils.resize(img,width=500)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    faces = face_detection.detectMultiScale(gray,scaleFactor=1.1,minNeighbors=5,minSize=(100,100),flags=cv2.CASCADE_SCALE_IMAGE)

    #print(faces) #locations of detected faces=======================================vị trí của các khuôn mặt được phát hiện

    for (x,y,w,h) in faces:
        cv2.rectangle(img,(x,y),(x+w,y+h),(0,3,300),2) #draw rectangle to main image==============vẽ hình chữ nhật vào hình ảnh chính
        pts = np.array([(x-1,y),(x+w+1,y),(x+w+1,y-22),(x-1,y-22)])
        vertices = np.array([pts], dtype=np.int32)
        mask = cv2.fillPoly(img, vertices, color=(0,3,300))
        detected_face = img[int(y):int(y+h), int(x):int(x+w)] #crop detected face=================crop phát hiện khuôn mặt
        detected_face = cv2.cvtColor(detected_face, cv2.COLOR_BGR2GRAY) #transform to gray scale==============chuyển đổi sang thang độ xám
        detected_face = cv2.resize(detected_face, (48, 48)) #resize to 48x48
        
        img_pixels = image.img_to_array(detected_face)
        img_pixels = np.expand_dims(img_pixels, axis = 0)
        
        img_pixels /= 255 #pixels are in scale of [0, 255]. normalize all pixels in scale of [0, 1]======#pixels có tỷ lệ [0, 255]. chuẩn hóa tất cả các pixel theo tỷ lệ [0, 1]
        
        predictions = model.predict(img_pixels) #store probabilities of 7 expressions======lưu trữ xác suất của 7 biểu thức
        
        #find max indexed array 0: angry, 1:disgust, 2:fear, 3:happy, 4:sad, 5:surprise, 6:neutral=======tìm mảng được lập chỉ mục tối đa
        max_index = np.argmax(predictions[0])
        
        emotion = emotions[max_index]
        
        #write emotion text above rectangle===========viết văn bản cảm xúc trên hình chữ nhật
        cv2.putText(img, emotion, (int(x), int(y)-5), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
        
        #process on detected face end=============quá trình trên khuôn mặt được phát hiện
        #-------------------------

    cv2.imshow('img',img)

    if cv2.waitKey(1) & 0xFF == ord('q'): #press q to quit
        break

#kill open cv things        
cap.release()
cv2.destroyAllWindows()