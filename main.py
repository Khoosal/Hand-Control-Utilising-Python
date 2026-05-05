import cv2

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
smile_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_smile.xml")
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_eye.xml")

def dectect_features(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5) # image, scale, min num of neightbours (lower the number, less quality the image needs to be to get detected and more false positive, higher number means harder to detect)
    for (x, y, w, h) in faces:

        frame = cv2.rectangle(frame, (x, y), (x+w, y+h) , color=(0, 255, 0), thickness=5)           # green rectangle for face
        face = frame[y : y+h, x : x+w]
        gray_face = gray[y : y+h, x : x+w]
        
        smiles = smile_cascade.detectMultiScale(gray_face, 2.5 , minNeighbors=9)
        for (xp, yp, wp, hp) in smiles:
            face = cv2.rectangle(face, (xp, yp), (xp+wp, yp+hp) , color=(0,0,255), thickness=5)     # red rectangle for smile

        eyes = eye_cascade.detectMultiScale(gray_face, 2.5 , minNeighbors=7)
        for (xp, yp, wp, hp) in eyes:
            face = cv2.rectangle(face, (xp, yp), (xp+wp, yp+hp) , color=(255,0,0), thickness=5)     # blue rectangle for eye


    return frame

stream = cv2.VideoCapture(0)  # 0 -> webcame device

if not stream.isOpened():
    print("Webcam not found :/")
    exit()

fps = stream.get(cv2.CAP_PROP_FPS)                                      # find fps of camera
width = int(stream.get(3))
height = int(stream.get(4))
output = cv2.VideoWriter("test_stream.mp4",                             # set name of video
                         cv2.VideoWriter_fourcc('m', 'p', '4', 'v'),    # save as an mp4 (given from fourcc codes)
                         fps=fps, frameSize=(width, height))

while(True):
    ret, frame = stream.read()                                          # return, frame
    frame = cv2.flip(frame, 1)                                          # 0 for flip on y axis, 1 for x axis, -1 for both axis
    if not ret:
        print("No more stream :/")
        break

    frame = dectect_features(frame)
    #output.write(frame)                                                # create and save an MP4 file
    cv2.imshow("Webcam", frame)                                         # show frame
    if cv2.waitKey(1) == ord('q'):                                      # press 'q' to exit
        break

stream.release()
cv2.destroyAllWindows()