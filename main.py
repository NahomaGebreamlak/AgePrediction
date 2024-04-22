import cv2
from deepface import DeepFace

face_classifier = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

video_capture = cv2.VideoCapture(0)

def detect_bounding_box(vid):
    gray_image = cv2.cvtColor(vid, cv2.COLOR_BGR2GRAY)
    faces = face_classifier.detectMultiScale(gray_image, 1.1, 5, minSize=(40, 40))
    return faces

while True:
    result, video_frame = video_capture.read()  # read frames from the video
    if not result:
        break  # terminate the loop if the frame is not read successfully

    faces = detect_bounding_box(video_frame)  # detect faces in the frame

    for (x, y, w, h) in faces:
        # Extract the face region from the frame
        face_region = video_frame[y:y + h, x:x + w]
        cv2.rectangle(video_frame, (x, y), (x + w, y + h), (255, 0, 0), 2)  # Draw rectangle around the face
        try:
            # Predict age and race for the face region
            demography = DeepFace.analyze(face_region, actions=['age', 'race'], enforce_detection=False)
            age = demography[0]["age"]
            dominant_race = demography[0]["dominant_race"]
            # Draw age and race text on the frame
            cv2.putText(video_frame, f"Age: {age}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
            cv2.putText(video_frame, f"Race: {dominant_race}", (x, y + h + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        except Exception as e:
            print("Error:", str(e))

    cv2.imshow("My Face Detection Project", video_frame)  # display the processed frame

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

video_capture.release()
cv2.destroyAllWindows()
