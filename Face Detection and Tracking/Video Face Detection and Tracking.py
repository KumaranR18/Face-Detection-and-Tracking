import cv2

alg = "haarcascade_frontalface_default.xml"      # Load Haar Cascade for face detection
haar_cascade = cv2.CascadeClassifier(alg)         # Initialize the detector

video_path = "Upload Your Video"
cam = cv2.VideoCapture(video_path)                 # Open video file or capture device

while True:
    _, img = cam.read()                            # Read a frame from the video
    grayImg = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # Convert frame to grayscale
    face = haar_cascade.detectMultiScale(grayImg, 1.3, 4)  # Detect faces

    for (x, y, w, h) in face:
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)  # Draw rectangles around faces

    cv2.imshow("FaceDetection", img)               # Display the frame with detections
    key = cv2.waitKey(1)
    if key == 27:                                  # Exit on ESC key
        break

cam.release()                                      # Release video capture object
cv2.destroyAllWindows()                            # Close all OpenCV windows
