import cv2

alg = "haarcascade_frontalface_default.xml"  # Load the Haar Cascade XML file for face detection
haar_cascade = cv2.CascadeClassifier(alg)    # Initialize the face detector

video_path = "Upload Your Image"               # Path to input image
img = cv2.imread(video_path)                    # Read the image from the given path

while True:
    grayImg = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # Convert image to grayscale for detection
    face = haar_cascade.detectMultiScale(grayImg, 1.3, 4)  # Detect faces and get bounding boxes
    
    for (x, y, w, h) in face:
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)  # Draw rectangle around detected faces
    
    cv2.imshow("FaceDetection", img)  # Show the image with detected faces
    
    key = cv2.waitKey(1)
    if key == 27:   # Exit if ESC key is pressed
        break

cv2.destroyAllWindows()  # Close all OpenCV windows
