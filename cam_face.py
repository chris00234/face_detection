import cv2 as cv

# Open the built-in camera (default camera index is usually 0)

haar_cascade = cv.CascadeClassifier('Face/haar_face.xml')

cap = cv.VideoCapture(0)



# Check if the camera is opened successfully
if not cap.isOpened():
    print("Error: Could not open camera.")
    exit()

while True:
    # Read a frame from the camera
    ret, frame = cap.read()

    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    
    faces_rect = haar_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=3)
    
    # Check if the frame was read successfully
    if not ret:
        print("Error: Failed to capture frame.")
        break
    
    # Display the frame
    for(x,y,w,h) in faces_rect:
        cv.rectangle(frame, (x,y), (x+w,y+h), (0,255,0), thickness=2)
    cv.imshow("Camera Feed", frame)

    # Break the loop if the user presses the 'q' key
    if cv.waitKey(1) & 0xFF == ord('q'):
        break

# Release the camera and close the OpenCV window
cap.release()
cv.destroyAllWindows()
