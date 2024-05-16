import cv2

# Initialize the camera
cap = cv2.VideoCapture(0)

# Load pre-trained face and eye cascade classifiers
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

# Parameters for calculating blink rate and sleep detection
blink_frames = 0
blink_threshold = 10
blink_count = 0
sleep_threshold = 3  # Örnek bir uyku eşiği (gözlerin kapanma sayısı)
eye_closure_threshold = 0.2  # Göz kapanma eşiği

# Main loop
while True:
    # Read a frame from the camera
    ret, frame = cap.read()
    
    # Check if frame is successfully read
    if not ret:
        print("Cannot read frame from the camera!")
        break

    # Convert the frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the frame
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.05, minNeighbors=5, minSize=(30, 30))

    # Draw rectangles around detected faces
    for (x, y, w, h) in faces:
        # Draw rectangle around the face
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
        
        # Extract the region of interest (ROI) for face detection in grayscale and color
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = frame[y:y+h, x:x+w]

        # Detect eyes in the face region
        eyes = eye_cascade.detectMultiScale(roi_gray)
        for (ex, ey, ew, eh) in eyes:
            # Draw rectangle around the eyes
            cv2.rectangle(roi_color, (ex, ey), (ex+ew, ey+eh), (0, 255, 0), 2)

            # Calculate eye closure ratio
            eye_closure_ratio = float(eh) / h

            # Check if eye is closed or blinking
            if eye_closure_ratio < eye_closure_threshold:
                blink_frames += 1
            else:
                if blink_frames >= blink_threshold:
                    # Check if face is tilted upward or eyes are closed
                    if ey < h * 0.25 or (ey + eh) > h * 0.75:  # Gözler yüzün üst veya alt kısmında (yüzün 1/4'ü veya 3/4'ü)
                        blink_count += 1
                        print("Göz Uykusu Tespit Edildi! Toplam Göz Kırpma Sayısı:", blink_count)
                        if blink_count >= sleep_threshold:
                            print("UYARI: Uykulu Görünüyorsunuz!")
                    else:
                        print("Göz kapağı kapanması tespit edildi.")
                blink_frames = 0

    # Display the output
    cv2.imshow('Gaze Tracking', frame)

    # Exit on key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
