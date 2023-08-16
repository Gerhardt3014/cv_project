import cv2

# Create a video capture object

cap = cv2.VideoCapture(0, cv2.CAP_V4L2)  # 调用摄像头‘0'一般是打开电脑自带摄像头，‘1'是打开外部摄像头（只有一个摄像头的情况）
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))  # 在AGX上不要去直接设置帧率，不好用，主要是opencv默认为YUY2格式的视频流，最高帧率只有40,而这个MJPG格式可以很高


# Check if the video capture object was successfully created
if not cap.isOpened():
    print("Error: Could not open camera.")
    exit()

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()

    # Check if the frame was successfully captured
    if not ret:
        print("Error: Could not read frame.")
        break

    # Display the captured frame
    cv2.imshow('Camera', frame)

    # Break the loop if the 'ESC' key is pressed
    if cv2.waitKey(1) & 0xFF == 27:
        break

# Release the video capture object
cap.release()

# Close all OpenCV windows
cv2.destroyAllWindows()
