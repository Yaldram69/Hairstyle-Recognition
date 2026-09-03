from ultralytics import YOLO
import cv2
import matplotlib.pyplot as plt

model = YOLO(r'C:\Users\CREATIVE IT PARK\PyCharmMiscProject\By Haroon\HairStyle_Recognition\HairStyle_Recognition\best (1).pt')  # Replace with your model's path



def predict_hairstyles_webcam(model_path=r'C:\Users\CREATIVE IT PARK\PyCharmMiscProject\By Haroon\HairStyle_Recognition\HairStyle_Recognition\best (1).pt', conf_threshold=0.5):
    """
    Predict hairstyles in real-time using a webcam and a trained YOLO model.

    Args:
        model_path (str): Path to the trained YOLO model. Default is '/content/best (1).pt'.
        conf_threshold (float): Confidence threshold for predictions. Default is 0.5.
    """
    # Load the trained YOLO model
    model = YOLO(model_path)

    # Define hairstyle class names
    hairstyle_classes = ['curly', 'kinky', 'straight', 'wavy']  # Class names for hairstyles

    # Get indices of hairstyle classes
    hairstyle_indices = [cls_idx for cls_idx, cls_name in model.names.items() if cls_name in hairstyle_classes]

    # Open the webcam
    cap = cv2.VideoCapture(0)  # 0 is the default webcam
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    # Loop to process each frame from the webcam
    while True:
        # Read a frame from the webcam
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame.")
            break

        # Predict using the YOLO model
        results = model.predict(source=frame, conf=conf_threshold, save=False, save_txt=False, save_crop=False)

        # Filter predictions: Only keep those belonging to hairstyle classes
        if results[0].boxes is not None:
            keep_indices = [i for i, cls in enumerate(results[0].boxes.cls.cpu().numpy()) if int(cls) in hairstyle_indices]
            results[0].boxes = results[0].boxes[keep_indices]

            # Check if masks exist and filter them too
            if results[0].masks:
                results[0].masks.data = results[0].masks.data[keep_indices]

        # Plot the filtered predictions on the frame
        annotated_frame = results[0].plot()

        # Display the annotated frame
        cv2.imshow("Hairstyle Prediction", annotated_frame)

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the webcam and close all OpenCV windows
    cap.release()
    cv2.destroyAllWindows()

# Run the function
predict_hairstyles_webcam()