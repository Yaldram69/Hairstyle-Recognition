from ultralytics import YOLO
import matplotlib.pyplot as plt
import cv2


def predict_hairstyles_simple(image_path, model_path=r':\Users\CREATIVE IT PARK\PyCharmMiscProject\By Haroon\HairStyle_Recognition\HairStyle_Recognition\best (1).pt', conf_threshold=0.5):
    """
    Predict hairstyles using a trained YOLO model, automatically handling segmentation and excluding face-related classes.

    Args:
        image_path (str): Path to the input image.
        model_path (str): Path to the trained YOLO model. Default is 'path/to/your/trained_model.pt'.
        conf_threshold (float): Confidence threshold for predictions. Default is 0.5.

    Returns:
        None. Displays the filtered predictions using Matplotlib.
    """
    # Load the trained YOLO model
    model = YOLO(model_path)

    # Predict using the YOLO model
    results = model.predict(source=image_path, conf=conf_threshold, save=False, save_txt=False, save_crop=False)

    # Define hairstyle class names
    hairstyle_classes = ['curly', 'kinky', 'straight', 'wavy']  # Class names for hairstyles

    # Get indices of hairstyle classes
    hairstyle_indices = [cls_idx for cls_idx, cls_name in model.names.items() if cls_name in hairstyle_classes]

    # Filter predictions: Only keep those belonging to hairstyle classes
    keep_indices = [i for i, cls in enumerate(results[0].boxes.cls.cpu().numpy()) if int(cls) in hairstyle_indices]
    results[0].boxes = results[0].boxes[keep_indices]

    # Check if masks exist and filter them too
    if results[0].masks:
        results[0].masks.data = results[0].masks.data[keep_indices]

    # Plot the filtered predictions
    annotated_image = results[0].plot()
    image_rgb = cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB for Matplotlib
    plt.figure(figsize=(12, 8))
    plt.imshow(image_rgb)
    plt.axis('off')
    plt.title("Hairstyle")
    plt.show()


img1 = predict_hairstyles_simple(image_path=r'C:\Users\CREATIVE IT PARK\PyCharmMiscProject\By Haroon\HairStyle_Recognition\HairStyle_Recognition\Testing_images\curly.jpeg', model_path=r'C:\Users\CREATIVE IT PARK\PyCharmMiscProject\By Haroon\HairStyle_Recognition\HairStyle_Recognition\best (1).pt')
