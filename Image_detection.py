import cv2
from ultralytics import YOLO
import os
import matplotlib.pyplot as plt


# Load YOLOv8m model
model = YOLO("yolov8m.pt")

# Path to image
image_path = r"/Users/revanth/Documents/emergency_vehicle/t5.jpg"

# Read image
image = cv2.imread(image_path)
if image is None:
    print(" Cannot open image file")
    print(f"Please check the path: {image_path}")
    exit()

# Define emergency and normal vehicles
emergency_vehicles = ['ambulance', 'emergency vehicle', 'truck', 'fire truck']
normal_vehicles = ['car', 'motorcycle']
conf_threshold = 0.3

# Initialize detection counters
total_detections = 0
emergency_detections = 0

# Get image dimensions and set adaptive drawing scale
height, width = image.shape[:2]
scale = max(height, width) / 1000
box_thickness = max(2, int(scale * 2))
font_scale = scale * 0.8
text_thickness = max(1, int(scale * 1.5))

try:
    # Run detection
    results = model(image, conf=conf_threshold)

    for r in results:
        for box in r.boxes:
            cls = int(box.cls[0])
            conf = float(box.conf[0])
            label = model.names[cls].lower()

            # Filter relevant vehicles
            if label in emergency_vehicles or label in normal_vehicles:
                x1, y1, x2, y2 = map(int, box.xyxy[0])

                is_emergency = (label in emergency_vehicles or
                                'truck' in label or
                                'emergency' in label)

                color = (0, 0, 255) if is_emergency else (0, 255, 0)
                label_text = f"Emergency Vehicle {conf:.2f}" if is_emergency else f"{label} {conf:.2f}"

                # Draw bounding box
                cv2.rectangle(image, (x1, y1), (x2, y2), color, box_thickness)

                # Draw text background
                (text_w, text_h), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, text_thickness)
                cv2.rectangle(image, (x1, y1 - text_h - 10), (x1 + text_w, y1), color, -1)

                # Draw label text
                cv2.putText(image, label_text, (x1, y1 - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), text_thickness)

                # Update counts
                total_detections += 1
                if is_emergency:
                    emergency_detections += 1

    # Resize image for display if too large
    max_width, max_height = 1280, 720
    if width > max_width or height > max_height:
        scale_w = max_width / width
        scale_h = max_height / height
        final_scale = min(scale_w, scale_h)
        image = cv2.resize(image, (int(width * final_scale), int(height * final_scale)))

    # Print stats
    print(f"✅ Total Detections: {total_detections}")
    print(f"🚨 Emergency Vehicles Detected: {emergency_detections}")

    # Display image
    cv2.imshow("Emergency Vehicle Detection", image)
    cv2.waitKey(0)

except Exception as e:
    print(f"⚠️ Error occurred: {str(e)}")
    

finally:
    cv2.destroyAllWindows()


# Prepare data
labels_list = []
confidences_list = []
colors_list = []

for r in results:
    for box in r.boxes:
        cls = int(box.cls[0])
        conf = float(box.conf[0])
        label = model.names[cls].lower()

        if label in emergency_vehicles:
            labels_list.append(label)
            confidences_list.append(conf)
            colors_list.append('red')  # Emergency vehicles in red
        elif label in normal_vehicles:
            labels_list.append(label)
            confidences_list.append(conf)
            colors_list.append('skyblue')  # Normal vehicles in blue

# Plot
if labels_list:
    plt.figure(figsize=(12, 6))
    bars = plt.bar(range(len(confidences_list)), confidences_list, tick_label=labels_list, color=colors_list)

    # Annotate confidence scores on bars
    for bar, score in zip(bars, confidences_list):
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2.0, yval + 0.01, f'{score:.2f}', ha='center', va='bottom', fontsize=9)

    plt.xlabel('Vehicle Type')
    plt.ylabel('Confidence Score')
    plt.title('Confidence Scores of Detected Vehicles')
    plt.ylim([0, 1])
    plt.xticks(rotation=30)
    plt.tight_layout()
    plt.show()
else:
    print("No vehicles detected.")