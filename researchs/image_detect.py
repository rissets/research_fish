from urllib.request import urlopen

import cv2
import torch
import torchvision.models as models
import torchvision.transforms as T

# --- 1. SETUP MODEL AND LABELS ---

# Load the pre-trained ResNet-50 model with the latest weights
print("Loading ResNet-50 model...")
weights = models.ResNet50_Weights.IMAGENET1K_V2
model = models.resnet50(weights=weights)
model.eval()  # Set the model to evaluation mode

# Get the appropriate preprocessing transformations for the model
preprocess = weights.transforms()

# Download and load the ImageNet class labels
print("Downloading ImageNet labels...")
url = "https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt"
try:
    response = urlopen(url)
    labels_text = response.read().decode("utf-8")
    labels = [s.strip() for s in labels_text.split("\n")]
except Exception as e:
    print(f"Error downloading labels: {e}")
    labels = ["Label download failed"] * 1000

# --- 2. LOAD AND PROCESS THE IMAGE ---

# IMPORTANT: Ganti path ini dengan lokasi gambar ikan Anda
image_path = "ikan.jpg"

print(f"Loading image from: {image_path}")
# Load the image using OpenCV
frame = cv2.imread(image_path)

if frame is None:
    print("Error: Could not read the image. Please check the file path.")
else:
    # Convert the image from BGR (OpenCV format) to RGB (PyTorch format)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Convert the image to a PyTorch tensor
    img_t = T.ToTensor()(rgb_frame)

    # Preprocess the tensor and add a batch dimension
    batch_t = preprocess(img_t).unsqueeze(0)

    # --- 3. PERFORM CLASSIFICATION ---

    # Run inference to get the prediction
    with torch.no_grad():
        out = model(batch_t)

    # Get the top prediction
    _, index = torch.max(out, 1)
    percentage = torch.nn.functional.softmax(out, dim=1)[0] * 100

    # Get the label name and confidence score
    label_name = labels[index[0]]
    score = percentage[index[0]].item()

    # Print the result to the console
    print("\n--- Prediction ---")
    print(f"Label: {label_name}")
    print(f"Confidence Score: {score:.2f}%")
    print("--------------------")

    # --- 4. DISPLAY THE RESULT ---

    # Draw the prediction text on the image
    text_to_display = f"{label_name}: {score:.2f}%"
    cv2.putText(
        frame, text_to_display, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2
    )

    # Display the annotated image in a window
    cv2.imshow("Fish Classification Result", frame)

    # Wait for a key press to close the image window
    print("\nPress any key to close the image window.")
    cv2.waitKey(0)
    cv2.destroyAllWindows()
