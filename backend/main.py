from PIL import Image
import torch
import torchvision.transforms as transforms
from torchvision.models import mobilenet_v3_small
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
import joblib
import numpy as np
from fastapi import FastAPI, UploadFile, File, Form
from typing import List
import os
import uuid
import io

# =====================================================
# CONFIG
# =====================================================

DATASET_DIR = "dataset"
MODEL_DIR = "models"

os.makedirs(DATASET_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

# =====================================================
# MOBILENET FEATURE EXTRACTOR
# =====================================================

device = torch.device("cpu")

mobilenet = mobilenet_v3_small(weights="DEFAULT")
mobilenet.classifier = torch.nn.Identity()
mobilenet.eval()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# =====================================================
# FASTAPI APP
# =====================================================

app = FastAPI(title="Teachable Machine API")

# =====================================================
# UPLOAD SAMPLES
# =====================================================

@app.post("/upload-sample")
async def upload_sample(
    class_name: str = Form(...),
    images: List[UploadFile] = File(...)
):

    try:

        class_folder = os.path.join(
            DATASET_DIR,
            class_name
        )

        os.makedirs(
            class_folder,
            exist_ok=True
        )

        saved_files = []

        for image in images:

            extension = image.filename.split(".")[-1]

            random_name = (
                f"{uuid.uuid4()}.{extension}"
            )

            file_path = os.path.join(
                class_folder,
                random_name
            )

            with open(file_path, "wb") as f:
                f.write(await image.read())

            saved_files.append(random_name)

        return {
            "message": "Images uploaded successfully",
            "class_name": class_name,
            "total_files": len(saved_files)
        }

    except Exception as e:

        return {
            "error": str(e)
        }

# =====================================================
# TRAIN MODEL
# =====================================================

@app.post("/train")
def train_model():

    try:

        X = []
        y = []

        class_folders = [
            folder
            for folder in os.listdir(DATASET_DIR)
            if os.path.isdir(
                os.path.join(
                    DATASET_DIR,
                    folder
                )
            )
        ]

        if len(class_folders) < 2:

            return {
                "error":
                "Need at least 2 classes to train"
            }

        for class_name in class_folders:

            class_path = os.path.join(
                DATASET_DIR,
                class_name
            )

            for file_name in os.listdir(class_path):

                image_path = os.path.join(
                    class_path,
                    file_name
                )

                try:

                    image = Image.open(
                        image_path
                    ).convert("RGB")

                    tensor = transform(
                        image
                    ).unsqueeze(0)

                    with torch.no_grad():

                        features = mobilenet(
                            tensor
                        )

                    features = (
                        features
                        .squeeze()
                        .numpy()
                    )

                    X.append(features)
                    y.append(class_name)

                except Exception as img_error:

                    print(
                        f"Error processing "
                        f"{image_path}"
                    )

                    print(
                        str(img_error)
                    )

        if len(X) == 0:

            return {
                "error":
                "No valid images found"
            }

        X = np.array(X)

        encoder = LabelEncoder()

        y_encoded = encoder.fit_transform(y)

        model = LogisticRegression(
            max_iter=1000
        )

        model.fit(
            X,
            y_encoded
        )

        joblib.dump(
            model,
            os.path.join(
                MODEL_DIR,
                "model.pkl"
            )
        )

        joblib.dump(
            encoder,
            os.path.join(
                MODEL_DIR,
                "label_encoder.pkl"
            )
        )

        return {
            "message":
            "Training completed successfully",
            "samples":
            len(X),
            "classes":
            list(
                encoder.classes_
            )
        }

    except Exception as e:

        print("TRAINING ERROR:")
        print(str(e))

        return {
            "error": str(e)
        }

# =====================================================
# PREDICT
# =====================================================

@app.post("/predict")
async def predict(
    image: UploadFile = File(...)
):

    try:

        model_path = os.path.join(
            MODEL_DIR,
            "model.pkl"
        )

        encoder_path = os.path.join(
            MODEL_DIR,
            "label_encoder.pkl"
        )

        if (
            not os.path.exists(model_path)
            or
            not os.path.exists(encoder_path)
        ):

            return {
                "error":
                "Model not trained yet"
            }

        model = joblib.load(
            model_path
        )

        encoder = joblib.load(
            encoder_path
        )

        image_bytes = await image.read()

        img = Image.open(
            io.BytesIO(image_bytes)
        ).convert("RGB")

        tensor = transform(
            img
        ).unsqueeze(0)

        with torch.no_grad():

            features = mobilenet(
                tensor
            )

        features = features.numpy()

        pred = model.predict(
            features
        )[0]

        confidence = (
            model
            .predict_proba(features)
            .max()
        )

        label = (
            encoder
            .inverse_transform([pred])[0]
        )

        return {
            "class": label,
            "confidence":
            round(
                float(confidence * 100),
                2
            )
        }

    except Exception as e:

        print("PREDICTION ERROR:")
        print(str(e))

        return {
            "error": str(e)
        }

# =====================================================
# ROOT
# =====================================================

@app.get("/")
def root():

    return {
        "message":
        "Teachable Machine API Running"
    }