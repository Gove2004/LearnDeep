import io
import uvicorn
from pathlib import Path
import torch
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import FileResponse
from PIL import Image
from torchvision import transforms
from dataset import GarbageClassificationImageClassificationDataset
from model import ResNet18

CKPT_PATH = Path("./Checkpoints/best_image_classification_model.pth")
DATA_DIR = Path("./Datasets/GarbageClassification")
INDEX_HTML = Path("./ImageClassification/index.html")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

class InferenceService:
    def __init__(self):
        self.device = DEVICE
        dataset = GarbageClassificationImageClassificationDataset(None, None, DATA_DIR)
        self.channels = dataset.get_channel_nums()
        self.class_names = dataset.get_class_names()
        self.mode = "RGB" if self.channels == 3 else "L"

        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])

        model = ResNet18(input_size=(self.channels, 224, 224), num_classes=len(self.class_names))
        model.load_state_dict(torch.load(CKPT_PATH, map_location=self.device))
        model.to(self.device)
        model.eval()
        self.model = model

    def predict(self, image: Image.Image, topk=5):
        x = self.transform(image.convert(self.mode)).unsqueeze(0).to(self.device)
        with torch.no_grad():
            probs = torch.softmax(self.model(x), dim=1)
            top_p, top_i = torch.topk(probs[0], min(topk, len(self.class_names)))
        return [
            {"class": self.class_names[i], "prob": round(p.item(), 4)}
            for p, i in zip(top_p, top_i)
        ]


svc = InferenceService()
app = FastAPI(title="Image Classification Service")


@app.get("/")
def index():
    return FileResponse(INDEX_HTML)


@app.get("/api/classes")
def classes():
    return {"classes": svc.class_names}


@app.post("/api/predict")
async def predict(image: UploadFile = File(...)):
    data = await image.read()
    img = Image.open(io.BytesIO(data))
    result = svc.predict(img)
    return {"result": result}


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
