# main.py
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
import base64
from io import BytesIO

# Import your refactored modules
from services.yolo_slicer import YoloSlicer
from services import scryfall_client
from classification.cnn_classifier import CNNClassifier
from classification.cnn_classifier_v2 import CNNClassifierV2
# from classification.knn_searcher import KNNSearcher

# --- FastAPI App Initialization ---
app = FastAPI(title="Magic Card Classifier API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Load Models and Services ---
# These are loaded once when the application starts, which is efficient.
slicer = YoloSlicer()
cnn_classifier = CNNClassifier()
cnn_classifier_v2 = CNNClassifierV2()
#knn_searcher = KNNSearcher()
# --- End Initialization ---

@app.get("/")
def read_root():
    return {"message": "Welcome to the Magic Card Classifier API! Use the /classify endpoint to submit an image."}

@app.post("/classify")
async def classify_magic_card(
    useKNN: bool = Form(...),
    file: UploadFile = File(...),
    x: float = Form(...),
    y: float = Form(...)
):
    """
    Detects, crops, and classifies a Magic: The Gathering card from an image.
    Switches between k-NN and CNN classification based on the `useKNN` parameter.
    """
    image_data = await file.read()
    try:
        full_image = Image.open(BytesIO(image_data)).convert("RGB")
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid image file.")

    # Step 1: Slice the card from the full image using YOLOv5
    cropped_image, bounding_box = slicer.slice_card_from_image(full_image, x, y)
    
    if not cropped_image:
        raise HTTPException(status_code=404, detail="No card detected near the specified location.")

    # Encode cropped image to base64 for the response
    buffered = BytesIO()
    cropped_image.save(buffered, format="JPEG")
    card_image_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')

    # Step 2: Classify using the selected method
    if useKNN:
        # Use OpenSearch k-NN Searcher
        print("Using knn b")
        # initial_guesses = knn_searcher.search_by_image(cropped_image)
        # print(initial_guesses)
        # id_fetcher = scryfall_client.fetch_card_by_scryfall_id
        initial_guesses = cnn_classifier_v2.classify_card(cropped_image)
        print(initial_guesses)
        id_fetcher = scryfall_client.fetch_card_by_oracle_id
    else:
        # Use local CNN Classifier
        initial_guesses = cnn_classifier.classify_card(cropped_image)
        print(initial_guesses)
        id_fetcher = scryfall_client.fetch_card_by_oracle_id

    if not initial_guesses:
        raise HTTPException(status_code=404, detail="Classifier returned no results.")

    # Step 3: Hydrate top guesses with full Scryfall data
    top_guesses_hydrated = []
    for guess in initial_guesses:
        scryfall_data = id_fetcher(guess['predicted_id'])
        if "error" not in scryfall_data:
            scryfall_data['classification_confidence'] = guess['confidence']
            top_guesses_hydrated.append(scryfall_data)
            
    if not top_guesses_hydrated:
        raise HTTPException(status_code=404, detail="Could not fetch card details from Scryfall for any guess.")

    # Step 4: Assemble and return the final response
    return {
        "search_method": "k-NN" if useKNN else "CNN",
        "bounding_box": bounding_box,
        "card_image_base64": card_image_base64,
        "classification_confidence" : top_guesses_hydrated[0]['classification_confidence'],
        "scryfall_data": top_guesses_hydrated[0],
        "all_guesses": top_guesses_hydrated
    }