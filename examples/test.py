import os
import json
from PIL import Image, ImageDraw, ImageFont
import base64
import io
from datetime import datetime
import google.generativeai as genai
from google.generativeai.types import FunctionDeclaration, Tool
import re

# --- Configuration & Prompts ---
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "YOUR_GOOGLE_API_KEY_HERE")
# Consider "gemini-1.5-pro-latest" for better results on complex tasks like mask generation.
GEMINI_MODEL_NAME = "gemini-2.5-flash-preview-05-20"
GEMINI_MODEL = None

# --- Function Definition for Gemini Tool ---
record_object_func = FunctionDeclaration(
    name="record_detected_object",
    description="Records a detected object with its label, bounding box, and segmentation mask. Call this function for each detected bird or insect.",
    parameters={
        "type_": "OBJECT",
        "properties": {
            "label": {
                "type_": "STRING",
                "description": "A descriptive label for the object (e.g., 'bird', 'insect')."
            },
            "box_2d": {
                "type_": "ARRAY",
                "items": {"type_": "INTEGER"},
                "description": (
                    "A bounding box ('box_2d') in the format [y0, x0, y1, x1] with normalized coordinates between 0 and 1000."
                )
            },
            "mask": {
                "type_": "STRING",
                "description": (
                    "A base64 encoded png that is a probability map with values between 0 and 255."
                )
            }
        },
        "required": ["label", "box_2d", "mask"]
    }
)
OBJECT_EXTRACTION_TOOL = Tool(function_declarations=[record_object_func])

SAFETY_SETTINGS = [
    {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
    {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
    {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
    {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"},
]

B64_1X1_OPAQUE_BLACK_PNG = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNkYAAAAAYAAjCB0C8AAAAASUVORK5CYII="
MOCK_API_RESPONSE_DATA = [
    {
        "label": "the bird (mock)",
        "box_2d": [160, 75, 440, 485],
        "mask": B64_1X1_OPAQUE_BLACK_PNG
    },
    {
        "label": "the insect (mock)",
        "box_2d": [190, 280, 70, 60],
        "mask": B64_1X1_OPAQUE_BLACK_PNG
    }
]

# --- Helper Functions ---

def initialize_gemini_model():
    global GEMINI_MODEL, GOOGLE_API_KEY
    if GEMINI_MODEL is None:
        if not GOOGLE_API_KEY or GOOGLE_API_KEY == "YOUR_GOOGLE_API_KEY_HERE":
            print("Error: GOOGLE_API_KEY is not configured.")
            return None
        print(f"Initializing Gemini model: {GEMINI_MODEL_NAME}...")
        try:
            genai.configure(api_key=GOOGLE_API_KEY)
            GEMINI_MODEL = genai.GenerativeModel(GEMINI_MODEL_NAME)
            print(f"Gemini model '{GEMINI_MODEL_NAME}' initialized.")
        except Exception as e:
            print(f"Error: Failed to initialize Gemini model: {e}")
            GEMINI_MODEL = None
    return GEMINI_MODEL

def load_image(image_source: str) -> Image.Image | None:
    try:
        if image_source.startswith(('http://', 'https://')):
            import requests
            response = requests.get(image_source, stream=True, timeout=30, headers={"User-Agent" : "GeminiSegmentation/0.0.1A"})
            response.raise_for_status()
            img = Image.open(io.BytesIO(response.content))
        else:
            if not os.path.exists(image_source):
                print(f"Error: Image path not found: {image_source}")
                return None
            img = Image.open(image_source)
        return img.convert("RGBA")
    except ImportError: print("Error: 'requests' library needed for URL images."); return None
    except requests.exceptions.RequestException as e: print(f"Error fetching URL: {e}"); return None
    except FileNotFoundError: print(f"Error: Image not found: {image_source}"); return None
    except Exception as e: print(f"Error loading image: {e}"); return None

def decode_json_blocks(response : str):
    pattern = re.compile(
        r"^```(?:json|JSON)\s*\n(.*?)\n^```$",
        re.MULTILINE | re.DOTALL
    )
    json_objects = []
    for match_content in pattern.findall(response):
        try:
            data = json.loads(match_content.strip())
            json_objects.append(data)
        except json.JSONDecodeError as e:
            print(f"Warning: Found a JSON block, but it's invalid and will be skipped. Error: {e}")
            pass
    return json_objects

def call_segmentation_api(image: Image.Image, use_mock: bool = True) -> list | None:
    if use_mock:
        print("Using MOCK API response.")
        return MOCK_API_RESPONSE_DATA

    model = initialize_gemini_model()
    if not model:
        print("Error: Gemini model not initialized.")
        return None

    print(f"Attempting to call Google Gemini API (model: {GEMINI_MODEL_NAME}) with function calling...")

    if not isinstance(image, Image.Image):
        print("Error: Invalid image type. Expected PIL.Image.")
        return None

    # Main prompt now includes image dimensions and strongly typed instructions
    # The details of the output structure are in the tool's FunctionDeclaration.
    prompt_for_gemini = f"""
Task: Analyze the provided image and give the segmentation masks for all insects and birds.
    Output a JSON list of segmentation masks where each entry contains the 2D
    bounding box in the key "box_2d", the segmentation mask in key "mask", and
    the text label in the key "label".
"""
    # print(f"DEBUG: Sending prompt to Gemini:\n{prompt_for_gemini}") # For debugging

    try:
        print("Sending request to Gemini API...")
        response = model.generate_content(
            [prompt_for_gemini, image],
            generation_config=genai.types.GenerationConfig(
                temperature=1,
            ),
            # tools=[OBJECT_EXTRACTION_TOOL],
            safety_settings=SAFETY_SETTINGS,
            request_options={"timeout": 300} # Increased timeout further
        )
        
        # print(f"DEBUG: Full Gemini Response object:\n{response}") # For debugging

        api_response_data = []
        if response.text:
            response_data = decode_json_blocks(response.text)
            if len(response_data) > 1:
                print(f'WARNING: Found multiple JSON objects: {response_data=}')
            elif len(response_data) == 0:
                raise RuntimeError(f'ERROR: No JSON objects found in model output: {response}')
            response_data = response_data[0]
            print(f'DEBUG: {response_data=}')
            for element in response_data:
                label = element.get('label', 'unknown_label')
                box_2d = element.get('box_2d', [])
                mask = element.get('mask', '')
                detection_item = {
                    "label" : label,
                    "box_2d" : [int(x) for x in box_2d] if box_2d and len(box_2d) == 4 else [],
                    "mask" : mask
                }
                if not detection_item["label"] or not detection_item["box_2d"]: # box_2d check simplified
                    print(f"Warning: Received malformed data from Gemini tool call (label or box_2d invalid): {detection_item}. Skipping item.")
                    continue
                api_response_data.append(detection_item)

        if not api_response_data:
            text_fallback = response.text if hasattr(response, 'text') else ""
            print(f"Warning: Gemini did not make the expected function calls. Text fallback: '{text_fallback.strip()}'")
            if hasattr(response, 'prompt_feedback'):
                 print(f"Prompt Feedback: {response.prompt_feedback}")
            if response.candidates and hasattr(response.candidates[0], 'finish_reason'):
                 print(f"Finish Reason: {response.candidates[0].finish_reason}")
                 if hasattr(response.candidates[0], 'safety_ratings'):
                    print(f"Safety Ratings: {response.candidates[0].safety_ratings}")

        return api_response_data

    except Exception as e:
        print(f"Error: Google Gemini API request failed: {e.__class__.__name__}: {e}")
        current_response = locals().get('response')
        if current_response:
            if hasattr(current_response, 'prompt_feedback'): print(f"Prompt Feedback: {current_response.prompt_feedback}")
            if current_response.candidates and hasattr(current_response.candidates[0], 'finish_reason'):
                 print(f"Finish Reason: {current_response.candidates[0].finish_reason}")
                 if hasattr(current_response.candidates[0], 'safety_ratings'):
                    print(f"Safety Ratings: {current_response.candidates[0].safety_ratings}")
        return None

def parse_api_response(api_response_data: list | None) -> list:
    parsed_detections = []
    if api_response_data is None:
        print("Warning: parse_api_response received None. Returning empty list.")
        return []
    if not isinstance(api_response_data, list):
        print(f"Warning: API response data is not a list. Got: {type(api_response_data)}. Data: {api_response_data}")
        return []

    for item_idx, item in enumerate(api_response_data):
        if not isinstance(item, dict):
            print(f"Warning: API response item #{item_idx} is not a dictionary. Skipping item: {item}")
            continue
        try:
            label = item.get('label', f'unknown_item_{item_idx}')
            box_raw = item.get('box_2d')
            mask_b64 = item.get('mask') # Could be empty string

            if not (isinstance(box_raw, list) and len(box_raw) == 4 and all(isinstance(n, (int, float)) for n in box_raw)):
                print(f"Warning: Invalid 'box_2d' format for item '{label}' ({item_idx}). Box: {box_raw}. Skipping item.")
                continue
            box = [int(x) for x in box_raw] # Ensure integers [xmin, ymin, width, height]

            mask_image = None
            if isinstance(mask_b64, str) and mask_b64.strip(): # Only process if non-empty string
                try:
                    if ',' in mask_b64: mask_b64 = mask_b64.split(',', 1)[1]
                    mask_bytes = base64.b64decode(mask_b64)
                    mask_image = Image.open(io.BytesIO(mask_bytes)).convert("RGB")
                except (base64.binascii.Error, IOError, TypeError) as e_mask:
                    print(f"Warning: Could not decode/open mask for '{label}'. Base64 (first 50): '{mask_b64[:50]}...'. Error: {e_mask}")
            elif not mask_b64:
                print(f"Info: Item '{label}' has an empty mask string, as provided by API or due to parsing.")

            parsed_detections.append({
                "label": label, "box_2d": box, "mask_image": mask_image,
                "mask_base64_original": item.get('mask')
            })
        except KeyError as e: print(f"Warning: Item #{item_idx} missing key: {e}. Item: {item}")
        except Exception as e: print(f"Warning: Error processing item {item.get('label', f'Item #{item_idx}')}: {e}")
    return parsed_detections

def visualize_detections(original_image: Image.Image, detections: list) -> Image.Image:
    viz_image = original_image.copy().convert("RGBA")
    draw = ImageDraw.Draw(viz_image)
    img_width, img_height = original_image.size

    try: font = ImageFont.truetype("arial.ttf", 18)
    except IOError:
        try: font = ImageFont.load_default(size=18)
        except AttributeError: font = ImageFont.load_default()

    colors = [(255,0,0,255), (0,0,255,255), (0,255,0,255), (255,255,0,255), (255,0,255,255), (0,255,255,255)]
    mask_alpha = 128

    for i, det in enumerate(detections):
        label = det['label']
        box = det['box_2d'] # [xmin, ymin, width, height]
        mask_image_pil = det.get('mask_image')

        if not (isinstance(box, list) and len(box) == 4 and all(isinstance(c, int) for c in box)):
            print(f"Warning: Invalid box_2d for '{label}' in visualization: {box}. Skipping.")
            continue
        
        y0, x0, y1, x1 = [v/1000 for v in box]
        x0 *= img_width
        x1 *= img_width
        y0 *= img_height
        y1 *= img_height
        y0, x0, y1, x1 = [int(round(v)) for v in [y0, x0, y1, x1]] 
        w, h = (x1 - x0), (y1 - y0)

        # Basic sanity check for bounding box coordinates against image dimensions
        if not (0 <= x0 < img_width and 0 <= y0 < img_height and 0 <= x1 < img_width and 0 <= y1 < img_height):
            print(f"Warning: Bounding box for '{label}' [{x0}, {y0}, {x1}, {y1}] ({box}) seems out of image bounds ({img_width}x{img_height}) or has invalid size. Clamping/Skipping for visualization.")
            # Optional: Clamp or skip. For now, we'll proceed but it might draw oddly or error.
            # Example clamping (simple, might need more sophisticated logic):
            # xmin = max(0, xmin)
            # ymin = max(0, ymin)
            # w = min(w, img_width - xmin)
            # h = min(h, img_height - ymin)
            # if w <=0 or h <=0: continue # Skip if clamped to no size
        if x0 > x1 or y0 > y1:
            print(f'Warning some image coordinates seem to be inverted for "{label}" [{x0}, {y0}, {x1}, {y1}] ({box}) in image bounds ({img_width}x{img_height})')

        color = colors[i % len(colors)]
        outline_color = color
        fill_color_for_mask = (color[0], color[1], color[2], mask_alpha)

        draw.rectangle([x0, y0, x1, y1], outline=outline_color, width=3)
        
        try:
            if hasattr(draw, 'textbbox'): text_bbox = draw.textbbox((0,0), label, font=font); text_height = text_bbox[3] - text_bbox[1]
            else: text_size = draw.textsize(label, font=font); text_height = text_size[1]
        except Exception: text_height = 20

        text_x, text_y = x0, y0 - text_height - 5
        if text_y < 5: text_y = x0 + 5
        draw.text((text_x, text_y), label, fill=outline_color, font=font)

        if mask_image_pil:
            if not (w > 0 and h > 0):
                print(f"Info: Box for '{label}' has zero area ({w}x{h}). Skipping mask overlay.")
                # continue # If in a loop, otherwise handle as appropriate
            else:
                # --- Start of modifications ---
                original_mask_pil_size = mask_image_pil.size # Should be (256, 256)

                # 1. Convert the (256,256,3) RGB image to a binary mask ("L" mode)
                #    based on the threshold (pixel values > 127).
                #    First, convert to grayscale if it's RGB.
                #    If it's already "L", this conversion doesn't hurt.
                try:
                    if mask_image_pil.mode == "RGB" or mask_image_pil.mode == "RGBA":
                        mask_grayscale_pil = mask_image_pil.convert("L")
                    elif mask_image_pil.mode == "L":
                        mask_grayscale_pil = mask_image_pil # Already grayscale
                    else:
                        print(f"Warning: Mask for '{label}' has unexpected mode '{mask_image_pil.mode}'. Attempting to convert to 'L'.")
                        mask_grayscale_pil = mask_image_pil.convert("L")

                    # Apply threshold: pixels > 127 become 255 (mask), others 0 (background)
                    # The .point method applies the lambda to each pixel value.
                    # Output is an "L" mode image with 0s and 255s.
                    actual_mask_for_pasting = mask_grayscale_pil.point(lambda p: 255 if p > 127 else 0, mode="L")

                except Exception as e:
                    print(f"Error processing input mask for '{label}' (thresholding step): {e}. Skipping mask overlay.")
                    actual_mask_for_pasting = None # Ensure it's None so the rest of the block is skipped

                if actual_mask_for_pasting:
                    # 2. Check if the processed mask (still 256x256 at this point)
                    #    needs to be resized to the bounding box dimensions (w,h).
                    if actual_mask_for_pasting.size != (w, h):
                        print(f"Info: Resizing mask for '{label}'. Original (after thresholding): {actual_mask_for_pasting.size}, Target BBox: ({w},{h}).")
                        try:
                            actual_mask_for_pasting = actual_mask_for_pasting.resize((w, h), Image.Resampling.NEAREST)
                        except Exception as resize_err:
                            print(f"         Could not resize mask for '{label}'. Skipping. Error: {resize_err}")
                            actual_mask_for_pasting = None # Skip if resize fails

                    if actual_mask_for_pasting: # Check again if resize was successful
                        # 3. Create the colored patch with the size of the bounding box (w,h)
                        colored_segment_patch = Image.new("RGBA", (w, h), fill_color_for_mask)
                        temp_overlay = Image.new("RGBA", viz_image.size, (0, 0, 0, 0)) # Transparent overlay

                        try:
                            # Paste the colored patch onto the temporary overlay,
                            # using the (potentially resized) actual_mask_for_pasting.
                            # The actual_mask_for_pasting should now be (w,h) in size.
                            temp_overlay.paste(colored_segment_patch, (x0, y0), actual_mask_for_pasting)
                            viz_image = Image.alpha_composite(viz_image, temp_overlay)
                        except ValueError as ve:
                            # Common error: "bad transparency mask" if mask isn't "L" or "1" or size mismatch
                            print(f"Error during mask composition for '{label}': {ve}")
                            print(f"         Mask mode: {actual_mask_for_pasting.mode}, Mask size: {actual_mask_for_pasting.size}")
                            print(f"         Patch size: {colored_segment_patch.size}, Paste coords: ({x0},{y0})")
                        except Exception as e:
                            print(f"Error during mask composition for '{label}': {e}")
                # --- End of modifications ---

        elif det.get("mask_base64_original") and det["mask_base64_original"].strip(): # type: ignore
            print(f"Info: Mask for '{label}' was provided as base64 but failed to load or process. Not visualized.")
    return viz_image

def process_image_and_save(image_source: str, output_dir_base: str = "output", use_mock_api: bool = True):
    print(f"Processing image: {image_source}")
    original_image = load_image(image_source)
    if not original_image: return

    print("Requesting segmentations from API...")
    api_response_data = call_segmentation_api(original_image, use_mock=use_mock_api)

    if api_response_data is None: print("Failed to get API response. Exiting."); return
    if not api_response_data and isinstance(api_response_data, list):
        print("API returned no detections.")

    print("Parsing API response...")
    parsed_detections = parse_api_response(api_response_data)

    print("Generating visualization...")
    visualization_image = visualize_detections(original_image, parsed_detections)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    fname_base = "".join(c if c.isalnum() or c in (' ', '.', '_') else '_' for c in os.path.splitext(os.path.basename(image_source))[0])
    output_dir = os.path.join(output_dir_base, f"{fname_base.replace(' ', '_')}_{timestamp}")
    os.makedirs(output_dir, exist_ok=True)
    print(f"Saving outputs to: {output_dir}")

    visualization_image.save(os.path.join(output_dir, "visualization.png"))
    print(f"  Visualization saved.")
    with open(os.path.join(output_dir, "api_function_call_args.json"), "w") as f:
        json.dump(api_response_data, f, indent=2)
    print(f"  API function call args saved.")
    
    savable_parsed = [{"label": d["label"], "box_2d": d["box_2d"], "mask_base64": d["mask_base64_original"]} for d in parsed_detections]
    with open(os.path.join(output_dir, "parsed_detections_with_masks.json"), "w") as f:
        json.dump(savable_parsed, f, indent=2)
    print(f"  Parsed detections saved.")
    print("Processing complete.")

if __name__ == "__main__":
    IMAGE_URL = "https://upload.wikimedia.org/wikipedia/commons/thumb/0/0f/Green_Figbird.jpg/640px-Green_Figbird.jpg"
    # IMAGE_URL = "https://storage.googleapis.com/gweb-cloudblog-publish/images/GettyImages-1193149035.max-2000x2000.jpg"
    # IMAGE_URL = "your_local_image.jpg" # Test with a local image

    USE_MOCK_API_CALL = False  # <<< SET TO FALSE TO USE REAL GEMINI API >>>
    # If using REAL API, ensure GOOGLE_API_KEY is set and consider GEMINI_MODEL_NAME.
    # For potentially better mask/box results, try:
    # GEMINI_MODEL_NAME = "gemini-1.5-pro-latest" # (and ensure it's set above too)


    if not USE_MOCK_API_CALL:
        if not GOOGLE_API_KEY or GOOGLE_API_KEY == "YOUR_GOOGLE_API_KEY_HERE":
            print("ATTENTION: Real API call selected, but GOOGLE_API_KEY is not configured. Exiting.")
            exit(1)
        print(f"Running with REAL Gemini API: {GEMINI_MODEL_NAME}.")
        if GOOGLE_API_KEY == "YOUR_GOOGLE_API_KEY_HERE":
             print("WARNING: GOOGLE_API_KEY is placeholder. This will fail.")
    else:
        print("Running with MOCK API calls.")

    try:
        _ = ImageFont.load_default(size=10)
        print("Pillow font rendering with size supported.")
    except Exception: print("Pillow font rendering test failed.")

    process_image_and_save(IMAGE_URL, use_mock_api=USE_MOCK_API_CALL)

    print("\n--- MASK & BOUNDING BOX NOTES ---")
    print("1. Mask Generation: This script asks Gemini to directly generate base64 PNG masks.")
    print("   If masks are missing/poor, it's a hard task. 'gemini-1.5-pro-latest' might do better.")
    print("   The most robust way is a 2-stage process (e.g., Gemini for boxes, then SAM for masks), not done here.")
    print("2. Bounding Box Accuracy: The prompt now includes image dimensions to help Gemini.")
    print("   If boxes are still off, it could be model limitations or image characteristics.")
    print("   Check 'api_function_call_args.json' to see raw coordinates from Gemini.")
    print("   The visualization code includes a warning if mask size != bbox size, which indicates an API error.")
    print("--------------------------------")