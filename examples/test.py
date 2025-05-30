import base64
import io
import json
import os
import re
from datetime import datetime

import google.generativeai as genai
from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm as TQDM

# --- Configuration & Prompts ---
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "YOUR_GOOGLE_API_KEY_HERE")
# Consider "gemini-1.5-pro-latest" for better results on complex tasks like mask generation.
GEMINI_MODEL_NAME = "gemini-2.5-flash-preview-05-20"
GEMINI_MODEL = None

SAFETY_SETTINGS = [
    {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
    {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
    {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
    {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"},
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
        ms = max(img.size)
        rat = 2000/ms
        img.resize([int(round(s*rat)) for s in img.size])
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

def call_segmentation_api(image: Image.Image, verbose : bool=False) -> list | None:
    model = initialize_gemini_model()
    if not model:
        raise RuntimeError("Error: Gemini model not initialized.")

    if verbose:
        print(f"Attempting to call Google Gemini API (model: {GEMINI_MODEL_NAME}) with function calling...")

    if not isinstance(image, Image.Image):
        raise TypeError("Error: Invalid image type. Expected PIL.Image.")

    # Main prompt now includes image dimensions and strongly typed instructions
    prompt_for_gemini = f"""
Analyze the provided image and give the segmentation masks for the bird(s) and its/their beak(s) (separately) and insect(s).
Output a JSON list of segmentation masks where each entry contains the 2D
bounding box in the key "box_2d", the segmentation mask in key "mask", and
the text label in the key "label".
"""
    # print(f"DEBUG: Sending prompt to Gemini:\n{prompt_for_gemini}") # For debugging

    try:
        if verbose:
            print("Sending request to Gemini API...")
        response = model.generate_content(
            [prompt_for_gemini, image],
            generation_config=genai.types.GenerationConfig(
                temperature=0.5
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
        raise RuntimeError("Warning: parse_api_response received None. Returning empty list.")
    if not isinstance(api_response_data, list):
        raise RuntimeError(f"Warning: API response data is not a list. Got: {type(api_response_data)}. Data: {api_response_data}")

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

def visualize_detections(original_image: Image.Image, detections: list[dict]) -> Image.Image:
    viz_image = original_image.copy().convert("RGBA")
    draw = ImageDraw.Draw(viz_image) # Initial draw object, will be updated if masks are applied
    img_width, img_height = viz_image.size

    # --- Setup ---
    try: font = ImageFont.truetype("arial.ttf", 16) # Slightly smaller font
    except IOError: # Fallback font loading
        try: font = ImageFont.load_default(size=16) # For Pillow 9.3.0+
        except (TypeError, AttributeError): # Catches TypeError for unexpected 'size' kwarg or AttributeError
            font = ImageFont.load_default()     # For Pillow < 9.3.0 or other issues

    color_palette_rgb = [
        (255,0,0), (0,0,255), (0,255,0), (255,255,0), (255,0,255), (0,255,255)
    ]
    mask_overlay_alpha = 128

    # --- Process Detections ---
    for i, det in enumerate(detections):
        label = det.get('label', 'N/A')
        raw_box = det.get('box_2d') # Expected: [y0, x0, y1, x1] in 0-1000
        mask_pil = det.get('mask_image')

        # 1. Calculate and Validate Bounding Box
        if not (isinstance(raw_box, list) and len(raw_box) == 4 and
                all(isinstance(c, (int, float)) for c in raw_box)):
            # print(f"Debug: Invalid box format for '{label}'. Skipping.")
            continue
        
        y0, x0, y1, x1 = raw_box
        y0 = int(round((y0 / 1000.0) * img_height))
        x0 = int(round((x0 / 1000.0) * img_width))
        y1 = int(round((y1 / 1000.0) * img_height))
        x1 = int(round((x1 / 1000.0) * img_width))

        if not (0 <= x0 < img_width and 0 <= y0 < img_height and \
                x0 < x1 <= img_width and y0 < y1 <= img_height): # Ensures x0<x1, y0<y1, and within bounds
            # print(f"Debug: Box for '{label}' out of bounds or invalid size [{x0},{y0},{x1},{y1}]. Skipping.")
            continue
        
        box_width_px, box_height_px = x1 - x0, y1 - y0 # Used for mask sizing

        # 2. Determine Colors
        base_rgb_color = color_palette_rgb[i % len(color_palette_rgb)]
        box_outline_rgba = (*base_rgb_color, 255)
        mask_fill_rgba = (*base_rgb_color, mask_overlay_alpha)

        # 3. Draw Bounding Box
        draw.rectangle([x0, y0, x1, y1], outline=box_outline_rgba, width=3)

        # 4. Draw Label with Background
        try: # Measure text
            if hasattr(draw, 'textbbox'): # Pillow 9.2.0+
                text_bb = draw.textbbox((0,0), label, font=font) # Measure from (0,0)
                text_w, text_h = text_bb[2] - text_bb[0], text_bb[3] - text_bb[1]
            else:
                text_w, text_h = draw.textsize(label, font=font)
        except Exception: text_w, text_h = 50, 18

        text_padding = 2
        label_bg_h = text_h + (text_padding * 2)
        label_bg_y0 = y0 - label_bg_h - text_padding # Position above box
        if label_bg_y0 < 0: label_bg_y0 = y1  # Adjust if off-screen

        draw.rectangle( # Label background
            [x0, label_bg_y0, x0 + text_w + text_padding * 2, label_bg_y0 + label_bg_h + text_padding * 2],
            fill=box_outline_rgba
        )
        text_fill = (0,0,0,255) if sum(base_rgb_color) > 382 else (255,255,255,255) # Black/white text
        draw.text((x0 + text_padding, label_bg_y0), label, fill=text_fill, font=font)

        # 5. Apply Mask Overlay (if available and box has area)
        if mask_pil and box_width_px > 0 and box_height_px > 0: # Second part of check is implicit from box validation
            try:
                # Process mask: convert to 'L', threshold, resize
                processed_mask = mask_pil.convert("L") if mask_pil.mode != "L" else mask_pil
                processed_mask = processed_mask.point(lambda p: 255 if p > 127 else 0, mode="L")
                if processed_mask.size != (box_width_px, box_height_px):
                    processed_mask = processed_mask.resize((box_width_px, box_height_px), Image.Resampling.NEAREST)

                # Create colored patch and apply using the mask
                color_patch = Image.new("RGBA", (box_width_px, box_height_px), mask_fill_rgba)
                # Create a temporary full-image overlay for this single mask
                temp_mask_overlay = Image.new("RGBA", viz_image.size, (0,0,0,0)) # Fully transparent
                temp_mask_overlay.paste(color_patch, (x0, y0), processed_mask)
                
                # Composite this mask's overlay onto the current visualization image
                viz_image = Image.alpha_composite(viz_image, temp_mask_overlay)
                draw = ImageDraw.Draw(viz_image)
            except Exception as e:
                pass # Silently skip problematic mask, box/label are already drawn
            
    return viz_image

def process_image_and_save(image_source: str, output_dir_base: str = "output", response_data : str | None=None, verbose : bool=False):
    if verbose:
        print(f"Processing image: {image_source}")
    original_image = load_image(image_source)
    if not original_image: return

    if response_data is None:
        if verbose:
            print("Requesting segmentations from API...")
        api_response_data = call_segmentation_api(original_image, verbose=verbose)
    else:
        if verbose:
            print("Loading saved response:", response_data)
        with open(response_data, "r") as f:
            api_response_data = json.load(f)

    if api_response_data is None: print("Failed to get API response. Exiting."); return
    if not api_response_data and isinstance(api_response_data, list):
        print("API returned no detections.")

    if verbose:
        print("Parsing API response...")
    parsed_detections = parse_api_response(api_response_data)

    if verbose:
        print("Generating visualization...")
    visualization_image = visualize_detections(original_image, parsed_detections)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    fname_base = "".join(c if c.isalnum() or c in (' ', '.', '_') else '_' for c in os.path.splitext(os.path.basename(image_source))[0])
    output_dir = os.path.join(output_dir_base, f"{fname_base.replace(' ', '_')}_{timestamp}")
    os.makedirs(output_dir, exist_ok=True)
    if verbose:
        print(f"Saving outputs to: {output_dir}")

    visualization_image.save(os.path.join(output_dir, "visualization.png"))
    if verbose:
        print(f"  Visualization saved.")
    with open(os.path.join(output_dir, "api_function_call_args.json"), "w") as f:
        json.dump(api_response_data, f, indent=2)
    if verbose:
        print(f"  API function call args saved.")
    
    savable_parsed = [{"label": d["label"], "box_2d": d["box_2d"], "mask_base64": d["mask_base64_original"]} for d in parsed_detections]
    with open(os.path.join(output_dir, "parsed_detections_with_masks.json"), "w") as f:
        json.dump(savable_parsed, f, indent=2)
    if verbose:
        print(f"  Parsed detections saved.")
        print("Processing complete.")

if __name__ == "__main__":
    IMAGE_URL = "https://chirpforbirds.com/wp-content/uploads/2021/03/bird-eating-insect-chirp.jpeg"
    #\
    # [
    #     "https://thumbs.dreamstime.com/b/exotic-bird-holds-exotic-insect-beak-wildlife-animals-exotic-bird-holds-exotic-insect-beak-121183530.jpg",
    #     "https://cdn.morningchores.com/wp-content/uploads/2020/08/purple-martin-800x533.jpg",
    #     "https://cdn.prod.website-files.com/623236d8ac23bb57bd352b40/62c5470fdb796e2c27471393_Pair_of_Merops_apiaster_feeding.jpeg",
    #     "https://www.rarebirdalert.co.uk/v2/Content/images/articles/2017-07-10Beeeater.jpg",
    #     "https://upload.wikimedia.org/wikipedia/commons/thumb/0/0f/Green_Figbird.jpg/640px-Green_Figbird.jpg",
    #     "https://nestboxlive.com/wp-content/uploads/2024/10/bee-eater-catching-a-bee.webp",
    #     "https://media.gettyimages.com/id/1159343281/video/streak-eared-bulbul-with-a-night-butterfly-in-its-mouth-over-bright-clear-green-nature.jpg?s=640x640&k=20&c=qPl0Ckq9Wnvx_y7g6hdJwgv1aQ7l8Z3Yu62oFWCq0W4=",
    #     "https://c02.purpledshub.com/uploads/sites/41/2018/08/iStock_3715752_LARGE-5f256a8.jpg"
    # ]
    STORED_RESPONSE = None
    # STORED_RESPONSE = "/home/asger/mini_trainer/examples/output/bee_eater_catching_a_bee_20250528_101428/api_function_call_args.json"
    VERBOSE = False

    if not GOOGLE_API_KEY or GOOGLE_API_KEY == "YOUR_GOOGLE_API_KEY_HERE":
        print("ATTENTION: Real API call selected, but GOOGLE_API_KEY is not configured. Exiting.")
        exit(1)
    print(f"Running with REAL Gemini API: {GEMINI_MODEL_NAME}.")
    if GOOGLE_API_KEY == "YOUR_GOOGLE_API_KEY_HERE":
            print("WARNING: GOOGLE_API_KEY is placeholder. This will fail.")

    try:
        _ = ImageFont.load_default(size=10)
        print("Pillow font rendering with size supported.")
    except Exception: print("Pillow font rendering test failed.")

    if isinstance(IMAGE_URL, str):
        process_image_and_save(IMAGE_URL, response_data=STORED_RESPONSE, verbose=VERBOSE)
    else:
        if STORED_RESPONSE is not None:
            raise NotImplementedError(f"Using stored response ({STORED_RESPONSE}) for multiple images is not implemented.")
        for im in TQDM(IMAGE_URL, desc="Predicting..."):
            process_image_and_save(im, verbose=VERBOSE)