import json
import logging
import random
import vertexai
from vertexai.generative_models import GenerativeModel, Part
from vertexai.language_models import TextEmbeddingModel
from config.settings import settings

logger = logging.getLogger(__name__)

# Initialize Vertex AI
try:
    vertexai.init(project=settings.GCP_PROJECT_ID, location="us-central1")
    logger.info(f"Google Vertex AI initialized successfully with project {settings.GCP_PROJECT_ID}")
except Exception as e:
    logger.error(f"Failed to initialize Google Vertex AI client: {e}")

def fallback_bedrock_scene_description(image_bytes: bytes, mime_type: str, prompt_text: str) -> dict:
    try:
        import boto3
        logger.info("Attempting AWS Bedrock fallback for scene description using Claude 3.5 Sonnet...")
        
        session = boto3.Session(
            aws_access_key_id=settings.BEDROCK_AWS_ACCESS_KEY_ID,
            aws_secret_access_key=settings.BEDROCK_AWS_SECRET_ACCESS_KEY,
            region_name=settings.BEDROCK_AWS_REGION
        )
        client = session.client("bedrock-runtime")
        
        img_format = mime_type.split("/")[-1].lower()
        if img_format not in ["jpeg", "png", "gif", "webp"]:
            img_format = "jpeg"
            
        response = client.converse(
            modelId=settings.BEDROCK_MODEL_SAFETY, # Claude 3.5 Sonnet
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "image": {
                                "format": img_format,
                                "source": {"bytes": image_bytes}
                            }
                        },
                        {
                            "text": prompt_text
                        }
                    ]
                }
            ],
            inferenceConfig={"temperature": 0.2, "maxTokens": 4000}
        )
        raw_text = response['output']['message']['content'][0]['text'].strip()
        parsed = json.loads(raw_text)
        logger.info("Successfully generated scene description using AWS Bedrock (Claude 3.5 Sonnet)")
        return parsed
    except Exception as e:
        logger.error(f"AWS Bedrock scene description fallback failed: {e}")
        raise e

def fallback_bedrock_chat_search_decision(prompt_text: str) -> dict:
    try:
        import boto3
        logger.info("Attempting AWS Bedrock fallback for chat search decision using Claude 3 Haiku...")
        
        session = boto3.Session(
            aws_access_key_id=settings.BEDROCK_AWS_ACCESS_KEY_ID,
            aws_secret_access_key=settings.BEDROCK_AWS_SECRET_ACCESS_KEY,
            region_name=settings.BEDROCK_AWS_REGION
        )
        client = session.client("bedrock-runtime")
        
        response = client.converse(
            modelId=settings.BEDROCK_MODEL_MAVERICK, # Claude 3 Haiku
            messages=[
                {
                    "role": "user",
                    "content": [{"text": prompt_text}]
                }
            ],
            inferenceConfig={"temperature": 0.2, "maxTokens": 2000}
        )
        raw_text = response['output']['message']['content'][0]['text'].strip()
        parsed = json.loads(raw_text)
        logger.info("Successfully generated chat search decision using AWS Bedrock")
        return parsed
    except Exception as e:
        logger.error(f"AWS Bedrock chat search decision fallback failed: {e}")
        raise e

def generate_scene_description(image_bytes: bytes, mime_type: str, labeled_faces: list = None) -> dict:
    """
    Sends the image to Google Vertex AI (Gemini) to generate:
    1. A semantic paragraph description optimized for embedding-based search
    2. Structured tags for exact-match filtering
    3. Category and details (document, landscape, etc.)
    4. Bounding boxes of humans with unclear faces
    """
    try:
        media_type = mime_type if mime_type in ["image/jpeg", "image/png", "image/gif", "image/webp"] else "image/jpeg"
        image_part = Part.from_data(data=image_bytes, mime_type=media_type)

        prompt_text = (
            "Analyze this image and respond with ONLY a valid JSON object. "
            "The JSON must have exactly six keys: \"description\", \"tags\", \"category\", \"document_details\", \"landscape_details\", and \"detected_humans\".\n\n"
            
            "\"description\": a single cohesive paragraph (no bullet points, no headers) written for semantic "
            "image search. Naturally weave in all of the following, without labeling them as sections:\n"
            "1. Environment & setting — indoor/outdoor, specific location type (kitchen, beach, park, city street, bedroom)\n"
            "2. Lighting & time of day — natural sunlight, golden hour, dim warm light, night, neon, shadows\n"
            "3. Actions & events — what is happening (eating, walking, playing, cooking, running) and the emotional state or activities of the people (e.g. funny, sad, happy, active, sleepy, smiling, laughing, crying, posing).\n"
            "4. People & appearance — approximate count, estimated age range, clothing color/style, expressions/state (e.g. funny face, sad look, wide awake and active, sleepy/tired eyes), glasses. "
            "Identify the specific named individuals listed in the IDENTITY CONTEXT below if provided. Otherwise, do not invent names.\n"
            "5. Objects & background — furniture, plants, structures, animals, distinctive or unique items (landmarks, "
            "instruments, vehicles)\n"
            "6. Mood & atmosphere — cozy, calm, festive, melancholy, energetic\n"
            "7. Colors & palette — dominant colors of clothing, scenery, or key objects\n"
            "8. Occasion or event type — birthday, wedding, graduation, casual day, holiday, if identifiable; otherwise omit\n"
            "9. Group size — solo, pair, small group, large gathering\n"
            "10. Season & weather — sunny, rainy, snowy, autumn leaves, if identifiable; otherwise omit\n"
            "11. Social context — general relationship type visible (family gathering, friends, couple, colleagues) "
            "described generically, using names/relations from IDENTITY CONTEXT if available.\n\n"
        )
        
        if labeled_faces:
            people_parts = []
            has_self = False
            for f in labeled_faces:
                name = f["name"]
                rel = f.get("relationship", "")
                if rel:
                    rel_clean = rel.lower().strip()
                    if rel_clean in ["my self", "myself", "me", "self"]:
                        people_parts.append(f"{name} (who is the user himself/herself, i.e., 'me')")
                        has_self = True
                    else:
                        people_parts.append(f"{name} (who is the user's {rel})")
                else:
                    people_parts.append(name)
                    
            people_context = ", ".join(people_parts)
            prompt_text += (
                f"IDENTITY CONTEXT:\n"
                f"The user has identified the following specific individuals in this photo: {people_context}.\n"
            )
            if has_self:
                prompt_text += (
                    f"IMPORTANT: One of the individuals in the photo is 'me' (the user themselves). "
                    f"You should write the description in a personalized way or first-person where appropriate. "
                    f"You can refer to this person as 'me', 'myself', or 'I' (e.g., 'I stand on the balcony' or "
                    f"'This is a photo of me, Suvadeep, standing solo on an outdoor balcony').\n"
                )
            prompt_text += (
                f"You MUST refer to these individuals by their names and relationships where appropriate in the \"description\" "
                f"instead of generic terms (e.g. write 'her husband John' or 'John' instead of 'a man').\n\n"
            )
        
        prompt_text += (
            "\"tags\": a JSON object with these fields (use null if not identifiable, use lowercase strings, "
            "arrays for multi-value fields):\n"
            "{\n"
            '  "setting": string,\n'
            '  "indoor_outdoor": "indoor" | "outdoor",\n'
            '  "lighting": string,\n'
            '  "time_of_day": string,\n'
            '  "activity": string,\n'
            '  "people_states": [string], -- e.g. ["funny", "sad", "happy", "active", "sleepy", "smiling", "laughing"]\n'
            '  "people_count": integer,\n'
            '  "mood": string,\n'
            '  "dominant_colors": [string],\n'
            '  "occasion": string,\n'
            '  "season_weather": string,\n'
            '  "objects": [string]\n'
            "}\n\n"

            "\"category\": string, must be exactly one of: \"document\", \"landscape\", \"people\", \"object_animal\", or \"other\".\n"
            "Select \"document\" for receipt, passport, book/page, ID, certificate, bill, note, or text screenshot. "
            "Select \"landscape\" for landscape, nature setting, street, city street, travel views. "
            "Select \"people\" if humans are the main focus (even if backward or blurry). "
            "Select \"object_animal\" for pets, animals, or prominent objects.\n\n"

            "\"document_details\": if category is \"document\", provide a JSON object with:\n"
            "{\n"
            '  "document_type": string, -- receipt, passport, invoice, certificate, note, screenshot, etc.\n'
            '  "extracted_title": string, -- a helpful suggested title for this document (e.g. "Tax Form 2025")\n'
            '  "extracted_text": string -- key visible text, dates, numbers, or summary of document text\n'
            "}\n"
            "If category is not \"document\", return null.\n\n"

            "\"landscape_details\": if category is \"landscape\", provide a JSON object with:\n"
            "{\n"
            '  "location": string, -- suggested location name or setting description (e.g. "Eiffel Tower", "beach")\n'
            '  "scenery_type": string -- mountain, cityscape, beach, forest, sunset, street, etc.\n'
            "}\n"
            "If category is not \"landscape\", return null.\n\n"

            "\"detected_humans\": a JSON array of objects representing all humans visible in the photo. "
            "Include those whose face is unclear, standing backward, or side-profile. For each human, specify:\n"
            "{\n"
            '  "description": string, -- brief visual helper (e.g., "man in blue shirt seen from behind")\n'
            '  "box": [ymin, xmin, ymax, xmax] -- 4 integers representing bounding box normalized on 0 to 1000 scale (0,0 is top-left, 1000,1000 is bottom-right). Focus on the head and upper body if possible.\n'
            "}\n"
            "If no humans are visible, return an empty array [].\n"
        )

        model = GenerativeModel("gemini-2.5-flash")
        
        response = model.generate_content(
            contents=[image_part, prompt_text],
            generation_config={
                "response_mime_type": "application/json",
                "temperature": 0.2
            }
        )

        raw_text = response.text.strip()
        parsed = json.loads(raw_text)

        logger.info("Successfully generated structured scene description using Gemini")
        return {
            "description": parsed.get("description", "").strip(),
            "tags": parsed.get("tags", {}),
            "category": parsed.get("category", "other"),
            "document_details": parsed.get("document_details"),
            "landscape_details": parsed.get("landscape_details"),
            "detected_humans": parsed.get("detected_humans", [])
        }

    except Exception as gemini_err:
        logger.error(f"Vertex AI Gemini invoke error: {gemini_err}. Trying AWS Bedrock fallback...")
        try:
            parsed = fallback_bedrock_scene_description(image_bytes, mime_type, prompt_text)
            return {
                "description": parsed.get("description", "").strip(),
                "tags": parsed.get("tags", {}),
                "category": parsed.get("category", "other"),
                "document_details": parsed.get("document_details"),
                "landscape_details": parsed.get("landscape_details"),
                "detected_humans": parsed.get("detected_humans", [])
            }
        except Exception as bedrock_err:
            logger.error(f"AWS Bedrock fallback also failed for scene description: {bedrock_err}. Falling back to mock description.")
            return {
                "description": "Mock Description: Friends laughing together outdoors near green trees during sunset, holding drinks.",
                "tags": {},
                "category": "other",
                "document_details": None,
                "landscape_details": None,
                "detected_humans": []
            }


def generate_text_embedding(text: str) -> list[float]:
    """
    Generates text embeddings using Vertex AI text-embedding-004.
    Outputs a 768-dimensional vector.
    """
    try:
        model = TextEmbeddingModel.from_pretrained("text-embedding-004")
        embeddings = model.get_embeddings([text])
        embedding = embeddings[0].values
        logger.info("Successfully generated text embedding from Vertex AI text-embedding-004")
        return embedding
    except Exception as e:
        logger.error(f"Vertex AI text embedding invoke error: {e}. Generating random mock vector.")
        return [random.uniform(-0.1, 0.1) for _ in range(768)]


def generate_chat_search_decision(query: str, candidates: list, history: list = None, relationships: str = "") -> dict:
    """
    Sends the user query, chat history, and candidate image metadata to Gemini.
    Gemini decides:
    1. Which candidate images are strictly appropriate for the user query.
    2. A friendly, natural chat response describing the findings, or asking for more clues if nothing is found.
    Returns: {"response_text": str, "appropriate_image_ids": list[str]}
    """
    try:
        # Format the chat history for Gemini context
        history_text = ""
        if history:
            history_text = "\n".join([f"{msg.get('role', 'user')}: {msg.get('text', '')}" for msg in history])

        # Format candidates for Gemini
        candidates_str = json.dumps(candidates, indent=2)

        prompt_text = (
            "You are a helpful AI memory assistant for a personal photo gallery app called MemoraAI.\n"
            "Your task is to analyze the user's search query, look at the candidate images retrieved from the vector database, "
            "determine which images (if any) are appropriate matches, and generate a natural conversational chat response.\n\n"
            "CRITICAL RULES:\n"
            "1. ONLY select an image if its description or tags match the query or context. Do not guess or hallucinate if the details do not match.\n"
            "2. If you find matching images, write a friendly response summarizing them (e.g. 'I found 2 photos of your brother John at the beach'). "
            "You should refer to people by their names and relationships (e.g. John who is your Brother) if known.\n"
            "3. If no images match, or if the user's query is a generic question, greeting, or follow-up that cannot be satisfied by the candidates, "
            "you MUST ask for clarification or more clues (e.g. 'I couldn't find any photos of that. Can you tell me more? Where was this photo taken, or who was with you?').\n"
            "4. Respond with ONLY a valid JSON object matching the schema below. Do not include markdown code block formatting in the raw text response, just clean JSON.\n\n"
            "IDENTITY RELATIONSHIPS CONTEXT:\n"
            f"{relationships}\n\n"
            "CHAT HISTORY:\n"
            f"{history_text}\n\n"
            f"USER QUERY: \"{query}\"\n\n"
            "CANDIDATE IMAGES:\n"
            f"{candidates_str}\n\n"
            "JSON RESPONSE SCHEMA:\n"
            "{\n"
            '  "response_text": "Friendly response text here",\n'
            '  "appropriate_image_ids": ["image_uuid_1", "image_uuid_2"]\n'
            "}\n"
        )

        model = GenerativeModel("gemini-2.5-flash")
        
        response = model.generate_content(
            contents=prompt_text,
            generation_config={
                "response_mime_type": "application/json",
                "temperature": 0.2
            }
        )

        raw_text = response.text.strip()
        parsed = json.loads(raw_text)
        return {
            "response_text": parsed.get("response_text", "").strip(),
            "appropriate_image_ids": parsed.get("appropriate_image_ids", [])
        }
    except Exception as gemini_err:
        logger.error(f"Failed to query Gemini for chat search decision: {gemini_err}. Trying AWS Bedrock fallback...")
        try:
            parsed = fallback_bedrock_chat_search_decision(prompt_text)
            return {
                "response_text": parsed.get("response_text", "").strip(),
                "appropriate_image_ids": parsed.get("appropriate_image_ids", [])
            }
        except Exception as bedrock_err:
            logger.error(f"AWS Bedrock fallback also failed for chat search decision: {bedrock_err}")
            return {
                "response_text": "I'm sorry, I encountered an error while processing your request. Can you tell me more about what you're looking for?",
                "appropriate_image_ids": [],
                "error": True
            }

