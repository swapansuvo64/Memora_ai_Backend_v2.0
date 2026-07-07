import json
import logging
import base64
import random
import boto3
from config.settings import settings

logger = logging.getLogger(__name__)

# Initialize Bedrock client
try:
    bedrock_client = boto3.client(
        service_name="bedrock-runtime",
        region_name=settings.BEDROCK_AWS_REGION,
        aws_access_key_id=settings.BEDROCK_AWS_ACCESS_KEY_ID,
        aws_secret_access_key=settings.BEDROCK_AWS_SECRET_ACCESS_KEY
    )
    logger.info("AWS Bedrock client initialized successfully")
except Exception as e:
    logger.error(f"Failed to initialize AWS Bedrock client: {e}")
    bedrock_client = None

def generate_scene_description(image_bytes: bytes, mime_type: str) -> str:
    """
    Sends the image to AWS Bedrock (Claude 3 Haiku) to generate a semantic scene description.
    """
    if bedrock_client is None:
        logger.warning("Bedrock client not initialized. Falling back to mock description.")
        return "Mock Description: A sunny afternoon in a cozy kitchen with wooden cabinets, coffee mugs, and warm light shining through the window."
        
    try:
        # Normalize mime type for Anthropic API
        media_type = mime_type if mime_type in ["image/jpeg", "image/png", "image/gif", "image/webp"] else "image/jpeg"
        
        image_base64 = base64.b64encode(image_bytes).decode("utf-8")
        
        body = json.dumps({
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": 500,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": media_type,
                                "data": image_base64
                            }
                        },
                        {
                            "type": "text",
                            "text": "Describe the setting, environment, objects, mood, and actions in this image. Keep it detailed but concise, suitable for semantic gallery searches. Do not name specific people, focus on the objects and environment."
                        }
                    ]
                }
            ]
        })
        
        # Use Claude 3 Haiku model defined in settings
        response = bedrock_client.invoke_model(
            modelId=settings.BEDROCK_MODEL_MAVERICK,
            contentType="application/json",
            accept="application/json",
            body=body
        )
        
        response_body = json.loads(response.get("body").read())
        description = response_body["content"][0]["text"]
        logger.info("Successfully generated scene description using Claude")
        return description.strip()
        
    except Exception as e:
        logger.error(f"AWS Bedrock Claude invoke error: {e}. Falling back to mock description.")
        return "Mock Description: Friends laughing together outdoors near green trees during sunset, holding drinks."

def generate_text_embedding(text: str) -> list[float]:
    """
    Generates text embeddings using Amazon Titan Text Embeddings.
    """
    if bedrock_client is None:
        logger.warning("Bedrock client not initialized. Generating random mock vector.")
        return [random.uniform(-0.1, 0.1) for _ in range(1536)] # Titan embeddings dimension is 1536
        
    try:
        body = json.dumps({
            "inputText": text
        })
        
        response = bedrock_client.invoke_model(
            modelId="amazon.titan-embed-text-v1",
            contentType="application/json",
            accept="application/json",
            body=body
        )
        
        response_body = json.loads(response.get("body").read())
        embedding = response_body["embedding"]
        logger.info("Successfully generated text embedding from Bedrock Titan")
        return embedding
        
    except Exception as e:
        logger.error(f"AWS Bedrock Titan embedding invoke error: {e}. Generating random mock vector.")
        return [random.uniform(-0.1, 0.1) for _ in range(1536)]
