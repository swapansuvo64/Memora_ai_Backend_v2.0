import os
import boto3
from dotenv import load_dotenv

def test_bedrock_connection():
    # Load .env file from the current directory
    env_path = os.path.join(os.path.dirname(__file__), ".env")
    if os.path.exists(env_path):
        print(f"Loading environment variables from: {env_path}")
        load_dotenv(env_path)
    else:
        print("Warning: .env file not found in current directory. Falling back to default env variables.")

    # Fetch variables
    region = os.getenv("BEDROCK_AWS_REGION", "us-east-1")
    access_key = os.getenv("BEDROCK_AWS_ACCESS_KEY_ID")
    secret_key = os.getenv("BEDROCK_AWS_SECRET_ACCESS_KEY")
    model_id = os.getenv("BEDROCK_MODEL_MAVERICK", "us.anthropic.claude-3-haiku-20240307-v1:0")

    print(f"\n--- Checking AWS Bedrock Configuration ---")
    print(f"AWS Region: {region}")
    print(f"Access Key ID: {access_key}")
    print(f"Secret Access Key: {'***' + secret_key[-4:] if secret_key else None}")
    print(f"Model ID: {model_id}")
    print(f"-----------------------------------------\n")

    if not access_key:
        print("❌ Error: Missing AWS credentials in environment. Please check BEDROCK_AWS_ACCESS_KEY_ID.")
        return

    is_absk = access_key.startswith("ABSK")

    try:
        if is_absk:
            print("Detected ABSK Bearer Token. Testing REST API connection to Titan Embeddings...")
            import requests
            url = f"https://bedrock-runtime.{region}.amazonaws.com/model/amazon.titan-embed-text-v1/invoke"
            headers = {
                "Authorization": f"Bearer {access_key}",
                "Content-Type": "application/json"
            }
            body = json.dumps({"inputText": "hello"})
            res = requests.post(url, headers=headers, data=body)
            res.raise_for_status()
            response_body = res.json()
            embedding = response_body["embedding"]
            print("✅ Success: Successfully authenticated and generated text embedding using ABSK API key!")
            print(f"Embedding length: {len(embedding)} dimensions (Sample: {embedding[:3]})")
        else:
            # Initialize client
            client = boto3.client(
                service_name="bedrock",
                region_name=region,
                aws_access_key_id=access_key,
                aws_secret_access_key=secret_key
            )
            
            # Test 1: List models
            print("Testing: Listing foundation models...")
            models = client.list_foundation_models()
            print("✅ Success: Successfully connected to AWS Bedrock API!")
            
            # Check if the requested model is listed / available
            available_models = [m['modelId'] for m in models['modelSummaries']]
            if model_id in available_models:
                print(f"✅ Success: Model '{model_id}' is available in your region.")
            else:
                print(f"⚠️ Warning: Model '{model_id}' was not found in listing. You might need to request access or verify modelId.")
            
    except Exception as e:
        print("❌ Connection Failed!")
        print(f"Error details: {e}")

if __name__ == "__main__":
    import json
    test_bedrock_connection()
