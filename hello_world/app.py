import json
import os
import boto3

bedrock_runtime = boto3.client(
    service_name='bedrock-runtime',
    region_name=os.environ.get('AWS_REGION', 'us-east-1')
)

def lambda_handler(event, context):
    """
    AWS Lambda handler to invoke an Anthropic Claude model via Amazon Bedrock
    and return a complete, non-streaming JSON response.
    It internally handles the stream from Bedrock but aggregates the result.
    This version has robust prompt handling for direct or API Gateway invocations.
    """
    print(f"Received event: {json.dumps(event)}")

    
    prompt_text = 'Hello, How are you?'
    lambda_request_body = {} 

    if isinstance(event, dict) and 'prompt' in event:
        prompt_text = event.get('prompt', prompt_text)
        print(f"Found prompt in top-level event: '{prompt_text}'")
    elif 'body' in event and event['body'] is not None:
        print(f"Found 'body' in event. Type: {type(event['body'])}")
        try:
            if isinstance(event['body'], str):
                lambda_request_body = json.loads(event['body'])
            elif isinstance(event['body'], dict):
                lambda_request_body = event['body']
            else:
                print("Warning: event['body'] is present but not a string or dict.")
            
            if 'prompt' in lambda_request_body:
                prompt_text = lambda_request_body.get('prompt', prompt_text)
                print(f"Found prompt in event['body']: '{prompt_text}'")
            else:
                print("Warning: 'prompt' key not found in parsed event['body']. Using default or prior value.")
        except json.JSONDecodeError:
            print("Error: Invalid JSON in event['body']. Using default or prior prompt.")
            
        except Exception as e:
            print(f"Error parsing event['body']: {str(e)}. Using default or prior prompt.")
    else:
        print("Warning: 'prompt' not found in top-level event or in event['body']. Using default prompt.")

    claude_prompt = f"Human: {prompt_text} Assistant:"
    print(f"Final Claude prompt being used: \"{claude_prompt}\"")


    bedrock_request_body = {
        "prompt": claude_prompt,
        "max_tokens_to_sample": 2048,
        "temperature": 0.5,
        "top_k": 250,
        "top_p": 0.5,
        "stop_sequences": [],
        "anthropic_version": "bedrock-2023-05-31" 
    }

    try:
        encoded_bedrock_body = json.dumps(bedrock_request_body).encode('utf-8')
    except TypeError as e:
        print(f"Error encoding Bedrock request body to JSON: {str(e)}")
        return {
            'statusCode': 500,
            'headers': {'Content-Type': 'application/json'},
            'body': json.dumps({"error": f"Internal error preparing request for Bedrock: {str(e)}"})
        }

    model_id = 'anthropic.claude-3-5-sonnet-20240620-v1:0' 
    params = {
        'modelId': model_id,
        'contentType': 'application/json',
        'accept': 'application/json',
        'body': encoded_bedrock_body
    }

    print(f"Invoking Bedrock model '{model_id}' with request body: {bedrock_request_body}")

    full_completion_from_bedrock = []
    try:
        response = bedrock_runtime.invoke_model_with_response_stream(**params)
    except Exception as e:
        print(f"Error invoking Bedrock model: {str(e)}")
        return {
            'statusCode': 502, 
            'headers': {'Content-Type': 'application/json'},
            'body': json.dumps({"error": f"Failed to invoke Bedrock model: {str(e)}"})
        }

    stream = response.get('body')
    if stream:
        try:
            for event_chunk in stream:
                chunk_obj = event_chunk.get('chunk')
                if chunk_obj:
                    chunk_bytes = chunk_obj.get('bytes')
                    if chunk_bytes:
                        try:
                            chunk_data = json.loads(chunk_bytes.decode('utf-8'))
                            completion_part = chunk_data.get('completion', '')
                            if completion_part:
                                full_completion_from_bedrock.append(completion_part)
                        except (json.JSONDecodeError, UnicodeDecodeError) as e:
                            print(f"Error decoding JSON chunk: {str(e)}. Chunk (first 100 bytes): {chunk_bytes[:100]}...")
            
            final_completion_text = "".join(full_completion_from_bedrock)
            print(f"Successfully aggregated Bedrock response. Total length: {len(final_completion_text)} characters.")
            if len(final_completion_text) > 200:
                 print(f"Full completion snippet: {final_completion_text[:100]}...{final_completion_text[-100:]}")
            else:
                 print(f"Full completion (if short): {final_completion_text}")

            return {
                'statusCode': 200,
                'headers': {'Content-Type': 'application/json'},
                'body': json.dumps({
                    "completion": final_completion_text,
                    "model_id": model_id,
                    "prompt_used_by_claude": claude_prompt, 
                    "original_prompt_received": prompt_text, 
                    "message": "Successfully processed request."
                })
            }

        except Exception as e:
            print(f"Error processing Bedrock stream: {str(e)}")
            return {
                'statusCode': 500,
                'headers': {'Content-Type': 'application/json'},
                'body': json.dumps({"error": f"Error during Bedrock stream processing: {str(e)}"})
            }
        finally:
            if hasattr(stream, 'close') and callable(stream.close):
                stream.close()
                print("Bedrock stream explicitly closed.")
    else:
        print("Error: No stream in Bedrock response.")
        return {
            'statusCode': 502,
            'headers': {'Content-Type': 'application/json'},
            'body': json.dumps({"error": "Bedrock invocation succeeded but returned no stream."})
        }

    print("Lambda handler execution reached an unexpected end point.")
    return {
        'statusCode': 500,
        'headers': {'Content-Type': 'application/json'},
        'body': json.dumps({"error": "An unexpected server error occurred in the Lambda handler."})
    }
