import boto3
import json
import settings

client = boto3.client(
        "sagemaker-runtime", 
        region_name="us-east-1", 
        aws_access_key_id=settings.data['id'],
        aws_secret_access_key= settings.data['key'])

def vectorize(sentence):
    endpoint_name = (
        "tensorflow-inference-2020-09-09-09-32-37-203"  # Your endpoint name.
    )
    content_type = (
        "application/json"  # The MIME type of the input data in the request body.
    )
    accept = "application/json"
    payload = sentence
    payload = json.dumps(payload)
    response = client.invoke_endpoint(
        EndpointName=endpoint_name,
        ContentType=content_type,
        Accept=accept,
        Body=payload,
    )
    res = json.loads((response["Body"].read().decode()))
    return res["predictions"][0]

def cosine_distance_with_tensors(s1, s2):

    from scipy.spatial import distance

    text_to_vector_v1 = vectorize(s1)
    text_to_vector_v2 = vectorize(s2)
    cosine = distance.cosine(text_to_vector_v1, text_to_vector_v2)
    return round((1 - cosine) * 100, 2)
