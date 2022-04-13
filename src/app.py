import json
import spacy
import logging
import traceback
from pydantic import BaseModel
from enrichment_pipeline import TextEnrichmentPipeline

logging.basicConfig(level=logging.INFO)
logging.info(f"\nLoaded Spacy Version: {spacy.__version__}\nLoading Spacy en_core_web_sm Model")
spacy_sm = spacy.load("en_core_web_sm")

with open('stopwords.txt', 'r') as f:
    stopwords = f.read().split("\n")

enrichment_pipeline = TextEnrichmentPipeline(spacy_model=spacy_sm, stop_words=stopwords)

def handler(event, context):

    if event['rawPath'] == '/':
        return {
            "statusCode": 200,
            "headers": {
                "Content-Type": "application/json"
            },
            "body": json.dumps({
                "Service": "Text Enrichment API",
                "Status": "Active"
            })
        }

    elif event['rawPath'] == '/ping':
        return {
            "statusCode": 200,
            "headers": {
                "Content-Type": "application/json"
            },
            "body": json.dumps({
                "Service": "Text Enrichment API",
                "Status": "Active",
                "Ping": "Success"
            })
        }

    elif event['rawPath'] == '/text-process':

        request_body = json.loads(event['body'])
        request_text = str(request_body['text'])

        try:
            enriched_text = enrichment_pipeline.enrich(text=request_text)

            return {
                "statusCode": 200,
                "headers": {
                    "Content-Type": "application/json"
                },
                "body": json.dumps({
                    "enriched_text": enriched_text,
                })
            }
        except Exception as e:
            return {
                "statusCode": 200,
                "headers": {
                    "Content-Type": "application/json"
                },
                "body": json.dumps({
                    "Error": str(traceback.format_exc),
                    "Exception": str(e)
                })
            }
    else:
        return {
            "statusCode": 200,
            "headers": {
                "Content-Type": "application/json"
            },
            "body": json.dumps({
                "Service": "Text Enrichment API",
                "Status": "Active",
                "Message": "API method not allowed"
            })
        }
