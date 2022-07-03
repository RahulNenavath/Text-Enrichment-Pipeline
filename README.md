# Text-Enrichment-Pipeline
Text Enrichment Pipeline is an information extraction pipeline which extracts Key-terms, Entities, and Text Statistics. 

## Tech Stack:
Python, Spacy3, NLTK, Textacy, PKE, Textstats, Docker, AWS Lambda, AWS ECR, GitHub Actions (CI/CD) 

## Deployment:
This project has been deployed as an AWS Lambda function using a container image from the AWS ECR service and made available. AWS Lambda functions are cost effective solutions for personal projects that don't have a lot of network traffic. GitHub Actions is used for CI/CD pipelines where every git push triggers the pipeline and the updated docker container is pushed to AWS ECR.

## Try It Out:
The Front-end Interface is made using streamlit and deployed on HuggingFace Spaces: 
https://huggingface.co/spaces/rahulNenavath305/Text-Enrichment-Pipeline
