FROM public.ecr.aws/lambda/python:3.7

COPY requirements.txt .

RUN yum install -y gcc git

RUN /var/lang/bin/python3.7 -m pip install --upgrade pip

RUN pip install --upgrade -r requirements.txt --target "${LAMBDA_TASK_ROOT}"

RUN /var/lang/bin/python3.7 -m spacy download en_core_web_sm
# RUN /var/lang/bin/python3.7 -m spacy download en_core_web_trf

RUN /var/lang/bin/python3.7 -m nltk.downloader punkt

RUN cp -r /root/nltk_data /usr/local/share/nltk_data 

COPY src/ .

COPY src/app.py ${LAMBDA_TASK_ROOT}

CMD ["app.handler"]