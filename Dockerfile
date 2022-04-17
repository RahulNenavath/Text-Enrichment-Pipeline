FROM public.ecr.aws/lambda/python:3.7

COPY requirements.txt .

ENV VIRTUAL_ENV=/opt/venv
RUN python3 -m venv $VIRTUAL_ENV
ENV PATH="$VIRTUAL_ENV/bin:$PATH"

ENV PORT=9500

RUN yum install -y gcc git

RUN /var/lang/bin/python3.7 -m pip install --upgrade pip

RUN pip install --upgrade -r requirements.txt --target "${LAMBDA_TASK_ROOT}"

RUN /var/lang/bin/python3.7 -m spacy download en_core_web_sm

COPY src/ .

COPY src/app.py ${LAMBDA_TASK_ROOT}

CMD ["app.handler"]