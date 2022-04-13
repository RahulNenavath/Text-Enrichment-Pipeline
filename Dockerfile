FROM public.ecr.aws/lambda/python:3.8

COPY requirements.txt .

ENV VIRTUAL_ENV=/opt/venv
RUN python3 -m venv $VIRTUAL_ENV
ENV PATH="$VIRTUAL_ENV/bin:$PATH"

ENV PORT=9500

RUN apt-get update && apt-get -y install gcc git build-essential

RUN /var/lang/bin/python3.8 -m pip install --upgrade pip

RUN pip --no-cache-dir install -r requirements.txt --target "${LAMBDA_TASK_ROOT}"

RUN /var/lang/bin/python3.8 -m spacy download en_core_web_sm

COPY src/ .

COPY src/main.py ${LAMBDA_TASK_ROOT}

CMD ["app.handler"]