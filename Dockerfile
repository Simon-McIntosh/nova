FROM python:3.9

RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir .['full']

CMD ["cat", "/etc/os-release"]
