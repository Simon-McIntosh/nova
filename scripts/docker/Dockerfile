FROM python:3.9

RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

CMD ["cat", "/etc/os-release"]
