FROM python:3.9
WORKDIR /usr/src/app
COPY . .
RUN pip install --no-cache-dir --upgrade pip 
RUN pip install --no-cache-dir .

CMD ["cat", "/etc/os-release"]
