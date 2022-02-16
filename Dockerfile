FROM python:3.9
WORKDIR /usr/src/app
COPY setup.py setup.cfg pyproject.toml versioner.py ./
COPY nova ./nova
RUN pip install --no-cache-dir --upgrade pip wheel
RUN pip install --no-cache-dir .

CMD ["cat", "/etc/os-release"]
