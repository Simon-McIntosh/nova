FROM python:3.9
WORKDIR /usr/src/app
COPY setup.py setup.cfg pyproject.toml versioneer.py ./
COPY nova ./nova
COPY tests ./tests
COPY .git ./.git
RUN pip install --no-cache-dir --upgrade pip wheel
RUN pip install --no-cache-dir .
RUN rm -rf ./.git
CMD ["cat", "/etc/os-release"]
