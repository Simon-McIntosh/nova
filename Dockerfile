FROM python:3.9

RUN pip install --no-cache-dir --upgrade pip 
RUN pip install --no-cache-dir git+ssh://git@git.iter.org/eq/nova.git['full']

CMD ["cat", "/etc/os-release"]
