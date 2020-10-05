FROM continuumio/miniconda3

COPY nova.yml /tmp/nova.yml 
RUN conda env create -f /tmp/nova.yml
SHELL ["conda", "run", "-n", "nova", "/bin/bash", "-c"]
ENTRYPOINT ["conda", "run", "-n", "nova", "python"]
CMD ["--version"]

