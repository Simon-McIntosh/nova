FROM continuumio/miniconda3

COPY nova.yml /tmp/nova.yml 
RUN conda env create -f /tmp/nova.yml
RUN echo "source activate nova" > ~/.bashrc
ENV PATH /opt/conda/envs/env/bin:$PATH

