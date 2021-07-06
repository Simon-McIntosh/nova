From conda/miniconda3-centos7:latest

COPY fenicsx.yml /tmp/fenicsx.yml 
#RUN conda config --set ssl_verify no
RUN conda env create -f /tmp/fenicsx.yml

SHELL ["conda", "run", "-n", "fenicsx", "/bin/bash", "-c"]
ENTRYPOINT ["conda", "run", "-n", "fenicsx", "spyder"]

