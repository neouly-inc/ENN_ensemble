FROM pytorch/pytorch:1.6.0-cuda10.1-cudnn7-runtime
LABEL maintainer "Neouly"

ENV CN_HOME=/workspace
ENV DEBIAN_FRONTEND noninteractive


RUN pip install scipy==1.7.3 scikit-learn==1.0.2 matplotlib==3.5.3 seaborn==0.12.2


CMD ["/bin/bash"]