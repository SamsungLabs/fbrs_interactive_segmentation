FROM nvidia/cuda:10.1-cudnn7-devel-ubuntu18.04

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
	    git \
	    curl \
        libglib2.0-0 \
        software-properties-common \
        python3.6-dev \
        python3-pip

WORKDIR /tmp

RUN pip3 install --upgrade pip
RUN pip3 install setuptools
RUN pip3 install matplotlib numpy pandas scipy tqdm pyyaml easydict scikit-image bridson Pillow
RUN pip3 install imgaug mxboard graphviz
RUN pip3 install git+https://github.com/albu/albumentations --no-deps
RUN pip3 install opencv-python-headless
RUN pip3 install Cython
RUN pip3 install gluoncv==0.5.0
RUN pip3 install mxnet-cu101==1.5.1.post0
RUN pip3 install scikit-learn

ENV TINI_VERSION v0.18.0
ADD https://github.com/krallin/tini/releases/download/${TINI_VERSION}/tini /usr/bin/tini
RUN chmod +x /usr/bin/tini
ENTRYPOINT [ "/usr/bin/tini", "--" ]
CMD [ "/bin/bash" ]
