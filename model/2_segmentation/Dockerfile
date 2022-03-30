FROM nvidia/cuda:11.6.0-devel-ubuntu20.04
RUN apt-get update
RUN apt-get -y install wget
RUN mkdir -p ~/miniconda3
RUN wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda3/miniconda.sh
RUN bash ~/miniconda3/miniconda.sh -b -u -p ~/miniconda3
RUN rm -rf ~/miniconda3/miniconda.sh
RUN ~/miniconda3/bin/conda init bash
RUN ~/miniconda3/bin/conda init zsh
RUN bash ~/.bashrc
RUN apt-get -y install freeglut3-dev
ENV DEBIAN_FRONTEND noninteractive
ENV DEBCONF_NONINTERACTIVE_SEEN true
RUN apt-get -y install libgtk2.0-dev
ADD . /seg
WORKDIR /seg
RUN bash ~/.bashrc && /root/miniconda3/condabin/conda env create -f environment.yaml
RUN /root/miniconda3/condabin/conda run -n schp pip install -r requirements.txt
RUN /root/miniconda3/condabin/conda run -n schp pip install torch==1.7.1+cu110 torchvision==0.8.2+cu110 -f https://download.pytorch.org/whl/torch_stable.html
ENTRYPOINT ["/root/miniconda3/condabin/conda", "run", "--no-capture-output", "-n", "schp", "python", "app.py"]
