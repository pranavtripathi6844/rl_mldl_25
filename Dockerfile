FROM ubuntu:20.04

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y \
    python3.8 \
    python3.8-dev \
    python3-pip \
    git \
    wget \
    unzip \
    gcc \
    g++ \
    libosmesa6-dev \
    libgl1-mesa-glx \
    libglfw3 \
    patchelf \
    && rm -rf /var/lib/apt/lists/*

RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.8 1
RUN update-alternatives --install /usr/bin/pip pip /usr/bin/pip3 1

WORKDIR /opt
RUN wget https://mujoco.org/download/mujoco210-linux-x86_64.tar.gz \
    && tar -xf mujoco210-linux-x86_64.tar.gz \
    && rm mujoco210-linux-x86_64.tar.gz

ENV MUJOCO_PY_MUJOCO_PATH=/opt/mujoco210
ENV LD_LIBRARY_PATH=/opt/mujoco210/bin:${LD_LIBRARY_PATH}

WORKDIR /workspace
COPY . .

RUN pip install --upgrade pip==22.* setuptools==65.5.0 wheel==0.38
RUN pip install -r requirements.txt
RUN pip install mujoco-py==2.1.2.14

CMD ["/bin/bash"]

