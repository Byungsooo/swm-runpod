FROM nvidia/cuda:12.1.1-cudnn8-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV STABLEWM_HOME=/workspace/stablewm_home

# Base packages
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3.10-dev \
    python3-pip \
    git \
    curl \
    wget \
    vim \
    tmux \
    htop \
    unzip \
    ca-certificates \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    swig \
    openssh-server \
    && rm -rf /var/lib/apt/lists/*

# Symlinks
RUN ln -sf /usr/bin/python3.10 /usr/bin/python && \
    ln -sf /usr/bin/pip3 /usr/bin/pip

# pip upgrade
RUN pip install --upgrade pip

# PyTorch (CUDA 12.1)
RUN pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# stable-worldmodel (PyPI release)
# Source code will be cloned separately to /workspace for editable development
RUN pip install 'stable-worldmodel[all]'

# Dev tools
RUN pip install \
    jupyter \
    jupyterlab \
    ipykernel \
    matplotlib \
    seaborn \
    pandas \
    numpy \
    wandb \
    boto3 \
    awscli \
    tqdm \
    rich

# tmux config
RUN echo "set -g mouse on" >> /root/.tmux.conf && \
    echo "set -g history-limit 10000" >> /root/.tmux.conf && \
    echo "set -g status-right '#S'" >> /root/.tmux.conf

# SSH config for VS Code Remote
RUN mkdir -p /var/run/sshd && \
    echo "PermitRootLogin yes" >> /etc/ssh/sshd_config && \
    echo "PasswordAuthentication no" >> /etc/ssh/sshd_config

WORKDIR /workspace

EXPOSE 22 8888

CMD ["/bin/bash"]
