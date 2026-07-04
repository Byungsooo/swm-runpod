FROM runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV STABLEWM_HOME=/workspace/stablewm_home
ENV MUJOCO_GL=egl

# Base packages
RUN apt-get update && apt-get install -y \
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
    cron \
    && rm -rf /var/lib/apt/lists/*

# Install libegl1 (GLVND EGL dispatch) for headless MuJoCo rendering.
# The version in the base image is a stub; we force-install the real one.
# --force-depends is safe here: the only missing dep (libegl-mesa0) is an
# optional software-rendering fallback we don't need on GPU pods.
RUN wget -q "http://archive.ubuntu.com/ubuntu/pool/main/libg/libglvnd/libegl1_1.4.0-1_amd64.deb" -O /tmp/libegl1.deb \
    && dpkg -i --force-depends /tmp/libegl1.deb \
    && rm /tmp/libegl1.deb \
    && ldconfig

# pip upgrade
RUN pip install --upgrade pip

# Pin torch/torchvision to a CUDA 12.x build compatible with the driver
# available on RunPod's nodes (PyPI's default resolves newer CUDA-13 wheels
# some node drivers can't run yet). Installed before stable-worldmodel[all]
# so that install sees a satisfying version already present and doesn't
# upgrade it to the unpinned CUDA-13 default.
RUN pip install --index-url https://download.pytorch.org/whl/cu126 \
    'torch==2.12.1' 'torchvision==0.27.1'

# stable-worldmodel (PyPI release)
# Source code will be cloned separately to /workspace for editable development
RUN pip install 'stable-worldmodel[all]'

# torchaudio is a leftover from the base runpod/pytorch image, not an actual
# project dependency, and ends up version-mismatched against the torch
# pinned above (previously broke transformers.AutoModel.from_pretrained).
RUN pip uninstall -y torchaudio

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

# Convenient alias to run first-time setup script
RUN echo "alias swm-setup='cd /workspace && curl -s https://raw.githubusercontent.com/Byungsooo/swm-runpod/main/setup.sh | bash'" >> /root/.bashrc

# Default git identity (override later with --global if needed)
RUN git config --global user.email "hamalove@gmail.com" && \
    git config --global user.name "Byungsooo"
 
# SSH config for VS Code Remote
RUN mkdir -p /var/run/sshd && \
    echo "PermitRootLogin yes" >> /etc/ssh/sshd_config && \
    echo "PasswordAuthentication no" >> /etc/ssh/sshd_config

WORKDIR /workspace

EXPOSE 22 8888

COPY start.sh /start.sh
COPY notify.sh /usr/local/bin/notify.sh
RUN chmod +x /start.sh /usr/local/bin/notify.sh

CMD ["/start.sh"]
