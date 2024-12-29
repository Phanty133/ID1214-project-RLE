FROM nvidia/cuda:12.2.2-runtime-ubuntu22.04
ENV DEBIAN_FRONTEND=noninteractive

COPY --from=ghcr.io/astral-sh/uv:latest /uv /bin/uv

# Install necessary packages
RUN apt-get update && apt-get install -y \
    sudo gnupg2 libgl1 libglib2.0-0 libsm6 libxext6 g++ libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# RUN useradd -m ubuntu
# RUN usermod --move-home -d /home/user --login user ubuntu && groupmod --new-name user ubuntu
RUN useradd -m user
RUN echo "user ALL=(ALL) NOPASSWD:ALL" > /etc/sudoers.d/user

USER user
ENV WORKSPACE /workspace
WORKDIR ${WORKSPACE}
RUN mkdir -p ${WORKSPACE} && \
    sudo chown -R 1000:1000 ${WORKSPACE}

# COPY ./ /workspace
