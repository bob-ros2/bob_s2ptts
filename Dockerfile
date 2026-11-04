# Build context is this package directory (self-contained repo):
#   docker build -t bob-s2ptts:latest -f ros2_ws/src/bob_s2ptts/Dockerfile ros2_ws/src/bob_s2ptts
# or simply `docker compose up -d --build` from this directory.

ARG ROS_DISTRO=humble
ARG UBUNTU_VERSION=22.04

# --- STAGE 1: Builder ---
# No CUDA toolkit needed: torch/torchaudio ship as cu128 manylinux wheels and
# fish-speech compiles nothing against CUDA (unlike flash-attn in bob_q3tts).
FROM ubuntu:${UBUNTU_VERSION} AS builder

ARG ROS_DISTRO
ENV DEBIAN_FRONTEND=noninteractive
WORKDIR /app

# Optional mirrors (Sonatype Nexus). Left empty -> stock Ubuntu/PyPI sources.
ARG APT_UBUNTU_MIRROR=
ARG APT_SECURITY_MIRROR=
ARG PIP_INDEX_URL=
ARG PIP_TRUSTED_HOST=
RUN if [ -n "$APT_UBUNTU_MIRROR" ]; then \
      sed -i "s|http://archive.ubuntu.com/ubuntu/|${APT_UBUNTU_MIRROR}/|g; s|http://security.ubuntu.com/ubuntu/|${APT_SECURITY_MIRROR}/|g" /etc/apt/sources.list; \
    fi

# ROS 2 base + Python build tools + portaudio headers (pyaudio builds from source)
RUN apt-get update && apt-get install -y \
    software-properties-common curl gnupg2 lsb-release git \
    && curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key -o /usr/share/keyrings/ros-archive-keyring.gpg \
    && echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] http://packages.ros.org/ros2/ubuntu $(lsb_release -cs) main" | tee /etc/apt/sources.list.d/ros2.list > /dev/null \
    && apt-get update && apt-get install -y \
    python3-pip python3-venv python3-colcon-common-extensions \
    build-essential portaudio19-dev ffmpeg \
    ros-${ROS_DISTRO}-ros-base \
    && rm -rf /var/lib/apt/lists/*

RUN if [ -n "$PIP_INDEX_URL" ]; then \
      pip3 config set global.index-url "$PIP_INDEX_URL" && \
      pip3 config set global.trusted-host "$PIP_TRUSTED_HOST"; \
    fi && \
    pip3 install --no-cache-dir -U pip

# Fish Audio S2-Pro engine: fetched by the image itself, pinned for reproducibility.
# Pin the same commit as the host-side venv clone (host quickstart in README).
ARG FISH_SPEECH_REF=0f8a12e8a651062fdee9c33a894bdccd3a8513f1
RUN git clone https://github.com/fishaudio/fish-speech.git /app/fish-speech && \
    git -C /app/fish-speech checkout ${FISH_SPEECH_REF}
RUN pip3 install --no-cache-dir -e "/app/fish-speech[cu128]" \
        --extra-index-url https://download.pytorch.org/whl/cu128 && \
    pip3 install --no-cache-dir soundfile sounddevice

# Build the ROS 2 package
COPY . /app/ros2_ws/src/bob_s2ptts/
WORKDIR /app/ros2_ws
RUN . /opt/ros/humble/setup.sh && \
    colcon build --packages-select bob_s2ptts --cmake-args -DCMAKE_BUILD_TYPE=Release

# --- STAGE 2: Runtime ---
FROM ubuntu:${UBUNTU_VERSION}

ARG ROS_DISTRO=humble
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ARG USERNAME=ros
ARG USER_UID=1000
ARG USER_GID=1000
ARG APT_UBUNTU_MIRROR=
ARG APT_SECURITY_MIRROR=
RUN if [ -n "$APT_UBUNTU_MIRROR" ]; then \
      sed -i "s|http://archive.ubuntu.com/ubuntu/|${APT_UBUNTU_MIRROR}/|g; s|http://security.ubuntu.com/ubuntu/|${APT_SECURITY_MIRROR}/|g" /etc/apt/sources.list; \
    fi

RUN apt-get update && apt-get install -y \
    software-properties-common curl gnupg2 lsb-release \
    && curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key -o /usr/share/keyrings/ros-archive-keyring.gpg \
    && echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] http://packages.ros.org/ros2/ubuntu $(lsb_release -cs) main" | tee /etc/apt/sources.list.d/ros2.list > /dev/null \
    && apt-get update && apt-get install -y \
    ros-${ROS_DISTRO}-ros-base \
    python3-pip \
    libasound2 alsa-utils ffmpeg libportaudio2 sox libasound2-plugins libsox-fmt-all \
    libpulse0 libpulse-mainloop-glib0 libdbus-1-3 \
    && rm -rf /var/lib/apt/lists/*

RUN groupadd -g ${USER_GID} ${USERNAME} && \
    useradd -m -u ${USER_UID} -g ${USER_GID} -s /bin/bash ${USERNAME} && \
    usermod -aG audio,video ${USERNAME}

COPY --from=builder /usr/local /usr/local
# fish-speech is an editable install; the package source must exist at /app/fish-speech
COPY --from=builder /app/fish-speech /app/fish-speech
COPY --from=builder --chown=ros:ros /app/ros2_ws/install /app/ros2_ws/install

# Audio via PulseAudio with ALSA fallback (same as bob_q3tts)
RUN printf 'pcm.!default { type pulse fallback "sysdefault" }\nctl.!default { type pulse fallback "sysdefault" }\n' > /etc/asound.conf

ENV HOME=/home/${USERNAME}
ENV PATH=${HOME}/.local/bin:${PATH}
ENV LD_LIBRARY_PATH=/usr/local/lib/python3.10/dist-packages/torch/lib:/usr/local/lib/python3.10/dist-packages/nvidia/cuda_runtime/lib:/usr/local/lib/python3.10/dist-packages/nvidia/cuda_cupti/lib:${LD_LIBRARY_PATH}
ENV ROS_HOME=/tmp/ros

WORKDIR /app/ros2_ws
USER ${USERNAME}

ENTRYPOINT ["/bin/bash", "-c", "source /opt/ros/humble/setup.bash && source install/setup.bash && exec \"$@\"", "--"]
CMD ["ros2", "run", "bob_s2ptts", "tts"]
