## Context set to be used on .devcontianer is not this folder
## Context is the root folder for the project, that way 
## The follwing only store in a context that can be used to other folders
FROM alpine as storage
WORKDIR /app
COPY requirements.txt ./



FROM quay.io/jupyter/minimal-notebook as devcontainer
# FROM ubuntu:22.04 as devcontainer
ENV DEBIAN_FRONTEND noninteractive
USER root
#Install base tools
RUN set -eux; \
  apt-get update -qq; \
  apt-get full-upgrade -yqq; \
  apt-get install -yq \
    curl \  
    git \
    wget \
    unzip \
    zip \
    make \
    build-essential \
    libssl-dev \
    libffi-dev \
    libpq-dev \
    sudo \
    pkg-config \
    jq  \
    vim \
    zsh \
    netcat \
    graphviz \
    tmux; \
  apt-get autoremove -yqq; apt-get clean; 

# Install node and db dependnecies

USER ${USER}
RUN set -ex;\
  wget https://raw.githubusercontent.com/ohmyzsh/ohmyzsh/master/tools/install.sh -O - | zsh || true; \
  echo 'export PATH="$HOME/.local/bin:$PATH"' >> ~/.zshrc; \
  sed -i -e "s|plugins=.*|plugins=(git poetry kubectl terraform aws azure python jupyter)|g" ~/.zshrc


COPY --from=storage --chown=jovyan:jovyan /app/* ./
RUN set -eux; \
  ls -la /home/**/** ; sleep 15; \
  pip3 install -r requirements.txt;


# ENV PYTHONPATH=/home/${USER}/projects
# WORKDIR /home/${USER}/projects
EXPOSE 8000
USER root