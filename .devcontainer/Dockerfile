# Copyright (c) Jupyter Development Team.
# Distributed under the terms of the Modified BSD License.
ARG  OWNER=jupyter
ARG  BASE_CONTAINER=$OWNER/scipy-notebook
FROM $BASE_CONTAINER

LABEL maintainer="Jupyter Project <jupyter@googlegroups.com>"

SHELL ["/bin/bash", "-o", "pipefail", "-c"]

USER root

WORKDIR /tmp

RUN  mkdir -p /home/jovyan/.jupyter
COPY .devcontainer/jupyter_lab_config.py /home/jovyan/.jupyter/jupyter_lab_config.py
COPY .devcontainer/jupyter_server_config.py /home/jovyan/.jupyter/jupyter_server_config.py
COPY .devcontainer/jupyter_notebook_config.py /home/jovyan/.jupyter/jupyter_notebook_config.py

USER ${NB_UID}

WORKDIR "${HOME}"