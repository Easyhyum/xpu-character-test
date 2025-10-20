# FROM nvcr.io/nvidia/l4t-ml:r36.2.0-py3
FROM nvcr.io/nvidia/pytorch:25.08-py3

WORKDIR /workspace
RUN rm -rf *

RUN apt-get update
RUN apt-get install -y tmux
RUN python3 -m pip install --upgrade pip

# Copy requirements first for better caching
COPY requirements.txt /workspace/

COPY run.py /workspace/
COPY api/ /workspace/api/

COPY hyperparameter.json /workspace/
COPY run.sh /workspace/

RUN pip3 install -r requirements.txt

RUN chmod +x /workspace/run.sh
RUN chmod +x /workspace/git_init.sh

# 인터랙티브 bash 쉘로 시작
CMD ["/bin/bash"]