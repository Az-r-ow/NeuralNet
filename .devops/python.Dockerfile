FROM python:3.10-bullseye

COPY . /app

WORKDIR /app

RUN git submodule init && git submodule update

# Install cmake (necessary for building shared object)
RUN apt-get update && \
    apt-get install -y cmake libgl1 && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

RUN chmod +x ./scripts/build_without_tests.sh 

RUN ./scripts/build_without_tests.sh

RUN cd examples/train-predict-MNIST && pip install -r requirements.txt

CMD ["/bin/bash"]