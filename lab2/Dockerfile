FROM amytabb/docker_ubuntu16_essentials
RUN apt-get install -y libomp-dev
COPY main.cpp .
WORKDIR .
RUN g++ -std=c++11 -fopenmp -o main main.cpp
CMD ["./main"]