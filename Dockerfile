FROM ubuntu:focal

# setup timezone
RUN echo 'Etc/UTC' > /etc/timezone && \
    ln -s /usr/share/zoneinfo/Etc/UTC /etc/localtime && \
    apt-get update && \
    apt-get install -q -y --no-install-recommends tzdata && \
    rm -rf /var/lib/apt/lists/*

# Install packages
RUN apt-get update && \
    apt-get install -y --no-install-recommends apt-utils nano git curl python3-pip ffmpeg libsm6 libxext6 && \
    rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip3 install -r requirements.txt
RUN rm -rf requirements.txt && \
    rm -r ~/.cache/pip

RUN apt-get update && apt-get install -y cmake
RUN apt-get install build-essential wget libgtk2.0-dev pkg-config libavcodec-dev libavformat-dev libswscale-dev  -y 
RUN apt-get install g++ unzip -y
RUN apt-get install python-dev python-numpy libtbb2 libtbb-dev libjpeg-dev libpng-dev libtiff-dev libdc1394-22-dev -y

# opencv 
RUN wget https://github.com/opencv/opencv/archive/refs/heads/4.x.zip
RUN unzip 4.x.zip
WORKDIR /opencv-4.x
RUN mkdir build
WORKDIR /opencv-4.x/build
RUN cmake -D CMAKE_BUILD_TYPE=Release -D CMAKE_INSTALL_PREFIX=/usr/local ..
RUN make -j4 
RUN make install
RUN pip3 install ipdb

WORKDIR /unsemlabag
RUN mkdir samples
RUN mkdir data


