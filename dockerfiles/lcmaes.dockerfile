# syntax=docker/dockerfile:1
# author: Jiri Tumpach 

FROM ubuntu:focal

##########################
# set up timezone
ENV TZ=Europe/Prague
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

#########################
# adding nameserver
RUN echo "nameserver 8.8.8.8" >> /etc/resolv.conf

##########################
# remove no-cache (speed up the build)
RUN rm /etc/apt/apt.conf.d/docker-clean

##########################
# download packages faster

RUN echo "deb [ allow-insecure=yes trusted=yes ] http://ppa.launchpad.net/apt-fast/stable/ubuntu focal main" >> /etc/apt/sources.list
RUN echo "deb-src [ allow-insecure=yes trusted=yes ] http://ppa.launchpad.net/apt-fast/stable/ubuntu focal main" >> /etc/apt/sources.list

# alternative (secure) way:
# RUN apt-get update -y && \
#	  apt-get install -y --no-install-recommends software-properties-common && \
# 	add-apt-repository -y ppa:apt-fast/stable 

RUN apt-get update 
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get install -y --no-install-recommends apt-fast <&-
ENV DEBIAN_FRONTEND=

RUN echo debconf apt-fast/maxdownloads string 8 | debconf-set-selections
RUN echo debconf apt-fast/dlflag boolean true | debconf-set-selections
RUN echo debconf apt-fast/aptmanager string apt-get | debconf-set-selections

##########################
# update ubuntu & install recommends
RUN apt-fast update -y && \
	  apt-fast upgrade -y --no-install-recommends && \
		apt-fast install -y --no-install-recommends \
			vim cmake make git tzdata apt-transport-https ca-certificates \
		  autoconf automake libtool libgoogle-glog-dev libgflags-dev libeigen3-dev \
			gcc g++ \
			python3.8-dev python3-numpy-dev python3-pip \
			libboost-tools-dev libboost-python-dev libboost-numpy-dev


##########################
# update certificates
RUN update-ca-certificates

##########################
# clone libcmaes
RUN git clone https://github.com/CMA-ES/libcmaes.git
WORKDIR libcmaes

##########################
# install libcmaes
RUN mkdir build
WORKDIR build
RUN cmake .. -DCMAKE_INSTALL_PREFIX=/usr/ -DLIBCMAES_BUILD_PYTHON=ON
RUN make -j16
RUN make install
RUN cp python/lcmaes.so /lib/python3/dist-packages/
WORKDIR ..
RUN rm -rf build
WORKDIR /

RUN ln -s /usr/bin/python3 /usr/bin/python


##########################
# install common python scripts

RUN apt-fast install -y --no-install-recommends \
			python3-scipy python3-sklearn python3-pandas python3-venv \
			python3-redis python3-h5py python3-numpy 

##########################
# make virtual environment volume

RUN touch /entrypoint_docker.sh
RUN echo "#!/bin/bash\n\
if [ -d /python_virtual_environment ]; then\n\
	if [ -e /python_virtual_environment/bin/activate ]; then\n\
		echo Running python virtual environment\n\
		source /python_virtual_environment/bin/activate\n\
	else \n\
		echo Creating and running python virtual environment ... \n\
		python3 -m venv --system-site-packages /python_virtual_environment\n\
		source /python_virtual_environment/bin/activate\n\
		python3 -m pip install --upgrade pip\n\
	fi\n\
else\n\
	echo Warning no virtual environment is in use \(no disk detected\) \n\
fi\n\
	bash \n\
" > /entrypoint_docker.sh

RUN chmod +x /entrypoint_docker.sh
ENTRYPOINT ["/entrypoint_docker.sh"]

