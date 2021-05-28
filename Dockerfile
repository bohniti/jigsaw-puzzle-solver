FROM rayproject/ray-ml


WORKDIR /usr/src/app

COPY . ./
RUN apt-get update && \
      apt-get -y install sudo
RUN useradd -m docker && echo "docker:docker" | chpasswd && adduser docker sudo
USER docker

RUN pip install --no-cache-dir -r requirements.txt
CMD /bin/bash