FROM rayproject/ray
USER root

WORKDIR /usr/src/app

COPY . ./

RUN pip install --no-cache-dir -r requirements.txt
CMD /bin/bash