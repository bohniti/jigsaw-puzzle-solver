FROM rayproject/ray:latest"$GPU"

WORKDIR /usr/src/app

COPY . ./

RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt
CMD /bin/bash