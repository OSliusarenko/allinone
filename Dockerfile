FROM ubuntu:16.04

RUN apt-get update && apt-get install -y python2.7 python-numpy python-scipy
RUN mkdir /calc

COPY *.py /calc/

ENTRYPOINT [ "cd", "/calc", "&&", "python" ]

CMD [ "CM dynamics.py" ]

