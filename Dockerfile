FROM python:2.7

RUN pip install scipy numpy
RUN mkdir /calc

COPY [ "*.py", "/calc/" ]
COPY startscript.sh .

VOLUME [ "/calc/out/" ]

ENTRYPOINT [ "./startscript.sh" ]
