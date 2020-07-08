FROM tensorflow/tensorflow:nightly

LABEL author="Jose Ayerdis"

ENV INSTALL_PATH /tensorflowproject

RUN mkdir -p $INSTALL_PATH
WORKDIR $INSTALL_PATH

# Solving dependencies
COPY requirements.txt $INSTALL_PATH
RUN mkdir /uploads
RUN ls . > /uploads/filename.txt
RUN pip install -r $INSTALL_PATH/requirements.txt


COPY model/ model/
COPY src/ src/
COPY data/ data/

#CMD ["python", "src/intro.py", "-OPTIONAL_FLAG"]
CMD [ "python","src/webapp/app.py" ]
EXPOSE 5000/tcp