FROM pytorch/pytorch:1.8.0-cuda11.1-cudnn8-runtime

RUN pip install jupyterlab
RUN pip install aquirdturtle_collapsible_headings
RUN pip install ipywidgets
RUN jupyter nbextension enable --py widgetsnbextension

RUN export SHELL=/bin/bash  

COPY requirements.txt .

RUN pip install -r requirements.txt
