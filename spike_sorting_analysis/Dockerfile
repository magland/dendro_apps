FROM python

RUN mkdir /src

# Install pynwb
RUN pip install pynwb

# Install spikeinterface
RUN pip install spikeinterface==0.99.1

# Install dendro
RUN pip install dendro==0.1.23

# Copy files into the container
RUN mkdir /app
COPY spike_sorting_analysis/*.py /app/
COPY spike_sorting_analysis/helpers /app/helpers
COPY spike_sorting_analysis/common /app/common
