FROM python:3.8

# Install required dependencies
RUN apt-get update && apt-get install -y \
    bison \
    build-essential \
    cmake \
    curl \
    flex \
    git \
    libbz2-dev \
    libboost-all-dev \
    libncurses-dev \
    libtinfo5 \
    libx11-dev \
    libxt-dev \
    ninja-build \
    wget \
    gedit \
    xorg-dev \
    && rm -rf /var/lib/apt/lists/*

# Create a working directory
WORKDIR /app

# Add Sublime Text's GPG key
RUN wget -qO - https://download.sublimetext.com/sublimehq-pub.gpg | apt-key add -

# Ensure apt is set up to work with https sources:
RUN apt-get update && apt-get install -y apt-transport-https


# Select the channel to use:
RUN echo "deb https://download.sublimetext.com/ apt/stable/" | tee /etc/apt/sources.list.d/sublime-text.list

# Update apt sources and install Sublime Text
RUN apt-get update && apt-get install -y sublime-text

# Install nle and minihack
RUN pip3 install matplotlib==3.5.0
RUN pip install --no-cache-dir nle minihack

# Set default command
CMD ["bash"]

