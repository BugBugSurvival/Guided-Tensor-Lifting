# ------------------------------------------------------------------------------
# Dockerfile for building Guided Tensor Lifting on Ubuntu 22.04 with Python 3.10
# Uses deadsnakes PPA for Python 3.10
# Builds TACO with Python bindings enabled
# Builds code analyses + CBMC validation
# 
# This Dockerfile sets up a complete environment for developing and testing
# the Guided Tensor Lifting toolchain, including dependencies like LLVM, Python,
# CBMC, and TACO. It ensures a reproducible and isolated build for development 
# and experimentation.
# ------------------------------------------------------------------------------

FROM ubuntu:22.04

# ------------------------------------------------------------------------------
# 1) Configure Environment Variables
# ------------------------------------------------------------------------------
# Set timezone and avoid interactive prompts during package installations
ENV TZ="Europe/London"
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone
ENV DEBIAN_FRONTEND=noninteractive

# ------------------------------------------------------------------------------
# 2) Set Environment Variables for the Tool
# ------------------------------------------------------------------------------
# Ensure the required libraries and executables are available in the PATH and
# PYTHONPATH for runtime execution.
# Add LLVM binaries to PATH for easier access
ENV PATH="/home/Guided-Tensor-Lifting/llvm/bin:$PATH"
ENV PYTHONPATH="/home/Guided-Tensor-Lifting/taco/build/lib:/home/Guided-Tensor-Lifting/cbmc-validation/build/python_packages/synth"
ENV PATH="/home/Guided-Tensor-Lifting/cbmc-validation/deps/cvc5/build/bin:/home/Guided-Tensor-Lifting/cbmc-validation/deps/cbmc/build/bin:${PATH}"
ENV LD_LIBRARY_PATH="/usr/local/lib:/usr/local/lib:/home/Guided-Tensor-Lifting/llvm/lib"

# ------------------------------------------------------------------------------
# 3) Install Base System Prerequisites
# ------------------------------------------------------------------------------
# Install essential development tools and libraries required for compilation
# and execution of Guided Tensor Lifting.
RUN apt-get update && apt-get install -y --no-install-recommends \
    software-properties-common \
    gnupg2 \
    dirmngr \
    ca-certificates \
    cmake \
    wget \
    tar \
    git \
    build-essential \
    ninja-build \
    bison \
    flex \
    libtool \
    jq \
    lld \
    libgmp3-dev \
    libssl-dev \
    libboost-all-dev \
    default-jdk \
    maven \
    && rm -rf /var/lib/apt/lists/*  # Cleanup to reduce image size

# ------------------------------------------------------------------------------
# 4) Add deadsnakes PPA & Install Python 3.10
# ------------------------------------------------------------------------------
# Python 3.10 is required for the project, and we use deadsnakes PPA for an 
# up-to-date version. 
RUN add-apt-repository ppa:deadsnakes/ppa && apt-get update && apt-get install -y \
    python3.10 \
    python3.10-dev \
    python3-pip \
    && rm -rf /var/lib/apt/lists/*  # Cleanup

# ------------------------------------------------------------------------------
# 5) Set Up Project Directory and Copy Code
# ------------------------------------------------------------------------------
# Define the working directory and copy all project files into the container.
WORKDIR /home/Guided-Tensor-Lifting
COPY . /home/Guided-Tensor-Lifting

# ------------------------------------------------------------------------------
# 6) Install Python Dependencies
# ------------------------------------------------------------------------------
RUN pip install --upgrade pip && \
    pip install --ignore-installed -r requirements.txt

# ------------------------------------------------------------------------------
# 7) Install LLVM 14.0.0
# ------------------------------------------------------------------------------
# LLVM is required for compilation and code analysis.
# The specific version 14.0.0 is downloaded and extracted for compatibility.
WORKDIR /home/Guided-Tensor-Lifting
RUN wget https://github.com/llvm/llvm-project/releases/download/llvmorg-14.0.0/clang+llvm-14.0.0-x86_64-linux-gnu-ubuntu-18.04.tar.xz && \
    tar -xf clang+llvm-14.0.0-x86_64-linux-gnu-ubuntu-18.04.tar.xz && \
    mv clang+llvm-14.0.0-x86_64-linux-gnu-ubuntu-18.04 llvm && \
    rm clang+llvm-14.0.0-x86_64-linux-gnu-ubuntu-18.04.tar.xz

# ------------------------------------------------------------------------------
# 8) Build TACO (Tensor Algebra Compiler)
# ------------------------------------------------------------------------------
# Clone and build TACO with Python bindings enabled.
WORKDIR /home/Guided-Tensor-Lifting
RUN git clone https://github.com/tensor-compiler/taco

WORKDIR /home/Guided-Tensor-Lifting/taco
RUN mkdir -p build && cd build && \
    cmake -DCMAKE_BUILD_TYPE=Release -DPYTHON=ON .. && \
    make -j"$(nproc)" && \
    make install

# ------------------------------------------------------------------------------
# 9) Build Code Analyses (lifting)
# ------------------------------------------------------------------------------
# Run the build script for tensor lifting code analyses.
WORKDIR /home/Guided-Tensor-Lifting/lifting
RUN bash ./build_code_analyses.sh /home/Guided-Tensor-Lifting/llvm

# ------------------------------------------------------------------------------
# 10) Build CBMC Validation (Using Ephemeral Git Identity)
# ------------------------------------------------------------------------------
# The CBMC validation process requires dependencies to be built.
# An anonymous git identity is used for ephemeral commits.
WORKDIR /home/Guided-Tensor-Lifting/cbmc-validation
RUN chmod +x /home/Guided-Tensor-Lifting/cbmc-validation/deps/cvc5/build/bin/cvc5
RUN export GIT_AUTHOR_NAME="Anonymous" && \
    export GIT_AUTHOR_EMAIL="anonymous@example.com" && \
    export GIT_COMMITTER_NAME="Anonymous" && \
    export GIT_COMMITTER_EMAIL="anonymous@example.com" && \
    bash ./build_tools/build_dependencies.sh && \
    bash ./build_tools/build_mlirSynth.sh
WORKDIR /home/Guided-Tensor-Lifting
RUN pip install -r cbmc-validation/deps/llvm-project/mlir/python/requirements.txt
RUN pip install clang==17.0.6 && \
    pip install libclang==17.0.6
    
# ------------------------------------------------------------------------------
# 11) Set Default Working Directory and Activate Python Virtual Environment
# ------------------------------------------------------------------------------
# Set the default working directory to the project root for convenience.
# Activate the Python virtual environment by default.
WORKDIR /home/Guided-Tensor-Lifting/lifting

# ------------------------------------------------------------------------------
# 12) Default Command: Open a Shell
# ------------------------------------------------------------------------------
# The container runs in an interactive shell by default to allow development
# and debugging.
CMD ["/bin/bash"]

