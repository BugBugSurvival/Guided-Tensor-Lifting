
# 🚀 Guided Tensor Lifting

STAGG(Synthesis of Tensor Algebra Guided by Grammars) is a program lifting framework introduced in the **Guided Tensor Lifting** paper. It *automates the translation of low-level tensor computations* (e.g., C code) into *high-performance TACO tensor algebra expressions*. By combining *large language models (LLMs)* and *probabilistic grammars*, STAGG simplifies tensor DSL lifting making it easier to adopt optimized tensor computation frameworks while ensuring correctness and efficiency.

---

## **📝 Background & Motivation**
Tensor computations are **at the core of modern machine learning**. While **high-performance tensor DSLs (e.g., TACO)** enable efficient execution, **manually rewriting code** for DSLs is:
1. Time-consuming and complex
2. Prone to errors
3. Difficult to scale across different tensor programs

**STAGG** automates the conversion of low-level **C tensor programs** into **TACO expressions** by:
1. **Querying an LLM** to generate potential translations.
2. **Constructing a probabilistic grammar** to model candidate solutions.
3. **Performing a guided search** over the grammar space to synthesize valid TACO programs.
4. **Verifying correctness** using **input-output validation** and **bounded model checking**.

---

## **🖥️ System Requirements**
🚨 **Important:** 
- Your system must meet these requirements for Guided Tensor Lifting to function correctly.
- Building the Docker image will take a while (10 minutes to 1 hour), so why not grab a cup of coffee ☕ or enjoy a warm onsen ♨️ while it completes?
  
| Component   | Requirements |
|------------|-----------------|
| **🛠️ Operating System** | Ubuntu 22.04 (LTS) or 20.04 (LTS) |
| **📦 CMake Version** | 3.3 |
| **🐍 Python Version** | 3.10 |
| **💾 Disk Space** | 30 GB available space |
| **🧠 Memory** | 16 GB RAM |



## **🐳 Using Docker (Recommended)**
Using Docker ensures a **reproducible**, **isolated**, and **hassle-free setup**.

### **1️⃣ Install Docker**
If you haven’t installed Docker, follow the official guide:  
👉 [Install Docker Engine on Ubuntu](https://docs.docker.com/engine/install/ubuntu/#install-using-the-repository)

### **2️⃣ Build the Docker Image**
> **⚠️ Note:** Ensure the local `Guided-Tensor-Lifting` project directory contains **only** the code from the GitHub repository every time you build the Docker image.

Then, navigate to the project directory 📂`Guided-Tensor-Lifting` and run:

```bash
docker build -t stagg .
```

### **3️⃣ Run the Docker Container**
To start an interactive session:

```bash
docker run -it --rm stagg
```

### **4️⃣ Execute the Tool**

Once inside the Docker Container, run STAGG with the desired arguments:

```bash
python3 run.py --grammar_style <arg1> --benchmark <arg2> --timeout <arg3> --enumerator <arg4> --check_llm <arg5>
```
##### **📌 Example Execution**
```bash
python3 run.py --grammar_style wcfg --benchmark len --timeout 3600 --enumerator top_down --check_llm false
```
🎯 **Results** are stored in 📂 `Guided-Tensor-Lifting/lifting/data/lifting_logs/`. 


#### **📌 Command-Line Arguments**
##### **📝 `--grammar_style <arg1>`**
Defines the grammar to be used:
- `wcfg`         - Weighted Context-Free Grammar with diffused weights.
- `wcfg_equal_p` - Weighted Context-Free Grammar with equal probabilities.
- `full_grammar` - Contains all production rules.
- `original`     - Only includes LLM-generated production rules.

##### **📝 `--benchmark <arg2>`**
- Name of the benchmark located in `lifting/data/benchmarks/`.

##### **📝 `--timeout <arg3>`**
- Sets the timeout duration (in seconds).

##### **📝 `--enumerator <arg4>`**
Enumeration strategy:
- `top_down`
- `bottom_up`

##### **📝 `--check_llm <arg5>`**
Check LLM solutions:
- `true`  - Validate LLM solutions only. Though `--enumerator <arg4>` is required, it will not be used.
- `false` - Directly enumerate solutions base on the pCFG.

### **⚡  Running Script**

To run a specific approach, navigate to 📂 `Guided-Tensor-Lifting/lifting` use:
```sh
./run_td_wcfg.sh  # Example: Running top-down enumerator with a WCFG that has diffused weights
```

📜 List of Execution Scripts under 📂 `Guided-Tensor-Lifting/lifting`:
- **`run.sh`** -  Runs the following `.sh` scripts.
- **`run_bu_equal.sh`** - Runs the bottom-up enumerator with a WCFG that has equal probabilities.
- **`run_bu_full.sh`**  -  Runs the bottom-up enumerator with a WCFG that includes all production rules.
- **`run_bu_orig.sh`**  -  Runs the bottom-up enumerator using only LLM-generated production rules.
- **`run_bu_wcfg.sh`**  -  Runs the bottom-up enumerator with a WCFG that has diffused weights.
- **`run_td_equal.sh`** -  Runs the top-down enumerator with a WCFG that has equal probabilities.
- **`run_td_full.sh`**  -  Runs the top-down enumerator with a WCFG that includes all production rules.
- **`run_td_orig.sh`**  -  Runs the top-down enumerator using only LLM-generated production rules.
- **`run_td_wcfg.sh`**  -  Runs the top-down enumerator with a WCFG that has diffused weights.


---

## **🔧 Manual Installation (Alternative)**
If you prefer a manual setup, follow these steps:

### **1️⃣ Set Environment Variables**
Before running the tool, export the following paths(modify the paths according to your system):

```bash
export PYTHONPATH=<PATH-TO-Guided-Tensor-Lifting>/taco/build/lib:$PYTHONPATH
export PYTHONPATH=$PYTHONPATH:<PATH-TO-Guided-Tensor-Lifting>/cbmc-validation/build/python_packages/synth
export PATH=$PATH:<PATH-TO-Guided-Tensor-Lifting>/cbmc-validation/deps/cvc5/build/bin:<PATH-TO-Guided-Tensor-Lifting>/cbmc-validation/deps/cbmc/build/bin
export LD_LIBRARY_PATH=$<PATH-TO-Guided-Tensor-Lifting>/llvm/lib:$LD_LIBRARY_PATH
```

### **2️⃣ Install CMake**
Install **CMake 3.3**:

### **3️⃣ Set Up Python 3.10 Virtual Environment**
Inside the project directory 📂 `Guided-Tensor-Lifting`, create and activate a virtual environment for dependency isolation:

```bash
python3.10 -m venv ./.venv/venv
source ./.venv/venv/bin/activate
pip install -r requirements.txt
```

> **⚠️ Note:** Perform all remaining steps within the activated `venv` environment.

### **4️⃣ Install LLVM 14.0.0**
Inside the project directory 📂 `Guided-Tensor-Lifting`, download and install **LLVM 14.0.0**:

```bash
wget https://github.com/llvm/llvm-project/releases/download/llvmorg-14.0.0/clang+llvm-14.0.0-x86_64-linux-gnu-ubuntu-18.04.tar.xz
tar -xf clang+llvm-14.0.0-x86_64-linux-gnu-ubuntu-18.04.tar.xz
mv clang+llvm-14.0.0-x86_64-linux-gnu-ubuntu-18.04 llvm
rm clang+llvm-14.0.0-x86_64-linux-gnu-ubuntu-18.04.tar.xz
```

### **5️⃣ Install TACO**
Inside the project directory 📂 `Guided-Tensor-Lifting`, clone and build the **TACO Tensor Compiler**:

```bash
git clone https://github.com/tensor-compiler/taco
cd taco && mkdir -p build && cd build
cmake -DCMAKE_BUILD_TYPE=Release -DPYTHON=ON .. && make -j$(nproc)
make install
```

### **6️⃣ Build Code Analyses**
Navigate to 📂 `Guided-Tensor-Lifting/lifting` and run:

```bash
bash ./build_code_analyses.sh <PATH-TO-Guided-Tensor-Lifting>/llvm
```

### **7️⃣ Build CBMC Validation**
Ensure dependencies are installed:

```bash
sudo apt install ninja-build bison flex libtool jq lld libgmp3-dev build-essential libboost-all-dev default-jdk maven -y
```
```bash
chmod +x /home/Guided-Tensor-Lifting/cbmc-validation/deps/cvc5/build/bin/cvc5
```
Navigate to 📂 `Guided-Tensor-Lifting/cbmc-validation`, build dependencies:

```bash
bash ./build_tools/build_dependencies.sh
bash ./build_tools/build_mlirSynth.sh
```
After that, install the Python dependencies needed by MLIR:
```bash
python -m pip install -r deps/llvm-project/mlir/python/requirements.txt
pip install clang==17.0.6
pip install libclang==17.0.6
```

---

## **🚀 Running STAGG**



### **Execute the Tool**
Navigate to 📂 `Guided-Tensor-Lifting/lifting` and run:

```bash
source ../.venv/venv/bin/activate
python3 run.py --grammar_style <arg1> --benchmark <arg2> --timeout <arg3> --enumerator <arg4> --check_llm <arg5>
```
#### **📌 Example Execution**
```bash
source ../.venv/venv/bin/activate
python3 run.py --grammar_style wcfg --benchmark len --timeout 3600 --enumerator top_down --check_llm false
```

🎯 **Results** are stored in 📂 `Guided-Tensor-Lifting/lifting/data/lifting_logs/`. 


#### **📌 Command-Line Arguments**
##### **📝 `--grammar_style <arg1>`**
Defines the grammar to be used:
- `wcfg`         - Weighted Context-Free Grammar with diffused weights.
- `wcfg_equal_p` - Weighted Context-Free Grammar with equal probabilities.
- `full_grammar` - Contains all production rules.
- `original`     - Only includes LLM-generated production rules.

##### **📝 `--benchmark <arg2>`**
- Name of the benchmark located in `lifting/data/benchmarks/`.

##### **📝 `--timeout <arg3>`**
- Sets the timeout duration (in seconds).

##### **📝 `--enumerator <arg4>`**
Enumeration strategy:
- `top_down`
- `bottom_up`

##### **📝 `--check_llm <arg5>`**
Check LLM solutions:
- `true`  - Validate LLM solutions only. Though `--enumerator <arg4>` is required, it will not be used.
- `false` - Directly enumerate solutions base on the pCFG.

### **⚡ Running Script**

To run a specific approach, navigate to 📂 `Guided-Tensor-Lifting/lifting` use:
```sh
./run_td_wcfg.sh  # Example: Running top-down enumerator with a WCFG that has diffused weights
```

📜 List of Execution Scripts under 📂 `Guided-Tensor-Lifting/lifting`:
- **`run.sh`** -  Runs the following `.sh` scripts.
- **`run_bu_equal.sh`** - Runs the bottom-up enumerator with a WCFG that has equal probabilities.
- **`run_bu_full.sh`**  -  Runs the bottom-up enumerator with a WCFG that includes all production rules.
- **`run_bu_orig.sh`**  -  Runs the bottom-up enumerator using only LLM-generated production rules.
- **`run_bu_wcfg.sh`**  -  Runs the bottom-up enumerator with a WCFG that has diffused weights.
- **`run_td_equal.sh`** -  Runs the top-down enumerator with a WCFG that has equal probabilities.
- **`run_td_full.sh`**  -  Runs the top-down enumerator with a WCFG that includes all production rules.
- **`run_td_orig.sh`**  -  Runs the top-down enumerator using only LLM-generated production rules.
- **`run_td_wcfg.sh`**  -  Runs the top-down enumerator with a WCFG that has diffused weights.



---

## **✅ Why Use STAGG?**
- 🔥 **Automates tensor DSL lifting**: No manual rewriting required!
- 📈 **Combines LLMs & heuristics**: Uses probabilistic grammars for accurate translations.
- 🚀 **Scales efficiently**: Reduces enumeration search space significantly.
- 🔍 **Ensures correctness**: Uses **bounded model checking** for verification.




