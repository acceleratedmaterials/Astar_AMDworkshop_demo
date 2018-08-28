# AMD Workshop

---

## Installation

1. Download and install the [Anaconda python distribution](https://www.anaconda.com/download/), (python3 version). The installation instructions are found [here](https://conda.io/docs/user-guide/install/).

2. Clone demo repository
    * Windows
        * Launch `Anaconda Prompt`
        * Type `git clone https://github.com/acceleratedmaterials/AMDworkshop_demo.git` and press enter
            * If the above command fails with error message `'git' is not recognized ...`, you need to      install `git`. One option is to use `conda` to install `git`: issue `conda install -c anaconda git` in the prompt. Alternatively, check [here](https://git-scm.com/download/win) on installation instructions for `git` on Windows)
        * Type `cd AMDworkshop_demo` and press enter

    * MacOS, Linux
        * Launch a `terminal`
        * Type `git clone https://github.com/acceleratedmaterials/AMDworkshop_demo.git` and press enter
        * Type `cd AMDworkshop_demo` and press enter

3. Install additional required packages
    * **Note for MacOS/Linux machines**: Check you are indeed using the anaconda binaries by typing `which python` and pressing enter. You should see outputs with something similar to `/User/<your name>/anaconda3/bin/python` (For linux machines, this will be `/home/...` instead of `/User/...`).
    **If you see this, no further action is required**. If instead, you get something like `/usr/bin/python` (i.e. without the anaconda3), you are using your system's python binaries. In this case, please add `/User/<your name>/anaconda3/bin` to your `$PATH` environment variable ([Guide](http://osxdaily.com/2014/08/14/add-new-path-to-path-command-line/)) and then proceed to the next step.

    * In your opened (Windows) anaconda prompt (MacOS, Linux) terminal, type `pip install -r requirements.txt` and press enter. This should install the packages listed in the [requirements file](requirements.txt).

## Getting started

1. Launch the `Anaconda Navigator` program

2. **For demos using `Jupyter notebook`**
    * Launch `Jupyter notebook` ![alt text](https://github.com/acceleratedmaterials/AMDworkshop_demo/blob/master/demo_pics/Jupyter.png)
    * Navigate to the `AMDworkshop_demo` folder ![alt text](https://github.com/acceleratedmaterials/AMDworkshop_demo/blob/master/demo_pics/2.png)
    * Navigate to the corresponding subfolders and click on `<demo name>.ipynb` to begin interacting with the demo

3. **For demos using `Spyder`**
    * Launch `Spyder` ![alt text](https://github.com/acceleratedmaterials/AMDworkshop_demo/blob/master/demo_pics/Spyder.png)
    * Navigate to the `AMDworkshop_demo` folder ![alt text](https://github.com/acceleratedmaterials/AMDworkshop_demo/blob/master/demo_pics/4.png)
    * Navigate to the corresponding subfolders Run the `<demo name>.py` files as required from `Spyder` IDE ([Guide](https://pythonhosted.org/spyder/))

## License

This project is licensed under the [MIT](LICENSE.md) license.
