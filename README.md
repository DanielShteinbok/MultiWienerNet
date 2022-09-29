# MultiWienerNet
## Deep learning for fast spatially-varying deconvolution 
This is a modified clone of the project by the Waller Lab, find the original [here](https://github.com/Waller-Lab/MultiWienerNet).
### Inscopix Instructions:
connect to `lambda-quad` over ssh. Then, run:
```
curl https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -o Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh
```

You may then need to press `ENTER` a whole bunch of times, and type `yes` to agree to their terms.

try running
```shell
conda
```
and if you get an error saying that the command isn't found, you may have to run
```
./miniconda3/bin/conda init
```
After this, you can close the terminal window, open a new one and ssh again as before.
Now, if you run
```bash
conda
```
you should get the Anaconda usage instructions.

Next, you want to create your environment. Run:
```bash
conda env create -f tensorflow/reproduce.yml
conda activate tf2
```
Now, to start up a jupyter notebook, run:
```bash
jupyter-notebook --no-browser --port=8888
```
Copy the link is printed by jupyter (it should start as `https://localhost:8888`...)

*NOTE:* if the number is not 8888 above, you should change the 8888 in the instructions below to match.

Open up a new terminal window, and run:
```shell
ssh -L 8888:localhost:8888 <username>@<machine name>
```
and enter password etc. to connect and build an ssh tunnel. 

Open up a browser on the host machine and paste in the URL you copied above.

### [Project Page](https://waller-lab.github.io/MultiWienerNet/) | [Paper](https://doi.org/10.1364/OPTICA.442438)

### Setup:
Clone this project using:
```
git clone https://github.com/Waller-Lab/MultiWienerNet.git
```

Install dependencies. We provide code both in Tensorflow and in Pytorch. Tensorflow version only contains an implementation for 2D deconvolutions, whereas the Pytorch contains both 2D and 3D deconvolutions. 

If using Pytorch, install the depencies as:

```
conda env create -f environment.yml
source activate multiwiener_pytorch
```

If using Tensorflow, install the depencies as:

```
conda env create -f environment.yml
source activate multiwiener_tf
```

## Using pre-trained models
We provide an example of a pre-trained MutliWienerNet for fast 2D deconvolutions as well as compressive 3D deconvolutions from 2D measurements. These examples are based on data for Randoscope3D. To adapt this model to your own data, please see below. 


### Loading in pretrained models
The pre-trained models can be downloaded: [here (pytorch)](https://drive.google.com/drive/folders/1teIPp2q2ce0l9FjYe0LuC9c-Rpq2fA8x?usp=sharing) and [here (tensorflow)](https://drive.google.com/drive/folders/1E3bye75ovDvfKsDG4IMe_hzo5wQU1zTP?usp=sharing) 
Please download these and place them in the pytorch/saved_models and tensorflow/saved_models

### Loading in data 
We provide several limited examples of 2D and 3D data in /data/
You can download the full dataset that we have used for training [here 2D](https://drive.google.com/drive/folders/199awM1qqQDqScgeI_HF65CG9PyjUWHGH?usp=sharing), [here 3D](https://drive.google.com/drive/folders/1QxtvjhCjnq9PtS9qMn5TVtSbg5sck3Ju?usp=sharing).
You also need to download the PFSs [here](https://drive.google.com/drive/folders/103q6fND3W7hH-TCkCRv6Ho0xfgyScbvK?usp=sharing) and add it to the /data folder. 

## Training for your own microscope/imaging system

### Characterize your imaging system forward model 
To retrain MultiWienerNet to work for your own imaging system, you first need to simulate realistic measurements from your imaging system to create sharp/blurred image pairs. If you already have a spatially-varying model for your imaging system (e.g. in Zemax), you can use this. If you do not have a model for your spatially-varying imaging system, we propose you follow the following calibration approach: 

* Scan a bead on a 8x8 grid across your microscope/imaging system's field of view. Repeat this for each depth plane of interest. 
* Run your data through our low rank code [here](https://github.com/Waller-Lab/MultiWienerNet/tree/main/common/process_psf_for_svd.ipynb)
    
### Create a dataset 
You can simulate data using the low rank forward model as shown in the SVD notebook above or use your own field-varying forward model to simulate measurements. 
### Train your network
We have provided training scripts for 2D imaging in tensorflow [here](https://github.com/Waller-Lab/MultiWienerNet/blob/main/tensorflow/2D%20deconvolution%20demo%20(pretrained).ipynb) and for single-shot 3D imaging in pytorch [here](https://github.com/Waller-Lab/MultiWienerNet/blob/main/pytorch/3D%20deconvolution%20demo%20(pretrained).ipynb). 
