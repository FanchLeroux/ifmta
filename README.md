# ifmta

imfta is a library develloped at IMT Atlantique, Brest, France allowing to compute computer generate holograms using severals ifta algorithms 

## Getting Started

### Installing

We encourage users to use virtual environments in their development pipeline when working with or developing ifmta.

You can simply create and activate a virtual environment with anaconda by using the following syntax:

```bash
conda create --name <environment_name>
conda activate <environment_name>
```

To deactivate the virtual environemnt, you can always use `deactivate` command in your terminal.

Then you will need to install python (the library was develloped on version 3.11.7). 
To do so with anaconda run the following code :

```bash
conda install python=3.11.7
```

Now let's download the ifmta package from gitlab. Place yourself in the folder on which you want to download the code 
and run the following code :

```bash
git clone https://github.com/FanchLeroux/ifmta
```

The ifmta package uses various python packages. To install them all at once yo can run the following code :

```bash
cd ifmta
pip install -r requirements.txt
```

Finally, you can install the ifmta package in editable mode by running the following code :
(make sure you are in the ifmta folder containing the setup.py file in your terminal when running the following line)

```bash
pip install -e .
```

#### Usage and examples

You can import ifmta and start designing your first EODs!

As an example, the script main_hologram_optimisation provide the workflow used by François Leroux for the design of EOD to be used in smart contact lenses, during his master thesis in spring 2024.

