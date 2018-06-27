# Optimal Transport and Machine Learning DS3 2018

Courses and practical sessions for the Optimal Transport and Machine learning course at [DS3 2018](http://www.ds3-datascience-polytechnique.fr).


The course has been prepared by [Marco Cuturi](http://marcocuturi.net/),
 [Rémi Flamary](http://remi.flamary.com/) and [Nicolas Courty](http://people.irisa.fr/Nicolas.Courty/).

### Course

The slides of the Cours by Marco Cuturi can be downloadable [here](https://www.dropbox.com/s/3kuqnhmf2q0dzjq/mlss18_argentina.pdf?dl=0).


More information can be found in the [Computational Optimal Transport](https://arxiv.org/pdf/1803.00567.pdf) book.


### Practical Sessions

You can download the introductory slides to the practical session [here](https://remi.flamary.com/cours/otml/OTML_TPDS3_2018.pdf).


#### Install Python and POT Toolbox

In order to do the practical sessions you need to have a working Python installation. 
The simplest way on any OS is to install the [Anaconda](https://www.anaconda.com/download/) distribution that can be freely downloaded from [here](https://www.anaconda.com/download/).

When anaconda is installed the simplest way to install pot is to launch the anaconda terminal and execute:

```
conda install -c conda-forge pot 
```

which will install the POT OT Toolbox automatically. Note that in Window you need to launch the anaconda terminal with admnistrator mode to install with conda.

The optional practical session 3 also requires the use of the Keras toolbox that can be installed similarly with:

```
conda install -c conda-forge keras 
```

#### Download the Notebooks for the session

You can download all the necessary files here: [OTML_DS3_2018.zip](https://github.com/rflamary/OTML_DS3_2018/archive/master.zip)

The zip file contains the following session:

0. [Introduction to OT with POT](0_Intro_OT.ipynb)
1. [Domain adaptation on digits with OT](1_DomainAdaptation.ipynb)
2. [Color Grading with OT](2_ColorGrading.ipynb)
3. [Wasserstein GAN in 2D](3_WGAN.ipynb) (requires keras)
4. [Word Mover's Distance on text](4_WMD.ipynb)

You can choose to do the practical session using the notebooks included or the python script. We recommend Notebooks for beginners. 

The solutions  for the practical sessions can be obtained at the following URL:

```
https://remi.flamary.com/cours/otml/solution_[NUMBER].zip
```

Where [NUMBER] has to be replaced by the integer part of the value of the Wasserstein distance obtained in Practical [Session 0](0_Intro_OT.ipynb) using the Manhattan/Cityblock ground metric.
