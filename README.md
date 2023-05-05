# INFO 251 Final project

In this project, we are training a machine learning model to predict housing prices in King county, WA, USA. We have also used our model to predict current housing prices from Zwillow. In this report, we will discuss our dataset, machine learning model and results. 

This repo consists of two important files:
- [houseprediction.ipynb](houseprediction.ipynb) is the file for EDA and Feature engineering phase
- [train.ipynb](train.ipynb) contains our machine learning part of the project

## How to reproduce our training data

1. You need to have an extension or a program to execute python notebook files installed. We reccomend VSCode 

2. Go to [houseprediction.ipynb](houseprediction.ipynb)

3. If you do not have tabulate installed, do so with

     ```pip install tabulate```

4. Run the notebook. All cells should run consecutively and produce results, and export them to a file named traindata.csv.
![run notebook image](run.png "How to run notebook")

The treated training data is already available at [traindata.csv](traindata.csv)

## How to run our models

1. As above, you also need to have an extension or program to execute python notebook files.

2. Go to [train.ipynb](train.ipynb)

3. If you do not have the necessary extension installed, install them with the following commands. You might want to run them to check if you have everything installed.

    ```pip install pandas```

    ```pip install sklearn```

    ```pip install matplotlib```

    ```pip install numpy```

    ```pip install seaborn```

    ```pip install numpy```

    ```pip install tabulate```

4. Run the notebook. Reccomended python kernel version is 3.9 or above 
![run notebook image](run.png "How to run notebook")

Notice that the train.ipynb notebook might take a while to run the model optimization.