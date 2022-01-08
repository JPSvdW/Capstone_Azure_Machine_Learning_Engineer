# Predicting whether a breast mass is malignant or benign.

This is the final project for the Udacity Machine Learning Engineer with Microsoft Azure Nanodegree Program.  In this project I will attempt to predict whether a tumor in a patient's breast is benign or malignant.  I have used a [dataset](https://www.kaggle.com/uciml/breast-cancer-wisconsin-data) from Kaggle based on a medical study done.

I will use two methods to create a machine learning model.  The first method to create a machine learning model will be AutoML and the second method will be Hyperdrive.  After creating a model from each of these methods, I will then deploy the model with the highest accuracy as a web service using an Azure Container Instance (ACI).  therafter I will submit a request to the deployed web service and obtain a prediction whether the tumor tissue is benign or malignant.  Finally I will convert the chosen model to an onnx model and terminate all services and compute resources.

## Project Set Up and Installation

The first step after setting up a workspace for Azure Machine Learning Studio is to add a compute instance in order to execute code in my Jupyter notebooks.

![image](https://user-images.githubusercontent.com/77330289/148642992-251d47c7-1b8e-49d0-b3e0-3011193c9984.png)

Screenshot 1: Compute instance.

Screenshot 1 shows that I have created a compute instance called "breast-cancer-compute".  I used a STANDARD_DS3_V2 compute instance.

After the compute instance has been created, I uploaded my two Jupyter Notebooks called, "automl.ipynb" and "hyperparameter_tuning.ipynb".

In order to use the SKLearn estimator function in my "hyperparameter_tuning.ipynb" Notebook I had to create an entry script in python.  My entry script name is "train.py".  In this script I added:
- Dependencies that will be used.
- A method to download the dataset from Kaggle.com using Tabular Dataset Factory.
- Converted the data to a pandas dataframe.
- Cleaned the data.
- Split the dataset into a training and test dataset.
- Added the Logistic Regression model to be used along with a method to fetch the hyperparameters values provided in my "hyperparameter_tuning.ipynb" notebook.

## Dataset

### Overview

The dataset is publicly available on the Kaggle website.  The dataset was created from a study done on breast cancer.  The attributes were created from a digitized image of a fine needle aspirate (FNA) of a breast mass.  The end goal of this dataset is to predict whether the breast mass of a patient is malignant or benign.
The dataset contains 30 attributes for each of the 569 patients and a single outcome or diagnosis.

A summary of the attributes is given below.

1) radius (Distance from centre to the perimeter)
2) texture (Standard deviation of gray-scale values)
3) perimeter (Perimeter of tissue sample)
4) area (Area of tissue sample)
5) smoothness (Local variation in radius lenghts)
6) compactness_mean (Perimeter^2/area - 1.0)
7) concavity_mean (Severity of concave portions of the contour)
8) concave points_mean (Number of concave portions of the contour)
9) symmetry_mean (Symmetry of the tissue sample)
10) fractal_dimension_mean ("Coastline approximation" - 1)

There are ten categories of features and for each of these categories, the mean, standard error and "worst" or largest (mean of the largest three values) were calculated to create a total of 30 features.  The outcome or target field is the diagnosis which have a value of either malignant or benign.

### Task
The objective is tu use this dataset from Kaggle.com and create a trained machine learning model to predict whether the tissue sample taken from a tumor inside the breast of a patient is benign or malignant.  A predicted outcome of 1 means malignant and a predicted outcome of 0 means benign.

### Access
This dataset was downloaded and uploaded to this repository from Kaggle.com and can be found at this [link](https://www.kaggle.com/uciml/breast-cancer-wisconsin-data)
- Tabular Dataset Factory is used to dowloaded and access the dataset from my repository in the data folder through a URL.
- The dataset was used in my "train.py" file and also in both my Jupyter Notebooks where I also show the first few lines of the dataset.  I also show some statistics of my dataset in my Notebooks.
- In my "automl.ipynb" Notebook I had registered the dataset on Azure Machine Learning Studio.

## Automated ML
In my "automl.ipynb" I executed a series of cells containing code that would create a machine learning model, save and register the best model from this experiment, and finaly deploy the model as the model with the best accuracy was obtained from the AutoML experiment.  The following is an overview of cells that were executed:

1) Import all dependencies that will be used in this notebook.
2) Show my workspace, resource group, subscription details and choose a name for my experiment.
3) Create an AML Compute cluster.

  ![image](https://user-images.githubusercontent.com/77330289/148545114-687edc03-83d0-48ee-bc87-906c5fa97981.png)
  
  Screenshot 2: AML compute cluster created.
  
  Screenshot 2 provides confirmation that the AML compute cluster was created successfully.  This screen was accessed in the compute section of Azure Machine Learning Studio under the Compute clusters tab.

4) Access the dataset from my Github repository and display the first few lines of the dataset and also some statistic on my dataset.
5) Cleaning and registering of my dataset.

  ![image](https://user-images.githubusercontent.com/77330289/148545162-db404f7d-a1a2-4d3d-9c35-8c17ae731ec0.png)
  
  Screenshot 3: Registered dataset.
  
  Screenshot 3 provides confirmation that my dataset was registered successfully in Azure Machine Learning Studio.  This screen was accessed under the datasets section.
  
6) Choose the settings and configuration for my AutoMl experiment and submitting the run.
    ![image](https://user-images.githubusercontent.com/77330289/148544410-64983b1c-43d7-41f1-a2f9-12b4ea1318db.png)
    
    Screenshot 4: Automl setting, configuration and run submission.
    
    Screenshot 4 taken from the notebook shows the AutoML settings and configuration chosen.
    
    ### AutoML settings
    
    | AutoML setting | Seting details | Value used |
    |----------------|----------------|------------|
    | experiment_timeout_minutes | Experiment duration in minutes | 30 |
    | max_concurrent_iterations | Maximum nuber of iterations that will be executed in parallel | 4 |
    | n_cross_validations | The number of cross validations to perform | 5 |
    |primary_metric | The primary metric that will be optomized to find the best model | "accuracy" |
    
    ### AutoML configuration
    
    | AutoML configuration | Config details | Value used |
    |----------------------|----------------|------------|
    |compute_target | The compute target that will be used to run the experiment | bc-compute |
    | task | type of task that AutoML will run | "classification" |
    | training_data | Dataset that will be used to train the models | final_training_dataset |
    | label_column_name | The column that will be used as the predicted outcome | "malignant" |
    | path | Path to my project folder | my_automl_project_folder |
    | enable_early_stopping | Method to stop training if the accuracy does not improve | True |
    | enable_onnx_compatible_models | Enable or disable the use of onnx compatible models | True |
    | featurization | Choice of featurization to be used | "auto" |
    | debug_log | Choose a name for the AutoML error/debug log | "automl_errors.log" |
    
    ![image](https://user-images.githubusercontent.com/77330289/148544548-e7bb90ce-ddfa-417e-b1dd-c2b3422f1204.png)
    
    Screenshot 5: List of experiments.
    
    screenshot 5 shows a list of experiments the focus here is on "my_automl_breat_cancer_experiment.  This list can be found under the experiments section in Azure Machine    Learning Studio.

    ![image](https://user-images.githubusercontent.com/77330289/148544590-5fdaf9ef-f99f-49bd-bbb6-2f5aa76a4163.png)
    
    Screenshot 6: AutoML experiment.
    
    Screenshot 6 shows that my AutoML experiment has completed successfully.  This screenshot can be found by clicking on "my_automl_breast_cancer experiment" under the experiments section of Azure Machine Learning Studio.

    ![image](https://user-images.githubusercontent.com/77330289/148544686-5ca11571-34bb-4cdc-933a-df18a248b432.png)
    
    Screenshot 7: Details of the AutoML experiment.
    
    Screenshot 7 provides high level details on the completed AutoML experiment.  This screenshot was accessed by clicking on lemon_pillow_1qj172kb at the bottom of screenshot 6.
    
    ![image](https://user-images.githubusercontent.com/77330289/148544777-74ef5aca-86e6-4f50-9b64-7e0499319d52.png)
    
    Screenshot 8: Models pruduced by the AutoML experiment.
    
    Screenshot 8 provides a list of the models trained during the AutoML experiment.  This screen was accessed under Models tab from screenshot 7.

    ![image](https://user-images.githubusercontent.com/77330289/148544861-d31784c1-1a99-4328-8b80-4e06ec3e0b36.png)
    
    Screenshot 9: Explanation of the best model (VotingEnsemble).
    
    screenshot 9 provides an explanation of the best model from the AutoML experiment which is a Voting Ensemble model.  The explanation in this screenshot provides the top four important features used in the training of this model.  This screen can be accessed by clicking on the explanation link next to the best model (VotingEnsemble) in screenshot 8.
    
7) Show run details by using the run details widget.
8) Display the details of the best model.
9) Save and register the best model.

  ![image](https://user-images.githubusercontent.com/77330289/148546673-fad25640-0ec2-4ce6-87e4-c91771ee9c1b.png)
  
  Screenshot 10: List of registered models.
  
  Screenshot 10 provides a list of the best models that was registered.  In this section we only focuss on the best AutoML model that was registered.  This screen was accessed under the models section.
  
  ![image](https://user-images.githubusercontent.com/77330289/148542087-5703344b-bd37-41ab-a3eb-ce4da5b47f51.png)
  
  Screenshot 11: Model registartion in Notebook.
  
  Screenshot 11 provides confirmation that I have registered the best AutoML model using code in a Jupyter Notebook.

  ![image](https://user-images.githubusercontent.com/77330289/148546777-1f32568c-6233-4eca-af47-c1b60707fed2.png)
  
  Screenshot 12: Details of the best AutoML model that was registered.
  
  Screenshot 12 provides confirmation that I successfully registered the best AutoML model.  This screenshot specifically show some high level details of the AutoML models that was registered.  This screenshot was accessed by clicking on "my-best-automl-model" in screenshot 10.
  
10) Deploy the best model as an Azure Container Instance (ACI).
11) Show the state of the deployed model.
12) Send a request to the deployed model and obtain a prediction.
13) print web service logs.
14) Convert and save the best model to a onnx format.
15) Delete the service.

  ![image](https://user-images.githubusercontent.com/77330289/148554586-32f8cfc9-c2be-404f-92b1-5a619088840a.png)
  
  Screenshot 12: Proof that services was deleted.
  
  Screenshot 12 provides proof that all services used in this project was deleted.  This screenshot was accessed under the endpoints section.

### Results

The results from the AutoML experiment were as follow:
- Best model = Voting Ensemble
- Accuracy = ~ 98%

![image](https://user-images.githubusercontent.com/77330289/148541306-0b38ece2-387f-4c41-ba57-550a8db29ce6.png)

Screenshot 13: RunDetails widget with list of models.

Screenshot 13 shows the RunDetails widget with a list of the completed models.  This screenshot was taken from my Notebook after the execution of the code for the RunDetails widget.

![image](https://user-images.githubusercontent.com/77330289/148541382-25be7a9b-9ccf-4187-a247-ed010a5afd2c.png)

Screenshot 14: RunDetails widget with a graph of the accuracy of the models.

Screenshot 14 shows the RunDetails widget with a graph of the accuracies of the completed models.  This screenshot was taken from my Notebook after the execution of the code for the RunDetails widget.

![image](https://user-images.githubusercontent.com/77330289/148541530-714ef6ef-070f-49b0-90d2-2691e59210db.png)

Screenshot 15: List of all the models trained.

Screenshot 15 shows a list of all the models trained by the AutoML experiment.  The duration of training and the metric is provided next to each model.

![image](https://user-images.githubusercontent.com/77330289/148541762-680de328-0cc3-4469-9b6d-7853d177e95c.png)

Screenshot 16: Output from the AutoML experiment.

Screenshot 16 provides some output regarding the AutoML experiment that was completed successfully.

![image](https://user-images.githubusercontent.com/77330289/148541877-1c642c0e-76fd-46f6-99c9-92ec636d66f8.png)

Screenshot 17: Metrics of the best AutoML model.

Screenshot 17 provides all of the metrics for the best AutoML model.

![image](https://user-images.githubusercontent.com/77330289/148541995-ae20adf0-f0e1-494c-a38b-97e2ad7b122e.png)

Screenshot 18: Details of the best AutoML model.

Screenshot 18 provides details of the best AutoML model.  Details like "Run ID", "Type" and "Status" is shown.

![image](https://user-images.githubusercontent.com/77330289/148648882-6aa920fd-c11a-46c5-aaba-b8deb6a5fc14.png)

Screenshot 19: Parameters of the trained Voting Ensemble model.

Screenshot 19 provides a view of the parametrs from the best AutoML model (VotingEnsemble) that was trained during the experiment.  To llok at the complete output of parameters, please look at the "ensemble_model_parameters.txt" file in this repository.

| Algorithm | Wheight |
|-----------|---------|
| logisticregression - maxabsscaler ('C': 1.7575106248547894)  | 0.15384615384615385 | 
| kneighborsclassifier - minmaxscaler | 0.07692307692307693 | 
| logisticregression - maxabsscaler ('C': 719.6856730011514) | 0.15384615384615385 | 
| kneighborsclassifier - robustscaler | 0.07692307692307693 | 
| randomforestclassifier - standardscalerwrapper | 0.15384615384615385 | 
| xgboostclassifier - standardscalerwrapper | 0.07692307692307693 | 
| xgboostclassifier - sparsenormalizer | 0.07692307692307693 | 
| svcwrapper - standardscalerwrapper | 0.23076923076923078 | 

A Voting Ensemble model combines the outputs of multiple algorithms.  The final model consists of a wheighted sum of all the chosen algorithms' outputs.

The accuracy of the model could be improved by increasing the experimant duration.  This would give increase training time which might lead to better model accuracy.  If a medical practitioner has in depth knowledge it can help to do specific feature engineering and provide those specific features rather that allowing AutoML to automate it.  This could also possibly improve the accuracy of the final model.

## Hyperparameter Tuning

I chose to use a logistic regression model to predict wheter the tissue from a breast tumor is benign or malignant.  I chose this model due to its simplicity with regards to implementation and interpretation, and low computational needs.  I have chosen to optimize the following two parameters:
- Regularization strength C with uniform search space of (0.1, 1)
- maximum iteration with a choice of (50, 100, 150, 200, 250)

I chose a random parameter sampling method due to it saving computational resources.  Although this sampling method saves on computational resources it still produces reasonably good models.  Sampling methods like Grid Parameter Sampling, will use every value within a search space thus using significantly more computational resources.  The Random Parameter Sampling method also allows for early termination policies.  This allows for the termination of runs performing poorly.

I chose an early termination policy called BanditPolicy.  This is one of the more aggressive policies, allowing to stop more runs.  This type of early termination policy results in significat reduction of computation time. The following parameters were used in the BanditPolicy:
- evaluation_interval = 1, this is the frequency at which the policy will be applied. 
- slack_factor = 0.1, this is the amount of slack between a executing run and the best performing training run.
- delay_evaluation = 4, number of interval to delay the policy and avoids premature termination.

In this section i executed a number of code cells from my "hyperparameter_tuning.ipynb" Notebook.  The following is an overview of steps executed in this notebook:
1) Import all dependencies.
2) Load, show first few lines of the dataset, and statistics.
3) Choose an experiment name.
4) Create a AML Compute Cluster or reuse if already exists.

  ![image](https://user-images.githubusercontent.com/77330289/148545114-687edc03-83d0-48ee-bc87-906c5fa97981.png)
  
  Screenshot 20: re-use AML compute cluster created.
  
  Screenshot 20 provides confirmation that the AML compute cluster was created successfully.  This cluster will be re-used in this section.
  
5) Add Hyperdrive configuration and settings.

  ![image](https://user-images.githubusercontent.com/77330289/148651570-f3267ba3-cff3-46c3-befe-7d4c37140239.png)
  
  Screenshot 21: Hyperdrive settings and configuration.
  
  Screenshot 21 shows the Hyperdrive settings and configuration chosen.

6) Submit experiment.

  ![image](https://user-images.githubusercontent.com/77330289/148545241-b8b33fc4-1d70-4945-bcf0-a1d49c1ab491.png)
  
  Screenshot 22: Completed Hyperdrive experiment.
  
  Screenshot 22 provides confirmation that my Hyperdrive experiment has completed successfully.  This screenshot wass accessed under the experiments tab and clicking on "breast_cancer_hyperdrive_experiment".

  ![image](https://user-images.githubusercontent.com/77330289/148545296-00d3fae1-6c4e-4f3c-b025-c7e077e7900a.png)
  
  Screenshot 23: Hyperdrive experiment details.
  
  Screenshot 23 provides some details on the best Hyperdrive model.  The screenshot was accessed by clicking on "quiet_river_p3hlx7x1" at the bottom of screenshot 22.

  ![image](https://user-images.githubusercontent.com/77330289/148545341-3a130a89-d6f2-4fd2-9fcf-29a07edaa1b4.png)
  
  Screenshot 24: Hyperdrive experiment child runs.
  
  Screenshot 24 provides a visual view of the child runs completed during the Hyperdrive experiment.  This screen can be accessed under the child runs tab in screenshot 23.

  ![image](https://user-images.githubusercontent.com/77330289/148545384-951bd145-f30e-4cd4-9ed0-da5e1107b441.png)
  
  Screenshot 25: List of child runs.
  
  Screenshot 25 shows the list of child runs completed.  This screen is shown when scrolling dow in screenshot 24.
  
  ![image](https://user-images.githubusercontent.com/77330289/148545629-5406b19d-f850-4e9c-b754-d9464c09bad9.png)
  
  Screenshot 26: Details of the best Hyperdrive run.
  
  Screenshot 26 provides details on the best Hyperdrive run.
  
8) Show RunDetails using widget.
9) Provide details on the best Hyperdrive model.
10) Save and register the best hyperdrive model.

![image](https://user-images.githubusercontent.com/77330289/148546980-80053e78-bea4-4ccd-b7f7-68f379b73e2d.png)

Screenshot 27: Details of the best hyperdrive model.

Screenshot 27 shows an overview of the details from the best Hyperdrive model that was registered in Azure Machine Learning Studio.

### Results

*TODO*: What are the results you got with your model? What were the parameters of the model? How could you have improved it?
The best hyperdrive model had an accuracy of approximately 96%.  The hyperparameter values for this model:
- Regularization strength of 0.31218517931449213
- Maximum iteration of 50

One way to improve the accuracy of this model is to take the regularization strength value and maximum iteration from this model and refine it with a Grid Parameter Sampling method. 

![image](https://user-images.githubusercontent.com/77330289/148542279-a07630f0-1567-4345-8f43-76bbe1cc8b0e.png)

Screenshot 28: RunDetails widget with run list.

Screenshot 28 provides the completed run list of the Hyperdrive experiment.

![image](https://user-images.githubusercontent.com/77330289/148542334-f0f60e96-df45-4288-8ae5-f39d2ef7ad2a.png)

Screenshot 29: RunDetials widget showing graph of run accuracies.

Screenshot 29 provides a graph of the accuracy for each completed run.

![image](https://user-images.githubusercontent.com/77330289/148542421-2c18c1ad-d05b-4dbc-afc0-6918b6948d1c.png)

Screenshot 30: Hyperdrive experiment output.

Screenshot 30 provides a text based output of the completed Hyperdrive experiment.

![image](https://user-images.githubusercontent.com/77330289/148542589-b80c88e3-8dc4-4b0a-9672-d5c1e636551f.png)

Screenshot 31: Detailed information from the best Hyperdrive model.

Screenshot 31 provides in depth information regarding the best model from the Hyperdrive experiment.

![image](https://user-images.githubusercontent.com/77330289/148542628-bd1df1c3-a955-4703-af02-a7cd7903938b.png)

Screenshot 32: Summary of the best Hyperdrive model.

Screenshot 32 provides a summary of the best Hyperdrive model.  The run Id, status, best run accuracy of ~96%, and the hyperparameter values of the best model is shown.

![image](https://user-images.githubusercontent.com/77330289/148542688-dffc2e4e-71e9-4138-b87b-696ee72a2dfe.png)

Screenshot 33: Save and register model.

Screenshot 33 provides confirmation that the best Hyperdrive model has been saved and registered.

## Model Deployment

The model with the best accuracy of approximately 98% and was obtained from the AutoML experiment.  The Hyperdrive model had a accuracy of approximately 96%.  Therefore I will deploy the best model from the AutoML experiment.

In order to deploy this model, I had to navigate to the outputs and logs section of the best AutoML model and download and save my environment variables file and scoring python file that was generated by AutoML.  These two files can be found in this repository with the names "my-best-model-conda-env.yml" and "score.py". 

![image](https://user-images.githubusercontent.com/77330289/148542886-c3894dab-eea8-4ba8-8678-6de7715a464c.png)

Screenshot 34: Model deployement set-up.

From screenshot 34, it is seen that I pass the environment details file and score.py file to the inference config method.  I used an Azure Container Instance Webservice deployment with one CPU core, 1GB of memory, enabled authentication, and enabled application insights.  From the output in this screenshot, it is observed that the deployment was successful.

![image](https://user-images.githubusercontent.com/77330289/148543135-3c05b07a-fbbf-4bca-a18a-17b300ca8cb9.png)

![image](https://user-images.githubusercontent.com/77330289/148543977-82c512a0-37d9-4a6c-8314-fbae5710bf76.png)

![image](https://user-images.githubusercontent.com/77330289/148544083-6ee64599-19bc-4341-af62-6a4dbee466c0.png)

![image](https://user-images.githubusercontent.com/77330289/148544190-dae46928-5e5b-4008-b538-11d0ce8bcd8c.png)

Deployed and healthy screenshot

## Deleting compute resources

![image](https://user-images.githubusercontent.com/77330289/148554501-55121340-1c03-4f50-80e6-1d9908eafee8.png)

![image](https://user-images.githubusercontent.com/77330289/148554535-488e5bc8-e576-4d4f-b696-63849d9abd5f.png)

## Screen Recording
A screen recording of this project can be accessed by this [link.](https://youtu.be/jTq5RRj3R8c)

## Standout Suggestions
*TODO (Optional):* This is where you can provide information about any standout suggestions that you have attempted.

![image](https://user-images.githubusercontent.com/77330289/148546137-3b2c7dde-4a22-47f1-83ef-a2e8c3354d82.png)

