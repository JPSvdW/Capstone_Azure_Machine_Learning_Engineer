# Predicting whether a breast mass is malignant or benign.

This is the final project for the Udacity Machine Learning Engineer with Microsoft Azure Nanodegree Program.  In this project I will attempt to predict whether a tumor in a patient's breast is benign or malignant.  I have used a [dataset](https://www.kaggle.com/uciml/breast-cancer-wisconsin-data) from Kaggle based on a medical study done.

I will use two methods to create a machine learning model.  The first method to create a machine learning model will be AutoML and the second method will be Hyperdrive.  After creating a model from each of these methods, I will then deploy the model with the highest accuracy as a web service using an Azure Container Instance (ACI).  therafter I will submit a request to the deployed web service and obtain a prediction whether the tumor tissue is benign or malignant.  Finally I will convert the chosen model to an onnx model and terminate all services and compute resources.

## Project Set Up and Installation
*OPTIONAL:* If your project has any special installation steps, this is where you should put it. To turn this project into a professional portfolio project, you are encouraged to explain how to set up this project in AzureML.

The first step after setting up a workspace for Azure Machine Learning Studio is to add a compute instance in order to execute code in my Jupyter notebooks.

![image](https://user-images.githubusercontent.com/77330289/148642992-251d47c7-1b8e-49d0-b3e0-3011193c9984.png)

Screenshot 1: Compute instance.

Screenshot 1 shows that I have created a compute instance called "breast-cancer-compute".  I used a STANDARD_DS3_V2 compute instance.

## Dataset
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

### Overview
*TODO*: Explain about the data you are using and where you got it from.

### Task
*TODO*: Explain the task you are going to be solving with this dataset and the features you will be using for it.

### Access
*TODO*: Explain how you are accessing the data in your workspace.

## Automated ML
*TODO*: Give an overview of the `automl` settings and configuration you used for this experiment

![image](https://user-images.githubusercontent.com/77330289/148544410-64983b1c-43d7-41f1-a2f9-12b4ea1318db.png)

automl config screenshot


### Results

![image](https://user-images.githubusercontent.com/77330289/148544548-e7bb90ce-ddfa-417e-b1dd-c2b3422f1204.png)

![image](https://user-images.githubusercontent.com/77330289/148544590-5fdaf9ef-f99f-49bd-bbb6-2f5aa76a4163.png)

![image](https://user-images.githubusercontent.com/77330289/148544686-5ca11571-34bb-4cdc-933a-df18a248b432.png)

![image](https://user-images.githubusercontent.com/77330289/148544777-74ef5aca-86e6-4f50-9b64-7e0499319d52.png)

![image](https://user-images.githubusercontent.com/77330289/148544861-d31784c1-1a99-4328-8b80-4e06ec3e0b36.png)

![image](https://user-images.githubusercontent.com/77330289/148544996-16720786-a32b-40d8-814f-a3a1e41b513e.png)

![image](https://user-images.githubusercontent.com/77330289/148545082-21c436ca-5f41-4178-a026-8003799641cd.png)

![image](https://user-images.githubusercontent.com/77330289/148545114-687edc03-83d0-48ee-bc87-906c5fa97981.png)

![image](https://user-images.githubusercontent.com/77330289/148545162-db404f7d-a1a2-4d3d-9c35-8c17ae731ec0.png)

![image](https://user-images.githubusercontent.com/77330289/148546673-fad25640-0ec2-4ce6-87e4-c91771ee9c1b.png)

![image](https://user-images.githubusercontent.com/77330289/148546777-1f32568c-6233-4eca-af47-c1b60707fed2.png)

![image](https://user-images.githubusercontent.com/77330289/148554501-55121340-1c03-4f50-80e6-1d9908eafee8.png)

![image](https://user-images.githubusercontent.com/77330289/148554535-488e5bc8-e576-4d4f-b696-63849d9abd5f.png)

![image](https://user-images.githubusercontent.com/77330289/148554586-32f8cfc9-c2be-404f-92b1-5a619088840a.png)


*TODO*: What are the results you got with your automated ML model? What were the parameters of the model? How could you have improved it?

*TODO* Remeber to provide screenshots of the `RunDetails` widget as well as a screenshot of the best model trained with it's parameters.

![image](https://user-images.githubusercontent.com/77330289/148541306-0b38ece2-387f-4c41-ba57-550a8db29ce6.png)

![image](https://user-images.githubusercontent.com/77330289/148541382-25be7a9b-9ccf-4187-a247-ed010a5afd2c.png)

![image](https://user-images.githubusercontent.com/77330289/148541530-714ef6ef-070f-49b0-90d2-2691e59210db.png)

![image](https://user-images.githubusercontent.com/77330289/148541762-680de328-0cc3-4469-9b6d-7853d177e95c.png)

Screenshots of widget

![image](https://user-images.githubusercontent.com/77330289/148541877-1c642c0e-76fd-46f6-99c9-92ec636d66f8.png)

![image](https://user-images.githubusercontent.com/77330289/148541995-ae20adf0-f0e1-494c-a38b-97e2ad7b122e.png)

Best model screen shots

![image](https://user-images.githubusercontent.com/77330289/148542087-5703344b-bd37-41ab-a3eb-ce4da5b47f51.png)

Register model screenshot

## Hyperparameter Tuning
*TODO*: What kind of model did you choose for this experiment and why? Give an overview of the types of parameters and their ranges used for the hyperparameter search


### Results

![image](https://user-images.githubusercontent.com/77330289/148545241-b8b33fc4-1d70-4945-bcf0-a1d49c1ab491.png)

![image](https://user-images.githubusercontent.com/77330289/148545296-00d3fae1-6c4e-4f3c-b025-c7e077e7900a.png)

![image](https://user-images.githubusercontent.com/77330289/148545341-3a130a89-d6f2-4fd2-9fcf-29a07edaa1b4.png)

![image](https://user-images.githubusercontent.com/77330289/148545384-951bd145-f30e-4cd4-9ed0-da5e1107b441.png)

![image](https://user-images.githubusercontent.com/77330289/148545629-5406b19d-f850-4e9c-b754-d9464c09bad9.png)

![image](https://user-images.githubusercontent.com/77330289/148546021-f050c2ef-b757-4357-8f3b-f25101c8825d.png)

![image](https://user-images.githubusercontent.com/77330289/148546980-80053e78-bea4-4ccd-b7f7-68f379b73e2d.png)


*TODO*: What are the results you got with your model? What were the parameters of the model? How could you have improved it?

*TODO* Remeber to provide screenshots of the `RunDetails` widget as well as a screenshot of the best model trained with it's parameters.

![image](https://user-images.githubusercontent.com/77330289/148542279-a07630f0-1567-4345-8f43-76bbe1cc8b0e.png)

![image](https://user-images.githubusercontent.com/77330289/148542334-f0f60e96-df45-4288-8ae5-f39d2ef7ad2a.png)

![image](https://user-images.githubusercontent.com/77330289/148542421-2c18c1ad-d05b-4dbc-afc0-6918b6948d1c.png)

Widget screenshots

![image](https://user-images.githubusercontent.com/77330289/148542589-b80c88e3-8dc4-4b0a-9672-d5c1e636551f.png)

![image](https://user-images.githubusercontent.com/77330289/148542628-bd1df1c3-a955-4703-af02-a7cd7903938b.png)

Best model screenshots

![image](https://user-images.githubusercontent.com/77330289/148542688-dffc2e4e-71e9-4138-b87b-696ee72a2dfe.png)

Save and register model screenshot



## Model Deployment
*TODO*: Give an overview of the deployed model and instructions on how to query the endpoint with a sample input.

![image](https://user-images.githubusercontent.com/77330289/148542886-c3894dab-eea8-4ba8-8678-6de7715a464c.png)

![image](https://user-images.githubusercontent.com/77330289/148543135-3c05b07a-fbbf-4bca-a18a-17b300ca8cb9.png)

![image](https://user-images.githubusercontent.com/77330289/148543977-82c512a0-37d9-4a6c-8314-fbae5710bf76.png)

![image](https://user-images.githubusercontent.com/77330289/148544083-6ee64599-19bc-4341-af62-6a4dbee466c0.png)

![image](https://user-images.githubusercontent.com/77330289/148544190-dae46928-5e5b-4008-b538-11d0ce8bcd8c.png)

![image](https://user-images.githubusercontent.com/77330289/148546220-3c60c216-619a-44f5-8df2-4f2ddc59dbe2.png)

![image](https://user-images.githubusercontent.com/77330289/148546267-2b06c5b2-5c9c-493d-923d-623ef31a40db.png)


Deployed and healthy screenshot

## Screen Recording
*TODO* Provide a link to a screen recording of the project in action. Remember that the screencast should demonstrate:
- A working model
- Demo of the deployed  model
- Demo of a sample request sent to the endpoint and its response

## Standout Suggestions
*TODO (Optional):* This is where you can provide information about any standout suggestions that you have attempted.

![image](https://user-images.githubusercontent.com/77330289/148546137-3b2c7dde-4a22-47f1-83ef-a2e8c3354d82.png)

