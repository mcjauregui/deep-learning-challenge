# deep-learning-challenge

The nonprofit foundation Alphabet Soup wants a tool that can help it select the applicants for funding with the best chance of success in their ventures. With your knowledge of machine learning and neural networks, you’ll use the features in the provided dataset to create a binary classifier that can predict whether applicants will be successful if funded by Alphabet Soup.  

From Alphabet Soup’s business team, you have received a CSV containing more than 34,000 organizations that have received funding from Alphabet Soup over the years. Within this dataset are a number of columns that capture metadata about each organization, such as:  

EIN and NAME—Identification columns  
APPLICATION_TYPE—Alphabet Soup application type  
AFFILIATION—Affiliated sector of industry  
CLASSIFICATION—Government organization classification  
USE_CASE—Use case for funding  
ORGANIZATION—Organization type  
STATUS—Active status  
INCOME_AMT—Income classification  
SPECIAL_CONSIDERATIONS—Special considerations for application  
ASK_AMT—Funding amount requested  
IS_SUCCESSFUL—Was the money used effectively  

Steps taken to proprocess the data, and to compile, train, and evaluate the model 

Step 1: Preprocess the Data
Using Pandas, Google Colab, and scikit-learn’s StandardScaler(), I preprocessed the dataset by 
a) Read in the charity_data.csv to a Pandas DataFrame  
b) Identifying the model target, IS_SUCCESSFUL and the model features (all other columns except EIN and NAME, which I dropped)  
c) Determining the number of unique values for each column  
d) Determining the number of data points for each unique valule for columns with more than 10 unique values  
e) Using the number of data points for each unique value to pick a cutoff point to combine "rare" categorical variables together in a new value, Other  
f) Using pd.get_dummies() to encode categorical variables by creating new Boolean columns for each value of the categorical variable  
g) Splitting the preprocessed data into a features array, X, and a target array, y   
h) Using these arrays and the train_test_split function to split the data into training and testing datasets  
i) Scaling the training and testing features datasets by creating a StandardScaler instance, then fitting it to the training data and using the transform function  

Step 2: Compile, Train, and Evaluate the Model  
I next defined, trained, and evaluated the model by   
a) Using TensorFlow and Keras to create a neural network with a binary classification model able to predict whether an Alphabet Soup-funded organization will be successful (based on the features in the dataset)     
b) Assigning the number of input features and nodes for each layer 
c) Creating the first hidden layer with an appropriate activation function  
d) Creating an output layer with an appropriate activation function  
e) Create a callback to save the model's weights every five epochs
f) Compiling and training the binary classification model  
g) Evaluating the model using the test data to determine the model's loss and accuracy values  
h) Save and export the results to an HDF5 file named AlphabetSoupCharity.h5 

If necessary, add a second hidden layer with an appropriate activation function.  

Step 3: Optimize the Model
I made three adjustments to the initial model, attempting to optimize the model to achieve a target predictive accuracy higher than 75%.  
I created a new Google Colab file, AlphabetSoupCharity_Optimization.ipynb, using the same dependencies, dataset, and preprocessing techniques as before. This file is AlphabetSoupCharity_Optimization.h5.    
I modified the original model is three ways:  
a) Adding more neurons to a hidden layer  
b) Adding more neurons to a hidden layer + Adding more hidden layers   
c) Adding more neurons to a hidden layer + Adding more hidden layers + Adding the number of epochs to the training regimen  

Analysis:

The initial model contained 3 layers: an input, an output, and one hidden.  Layers 1 and 2 contained 6 neurons and layer 3 contained 1. Across the 10 epochs, loss fluctuated by stayed around the 0.565 point. Similarly, accuracy remained around 0.72.   The final loss and accuracy values with the test data ended up at 0.733 and 0.556. The model did not do a good job at predicting nonprofit success.   

![Orig_a](https://github.com/mcjauregui/deep-learning-challenge/assets/151464511/dc90f53a-c11f-49c5-adcf-c5cfef27ce27)

![Orig_b](https://github.com/mcjauregui/deep-learning-challenge/assets/151464511/995ccdb0-d4b8-407b-a0f7-5cfce584a0e8)

The first attempt a optimizing the model entailed holding the number of layers and epochs the same but doubling the number of neurons in layers 1 and 2. This resulted in a total number of parameters equal to 709. This adjustment resulted in a loss of 1.128 and an accuracy of 0.664, which means the adjusted model is slightly better at predicting nonprofit success. 

![Opt_01_a](https://github.com/mcjauregui/deep-learning-challenge/assets/151464511/9a23f2d7-c277-4184-9c07-87eadf93cd42)

![Op01_b](https://github.com/mcjauregui/deep-learning-challenge/assets/151464511/995953e9-153d-4ec7-bc9b-bab5de9e4ee7)

For the second attempt at optimizing the model, I "incremented" it by adding two new hidden layers. Layers 1 through 4 contained 12 neurons each, and layer 5 remained with 1. Total number of parameters grew to 1,021. Despite the training data yielding accuracy scores around 0.72, the test data in this version of the model only reached 0.610 after 10 epochs.

![Opt_02_a](https://github.com/mcjauregui/deep-learning-challenge/assets/151464511/1d826fb7-1a73-4a08-bf28-a6baeacc28a4)
![Opt_02_b](https://github.com/mcjauregui/deep-learning-challenge/assets/151464511/c09b2d64-01e0-439a-b6c6-cec1f56695ea)

In the thrid optimization attempts, I increased only the number of epochs during training. Layers and neurons remained the same as in the previous optimization attempt. Again, accuracy scores during the training phase ranged between 0.72 and 0.73, but the model scored only a  0.624 with the testing data. Additionally, the model triggered an early stop at epoch 24. By epoch 25 the accuracy scores were not improving so the model finshed before reaching 100 epochs. 

![Opt_03_a](https://github.com/mcjauregui/deep-learning-challenge/assets/151464511/04772fc9-f942-454e-8aad-94bc16e2f20f)
![Opt_03_b](https://github.com/mcjauregui/deep-learning-challenge/assets/151464511/c2156def-ae79-434f-90b4-2503e2076279)


