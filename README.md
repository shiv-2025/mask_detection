# mask_detection

It is no doubt that doing a data science and machine learning project, starting from collecting the data, processing the data, visualizing insights about the data, and developing a machine learning model to do a predictive task is a fun thing to do.

Creating a web app is one of the solutions such that other people can make use of our machine-learning model. Fortunately, it is very simple to create a web app nowadays using the Streamlit library of Python

For this computer vision program, we have used a dataset available on the Kaggle website

Overall there are three steps followed in this project:

*    Creating the first Python file to load the data, build the model, and finally train the model.
*    Creating the second Python file to load the model and build the web app.
*    Deploying the web app using Heroku.

Few important points related to the mask detection project:

*    Created a website that predicts whether the person wearing a mask or not. We have used the Heroku platform for deployment.
*    Used the Sequential model of Keras which allows us to create models layer-by-layer for most problems. It is eco-friendly and it is very easy to create neural network models.
*    After training the model, it is stored in HDF5 format (HDF5 file stands for Hierarchical Data Format 5. It is an open-source file that comes in handy to store a large amount of data).
*    Created a webpage using the Streamlit library of Python. It creates a web interface that allows uploading images for prediction.
