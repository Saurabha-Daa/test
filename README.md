# Description
This app is created to predict default tendency of a loan applicant. This app is based on a Kaggle competition which can be viewed at https://www.kaggle.com/c/home-credit-default-risk/data. 

# How to run the code
Run app.py using streamlit. Please note that app1.py also exists. However, app1.py does not need to be run.

# How to use the rendered page
1) Template to enter query data can be downloaded from the rendered web page.
2) Data in the given format can be uploaded using file uploader option on the page.
3) Once the file gets uploaded, predictions are displayed on the web page. Also a download button appears which can be clicked to download the csv file which consists of query data set with additional 'DEFAULT TENDENCY' column.
4) If the uploaded file is not in the format of the given template, an error is displayed which says 'Query columns do not match the columns of required format as given in template. Please upload query data in the given format'. 
5) Sidebar contains links to Github page (for codes) and Kaggle cometition page (for task description and data).
