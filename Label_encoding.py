
import pandas as pd
import seaborn as sns
from sklearn import preprocessing
from sklearn.preprocessing import OneHotEncoder
from ydata_profiling import ProfileReport
import matplotlib.pyplot as plt


#Loading in dataset and applying the necessary header row
Employeesdf = pd.read_excel('New_Interviewee_Case_Study_Dataset_FINAL__282_29.xlsx')
# Use pandas get_dummies method to encode the binary variables
Employeesdf = pd.get_dummies(Employeesdf, columns=["Over18", "complaintresolved", "Left"], prefix=["Over18", "complaintresolved", "Left"])

# Drop the "Left_No" column and rename the "Left_Yes" column to "Left"
Employeesdf = Employeesdf.drop(columns=["Left_No"])
Employeesdf = Employeesdf.rename(columns={"Left_Yes": "Left"})

# Convert the "Left" column to integer data type
Employeesdf["Left"] = Employeesdf["Left"].astype(int)
Employeesdf.to_csv('employeesdf.csv')



