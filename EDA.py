from matplotlib.patches import Patch

import pandas as pd
import seaborn as sns
from sklearn import preprocessing
from sklearn.preprocessing import OneHotEncoder
from ydata_profiling import ProfileReport
import matplotlib.pyplot as plt
import matplotlib
sns.set_style('darkgrid') # darkgrid, white grid, dark, white and ticks
plt.rc('axes', titlesize=13)     # fontsize of the axes title
plt.rc('axes', labelsize=7)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=7)    # fontsize of the tick labels
plt.rc('ytick', labelsize=7)    # fontsize of the tick labels
plt.rc('legend', fontsize=7)    # legend fontsize
plt.rc('font', size=5)          # controls default text sizes

matplotlib.rcParams.update({'font.size':7 })
plt.rcParams["font.weight"] = "bold"
plt.rcParams["axes.labelweight"] = "bold"



employeesdf = pd.read_csv('employeesdf.csv').drop(['Unnamed: 0'],axis=1)
# print(employeesdf.keys())
# # employeesdf.iloc[:,:].hist(figsize= (8,6))
# # plt.show()



# Department_leavers = pd.DataFrame(columns=['Department', '% of leavers'])
# i = 0
# for department in list(employeesdf['Department'].unique()):
#     ratio = employeesdf[(employeesdf['Department'] == department) & (employeesdf['Left'] == 1)].shape[0] / employeesdf[employeesdf['Department'] == department].shape[0]
#     Department_leavers.loc[i] = (department, ratio * 100)
#     i += 1
#
# plt.figure(figsize=(10,6))
# sns.set_style("dark")
# bar = plt.barh(Department_leavers['Department'], Department_leavers['% of leavers'],
#                 color = sns.color_palette("husl", len(Department_leavers)), edgecolor = 'cyan')
# plt.xlabel('% of leavers')
# plt.ylabel('Department')
# plt.title('Leavers by Department (%)')
#
# # Add a legend
# handles = [Patch(color=bar[i].get_facecolor(), label=Department_leavers.loc[i,'Department'])
#            for i in range(len(Department_leavers))]
# plt.legend(handles=handles)
#
# plt.show()

# Gender_leavers = pd.DataFrame(columns=['Gender', '% of leavers'])
# i = 0
# for Gender in list(employeesdf['Gender'].unique()):
#     ratio = employeesdf[(employeesdf['Gender'] == Gender) & (employeesdf['Left'] == 1)].shape[0] / employeesdf[employeesdf['Gender'] == Gender].shape[0]
#     Gender_leavers.loc[i] = (Gender, ratio * 100)
#     i += 1
#
# plt.figure(figsize=(10,6))
# sns.set_style("dark")
# bar = plt.barh(Gender_leavers['Gender'], Gender_leavers['% of leavers'],
#                 color = sns.color_palette("husl", len(Gender_leavers)), edgecolor = 'cyan')
# plt.xlabel('% of leavers')
# plt.ylabel('Gender')
# plt.title('Leavers by Gender (%)')
# handles = [Patch(color=bar[i].get_facecolor(), label=Gender_leavers.loc[i,'Gender'])
#            for i in range(len(Gender_leavers))]
# plt.legend(handles=handles)
#
# plt.show()




# MI_leavers = pd.DataFrame(columns=['MonthlyIncome', '% of leavers'])
# i = 0
# for earners in list(employeesdf['MonthlyIncome'].unique()):
#     ratio = employeesdf[(employeesdf['MonthlyIncome'] == earners) & (employeesdf['Left'] == 1)].shape[0] / employeesdf[employeesdf['MonthlyIncome'] == earners].shape[0]
#     MI_leavers.loc[i] = (earners, ratio * 100)
#     i += 1
#
# plt.figure(figsize=(10,6))
# sns.set_style("dark")
# bar = plt.barh(MI_leavers['MonthlyIncome'], MI_leavers['% of leavers'],
#                 color = sns.color_palette("husl", len(MI_leavers)), edgecolor = 'cyan')
# plt.xlabel('% of leavers')
# plt.ylabel('MonthlyIncome')
# plt.title('Leavers by MonthlyIncome (%)')
# handles = [Patch(color=bar[i].get_facecolor(), label=MI_leavers.loc[i,'MonthlyIncome'])
#            for i in range(len(MI_leavers))]
# plt.legend(handles=handles)
#
# plt.show()

#Leavers by Working From Home
# employeesdf1 = employeesdf
# employeesdf1['workingfromhome'] = employeesdf1['workingfromhome'].map({0:'No',1:'Yes'})
# FromHomedf1 = pd.DataFrame(columns=['workingfromhome', '% of leavers'])
#
# i = 0
# for employees in list(employeesdf1['workingfromhome'].unique()):
#     ratio = employeesdf1[(employeesdf1['workingfromhome'] == employees) & (employeesdf1['Left'] == 1)].shape[0] / employeesdf1[employeesdf1['workingfromhome'] == employees].shape[0]
#     FromHomedf1.loc[i] = (employees, ratio * 100)
#     i += 1
#
# plt.figure(figsize=(10,6))
# sns.set_style("dark")
# bar = plt.barh(FromHomedf1['workingfromhome'], FromHomedf1['% of leavers'],
#                 color = sns.color_palette("husl", len(FromHomedf1)), edgecolor = 'cyan')
# plt.xlabel('% of leavers')
# plt.ylabel('workingfromhome')
# plt.title('Leavers Working From Home (%)')
# handles = [Patch(color=bar[i].get_facecolor(), label=FromHomedf1.loc[i,'workingfromhome'])
#            for i in range(len(FromHomedf1))]
# plt.legend(handles=handles)
#
# plt.show()


# DistH_leavers = pd.DataFrame(columns=['DistanceFromHome', '% of leavers'])
# i = 0
# for employee in list(employeesdf['DistanceFromHome'].unique()):
#     ratio = employeesdf[(employeesdf['DistanceFromHome'] == employee) & (employeesdf['Left'] == 1)].shape[0] / employeesdf[employeesdf['DistanceFromHome'] == employee].shape[0]
#     DistH_leavers.loc[i] = (employee, ratio * 100)
#     i += 1
#
# plt.figure(figsize=(10,6))
# plt.barh(DistH_leavers['DistanceFromHome'], DistH_leavers['% of leavers'])
# plt.xlabel('% of leavers')
# plt.ylabel('DistanceFromHome')
# plt.title('Leavers by Distance from Home (%)')
# plt.show()

#
# #Class Distribution
# employeesdf['Left'] = employeesdf['Left'].astype(int)
# Percentage_former = (employeesdf[employeesdf['Left'] == 1].shape[0] / employeesdf.shape[0]) * 100
# Percentage_active = (employeesdf[employeesdf['Left'] == 0].shape[0] / employeesdf.shape[0] * 100)
# print('The percentage of current employees is: {:.1f}, The percentage of former employees is: {:.1f}'.format(
#     Percentage_active, Percentage_former))

#
# plt.hist(employeesdf['Left'])
# plt.title('Distribution of Current and Former Employees')
# plt.suptitle('The percentage of current employees is: {:.1f}, The percentage of former employees is: {:.1f}'.format(
# Percentage_active, Percentage_former))
# plt.show()

# #Calculate correlations
# sns.heatmap(employeesdf.corr(), annot=True, fmt='.2f')
# plt.title('Correlation heatmap')
# plt.show()


# employeesdf1 = employeesdf
# employeesdf1['workingfromhome'] = employeesdf1['workingfromhome'].map({0:'No',1:'Yes'})
# FromHomedf1 = pd.DataFrame(columns=['workingfromhome', '% of leavers'])
#
# i = 0
# for employees in list(employeesdf1['workingfromhome'].unique()):
#     ratio = employeesdf1[(employeesdf1['workingfromhome'] == employees) & (employeesdf1['Left'] == 1)].shape[0] / employeesdf1[employeesdf1['workingfromhome'] == employees].shape[0]
#     FromHomedf1.loc[i] = (employees, ratio * 100)
#     i += 1
#
# plt.figure(figsize=(10,6))
# sns.set_style("dark")
# bar = plt.barh(FromHomedf1['workingfromhome'], FromHomedf1['% of leavers'],
#                 color = sns.color_palette("husl", len(FromHomedf1)), edgecolor = 'cyan')
# plt.xlabel('% of leavers')
# plt.ylabel('workingfromhome')
# plt.title('Leavers Working From Home (%)')
# handles = [Patch(color=bar[i].get_facecolor(), label=FromHomedf1.loc[i,'workingfromhome'])
#            for i in range(len(FromHomedf1))]
# plt.legend(handles=handles)
#
# plt.show()


# Performance_rating_leavers = pd.DataFrame(columns=['PerformanceRating', '% of leavers'])
# i = 0
# for performers in list(employeesdf['PerformanceRating'].unique()):
#     ratio = employeesdf[(employeesdf['PerformanceRating'] == performers) & (employeesdf['Left'] == 1)].shape[0] / employeesdf[employeesdf['PerformanceRating'] == performers].shape[0]
#     Performance_rating_leavers.loc[i] = (performers, ratio * 100)
#     i += 1
#
# plt.figure(figsize=(10,6))
# sns.set_style("dark")
# bar = plt.barh(Performance_rating_leavers['PerformanceRating'], Performance_rating_leavers['% of leavers'],
#                 color = sns.color_palette("husl", len(Performance_rating_leavers)), edgecolor = 'cyan')
# plt.xlabel('% of leavers')
# plt.ylabel('PerformanceRating')
# plt.title('Leavers by Performance Rating (%)')
# handles = [Patch(color=bar[i].get_facecolor(), label=Performance_rating_leavers.loc[i,'PerformanceRating'])
#            for i in range(len(Performance_rating_leavers))]
# plt.legend(handles=handles)
#
# plt.show()


# # Get the unique values in the 'PerformanceRating' column
# unique_ratings = employeesdf['PerformanceRating'].unique()
#
# # Loop over the unique values and plot each bar with a different color and label
# for rating in unique_ratings:
#     rating_data = employeesdf[employeesdf['PerformanceRating'] == rating]
#     plt.bar(rating_data['PercentSalaryHike'], rating_data['PerformanceRating'],
#             color=sns.color_palette("Pastel1", len(unique_ratings))[int(rating) - 1], label=rating)
# plt.legend(title="Performance Rating")
# plt.show()
#
# plt.bar(employeesdf['PerformanceRating'],employeesdf['PercentSalaryHike'])
# plt.title('Relationship between Salary Hikes and Employee Performance')
# plt.xlabel('Performance Rating %')
# plt.ylabel('Salary Hike %')
# plt.show()



# employees_high_perf = employeesdf[employeesdf['PerformanceRating'].isin([4, 5])]
#
# # Create the countplot
# sns.countplot(x='PerformanceRating', hue='Left', data=employees_high_perf)
#
# # Show the plot
# plt.show()

print(employeesdf['Left'].value_counts())

