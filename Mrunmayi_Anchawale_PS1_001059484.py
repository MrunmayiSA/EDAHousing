#!/usr/bin/env python
# coding: utf-8

# <h1>DS 5230 - PS1 Data Analysis and Preprocessing </h1>
# <h2>Name : Mrunmayi Anchawale</h2>
# Special instructions : Since the same dataset is getting modified in each step of preprocessing in Question 6, execute the statement blocks only once and in order. Please re-read the csv if you get an error.

# <h2>Question 1</h2>
# What is the data type of each feature? (ordinal/nominal/interval/ratio, discrete/continuous)

# In[24]:


import pandas as pd
housing_data_frame = pd.read_csv("C:/Users/Anchawale/Downloads/housing.csv")


# In[25]:


#To view the first few rows of the dataframe
print(len(housing_data_frame))
housing_data_frame.head()


# 
# <h4>Data type of each feature :</h4>
# 1. longitude - continuous interval (Subtracting by 0.01 gives you another longitude. Also, a higher longitudinal or latitudinal value does represent a farther distance)<br>
# 2. latitude - continuous interval (Subtracting by 0.01 gives you another latitude)<br>
# 3. housing_median_age - discrete interval<br>
# 4. total_rooms - discrete ratio<br>
# 5. total_bedrooms - discrete ratio<br>
# 6. population - discrete ratio<br>
# 7. households - discrete ratio<br>
# 8. median_income - continuous interval<br>
# 9. median_house_value - continuous interval<br>
# 10. ocean_proximity - discrete ordinal<br>
# <br>

# <h2> Question 2</h2>
# Display summary statistics of the data. What can you learn from it on the data?

# In[26]:


#To get the summary statistics of the data (numerical attributes)
housing_data_frame.describe()


# In[27]:


#To get the summary statistics of 'ocean_proximity' (categorical attribute)
housing_data_frame["ocean_proximity"].value_counts()


# <h4>What can be learnt from the summary statistics :</h4>
# 25% of the districts have a housing_median_age lower than 18 (fairly new buildings)<br>
# There are atleast 2 total_rooms in each district<br>
# 75% of the districts have households that earn $47,432 average income<br>

# <h2>Question 3</h2>
# Compute the correlation between each feature and the target median_house_value. Which features have strong correlation with the target?
# 

# In[28]:


#Correlation between each feature and the target median_house_value
corr_matrix = housing_data_frame.corr()
corr_matrix["median_house_value"].sort_values(ascending=False)


# In[29]:


import seaborn as sn
sn.heatmap(corr_matrix, annot=True)
plt.show()


# <h4>Which features have strong correlation with the target: </h4>
# The correlation coeffiecient only measures linear correlations. It ranges from -1 to 1. <br>
# When it is close to 1, it means strong positive correlation. When it is close to -1, it means strong negative correlation. <br>
# A value of 0.6 indicates strong positive correlation between <b>median_income</b> and <b>median_housing_value</b>.<br> 
# A value of -0.14 indicates small negative correlation between <b>latitude</b> and <b>median_house_value</b>.<br>

# <h2>Question 4</h2>
# Use data visualization tools to explore the data set. Display at least three different types of graphs.

# In[30]:


#Visualization 1
from pandas.plotting import scatter_matrix
columns = ["median_house_value", "median_income", "total_rooms", "housing_median_age"]
scatter_matrix(housing_data_frame[columns], figsize=(12,8))


# In[31]:


#Visualization 2
import matplotlib.pyplot as plt
housing_data_frame.hist(bins=50, figsize=(15,15))
plt.show()


# In[32]:


#Visualization 3
plt.figure(figsize=(15,10))
plt.scatter(housing_data_frame['longitude'],housing_data_frame['latitude'],
            c=housing_data_frame['median_house_value'],
            s=housing_data_frame['population']/10,
            cmap='viridis')
plt.colorbar()
plt.xlabel('longitude')
plt.ylabel('latitude')
plt.title('house prices in various locations of California')
plt.show()


# The radius of each circle represents the district's population (option s), and the color represents the price (option c). As we can see, prices are very high in coastal areas.<br>
# We can see that Los Angeles (34.052235, -118.243683), San Diego (32.715736, -117.161087), Sacramento (37.773972, -122.431297), Fresno (36.746841, -119.772591) and the Bay Area (37.749997, -122.2833322) are high price areas.

# In[33]:


#Visualization 4
#boxplot of median_house_value on categories under ocean_proximity 
import seaborn as sns
plt.figure(figsize=(10,6))
sns.boxplot(data=housing_data_frame,x='ocean_proximity',y='median_house_value',palette='viridis')
plt.plot()


# <h2>Question 5</h2>
# What type of problems can you detect in the data set? Name at least three different
# problems.

# In[34]:


housing_data_frame.isnull().sum()


# <h4>Problems with the data:</h4>
# 
# 1. The attribute total_bedrooms has 207 missing values.<br>
# 2. The category 'island' in the attribute ocean_proximity has only 5 data points, while other categories have atleast 2200 data points.<br>
# 3. We do not know the currency of median_income, although it should be US dollars since it's California. There is no income above 15 and below 0.5 so looks like values are capped.<br>
# 4. The atributes median_house_value and housing_median_age also seem to be capped<br>
# 5. All attributes have very different scales<br>
# 6. Many histograms in the scattermatrix shown above are skewed towards the right of the median. This imbalance will make it difficult for machine learning algorithms to detect patterns.<br>

# <h2> Question 6 </h2>
# Clean the data set using the data preprocessing techniques discussed in class. Show a
# sample of the data set before and after the cleaning.
# 

# In[35]:


#Distribution of data before cleaning
housing_data_frame.head()


# In[36]:


housing_data_frame.describe()


# In[37]:


#Imputation of Missing Data for total_bedrooms 
housing_data_frame["total_bedrooms"].fillna(housing_data_frame["total_bedrooms"].median(), inplace=True)
housing_data_frame.describe()


# As seen above, the mean and quantile values for total_bedrooms have changed a bit after imputation of missing values

# In[38]:


#Visualizing the skewness in data
housing_data_frame.skew()


# In[39]:


#Removing skewdness
import numpy as np
housing_data_frame["total_rooms"] = np.log(housing_data_frame["total_rooms"])
housing_data_frame["total_bedrooms"] = np.log(housing_data_frame["total_bedrooms"])
housing_data_frame["population"] = np.log(housing_data_frame["population"])
housing_data_frame["households"] = np.log(housing_data_frame["households"])
housing_data_frame["median_income"] = np.log(housing_data_frame["median_income"])
print(housing_data_frame.skew())
housing_data_frame.hist(bins=50, figsize=(15,15))
plt.show()


# The distribution shifts towards normal after removing skewdness

# In[40]:


#Outlier removal
Q1 = housing_data_frame.quantile(0.25)
Q3 = housing_data_frame.quantile(0.75)
IQR = Q3 - Q1
housing_data_frame = housing_data_frame[~((housing_data_frame < (Q1 - 1.5 * IQR)) |(housing_data_frame > (Q3 + 1.5 * IQR))).any(axis=1)]
print(housing_data_frame.shape)


# As seen above, there were a total of 20640 observations, out of which 18320 are left after outlier removal.

# In[41]:


#Encoding for the categorical attribute 'ocean_proximity'
#Using Label encoder
import numpy as np
from sklearn.preprocessing import LabelEncoder
encoder = LabelEncoder()
housing_data_frame["ocean_proximity_cat"] = encoder.fit_transform(housing_data_frame["ocean_proximity"])

#Using OneHotEncoder on top of LabelEncoder
from sklearn.preprocessing import OneHotEncoder
encoder = OneHotEncoder(handle_unknown='ignore')
housing_ocean_1henc = pd.DataFrame(encoder.fit_transform(housing_data_frame[["ocean_proximity_cat"]]).toarray(), columns=['<1H OCEAN', 'INLAND', 'ISLAND', 'NEAR BAY', 'NEAR OCEAN'])
housing_data_frame = housing_data_frame.join(housing_ocean_1henc)
housing_data_frame.head()


# In[42]:


#Feature Scaling

#Excluding the columns that don't need feature scaling
housing_data_frame1 = housing_data_frame.drop(["latitude","longitude","ocean_proximity", "ocean_proximity_cat", "<1H OCEAN", "INLAND", "ISLAND", "NEAR BAY", "NEAR OCEAN"], axis=1)
#Scaling
from sklearn.preprocessing import StandardScaler
trans = StandardScaler()
housing_data_frame1 = trans.fit_transform(housing_data_frame1)
dataset = pd.DataFrame(housing_data_frame1)
dataset.hist()
plt.show()


# In[43]:


#Sample of dataset after cleaning
housing_data_frame.head()


# In[44]:


housing_data_frame.describe()


# <h2>Question 7</h2>
# Extract at least two new features from the data set that have strong correlation with the target feature.   

# In[45]:


#Creating new features

#New features : Self-explanatory
housing_data_frame["rooms_per_household"] = housing_data_frame["total_rooms"]/housing_data_frame["households"] 
housing_data_frame["bedrooms_per_house"] = housing_data_frame["total_bedrooms"]/housing_data_frame["total_rooms"] 
housing_data_frame["population_per_household"] = housing_data_frame["population"]/housing_data_frame["households"]
housing_data_frame["income per working population"]=housing_data_frame['median_income']/(housing_data_frame['population']-housing_data_frame['households'])

#New features : 'building_age_cat' that shows building's age as new, mid old or old
def type_building(x):
    if x<=10:
        return "new"
    elif x<=30:
        return 'mid old'
    else:
        return 'old'

housing_data_frame["median_building_age"] = housing_data_frame.apply (lambda row: type_building(row.housing_median_age), axis=1)
encoder = LabelEncoder()
housing_data_frame["building_age_cat"] = encoder.fit_transform(housing_data_frame["median_building_age"])
from sklearn.preprocessing import OneHotEncoder
encoder = OneHotEncoder(handle_unknown='ignore')
housing_age_1henc = pd.DataFrame(encoder.fit_transform(housing_data_frame[["building_age_cat"]]).toarray(), columns=['mid old', 'new', 'old'])
housing_data_frame = housing_data_frame.join(housing_age_1henc)
housing_data_frame.head()

#Central : 37.16611 -119.44944
#North Coast : 38.0193 -122.8631
#Bay Area : 37.8272 -122.2913
#Southern: 34.9592 -116.4194


# In[47]:


#Re-checking the correlation after adding new columns
corr_matrix = housing_data_frame.corr()
corr_matrix["median_house_value"].sort_values(ascending=False)


# After preprocessing, 'income per working population' has the highest correlation with median_house_value at 0.64<br>
# The new column population_per_household has a -0.26 negative correlation with median_house_value. This means that the houses with a lower population_per_household ratio tend to be more expensive. <br>
# Although the correlation between rooms_per_household and median_house_value is very small (0.10), it indicates that larger the houses, the more expensive they are.

# In[ ]:




