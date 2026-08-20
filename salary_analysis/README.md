# README

## Introduction
According to the U.S. Bureau of Labor Statistics (BLS), data science employment is projected to expand by 36% between 2023 and 2033. This rapid market growth is heavily driven by the permanent shift toward remote and hybrid work models, which accelerated a massive influx of virtual job openings across the data sector. This project analyzes those shifting employment dynamics to identify key trends in the modern tech workforce.

The primary goal of this analysis is to lower the barrier to entry for job seekers breaking into the data space.To achieve this, the project evaluates hiring patterns across the top three data domains, specifically isolating the impact of:

    1.) Work setting
    2.) Experience level

For the analysis, we'll be using the jobs_in_data dataset gathered from Kaggle:
    
   [https://www.kaggle.com/datasets/hummaamqaasim/jobs-in-data/data]()

## Data Preprocessing
First, we'll import the necesary libraries that we'll be using in the analysis.

```python
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
```
Next, we'll load the dataset that we'll be using and print the first five rows. We'll be limiting our analysis for the companies located in the United States.

```python
#Load the dataset
df = pd.read_csv('jobs_in_data.csv')

#For this analysis, we consider only companies located in the United States.
df_US = df[df['company_location'] == 'United States']
df_US.head()
```
![df_US_head table visualization](images\df_US_head.png)


Getting initial insights in our dataset.

```python
#Getting initial insights in our data.
df_US.info()
```

    <class 'pandas.core.frame.DataFrame'>
    Index: 8132 entries, 1 to 9354
    Data columns (total 12 columns):
    #   Column              Non-Null Count  Dtype 
    ---  ------              --------------  ----- 
    0   work_year           8132 non-null   int64 
    1   job_title           8132 non-null   object
    2   job_category        8132 non-null   object
    3   salary_currency     8132 non-null   object
    4   salary              8132 non-null   int64 
    5   salary_in_usd       8132 non-null   int64 
    6   employee_residence  8132 non-null   object
    7   experience_level    8132 non-null   object
    8   employment_type     8132 non-null   object
    9   work_setting        8132 non-null   object
    10  company_location    8132 non-null   object
    11  company_size        8132 non-null   object
    dtypes: int64(3), object(9)
    memory usage: 825.9+ KB

Fortunately, our dataset's already clean and has no missing values. Hence, we can now start the analysis.

# The Analysis
As defined in the project scope, this study focuses exclusively on the top three data job categories by volume within the United States market. In the following step, we will filter the dataset by geographic location (company_location == 'United States') and execute a frequency count on job_category to programmatically identify our primary analytical cohorts.

```python
#Get the top 3 data job categories in the United States
df_US['job_category'].value_counts().head(3)

#Save the top 3 data job categories to a variable
top_3_job = df_US['job_category'].value_counts().head(3).index.to_list()
top_3_job
```
    job_category
    Data Science and Research    2635
    Data Engineering             1977
    Data Analysis                1252
    Name: count, dtype: int64

For the detailed steps of every plots/graphs shown, please see my notebook here: [Step by Step Analysis Notebook](notebook.ipynb)

## 1.) Mean salary of the Top 3 Data Job Categories on Different Work Settings

The post-pandemic landscape fundamentally altered candidate priorities, transforming workplace flexibility from a fringe perk into a core negotiating pillar. This section explores the interplay between candidate preferences and employer compensation strategies. Specifically, we evaluate how average salaries (salary_in_usd) vary across different work_setting models (Remote, Hybrid, On-site) within each of our top three data cohorts to determine if choosing flexibility incurs a financial trade-off.

To find the mean salary of top 3 data job categories, I filtered the 'job_category' column to the said categories, then utilized pandas groupby to finally calculate for the mean salary.

### Visualize the Data
```python
fig, ax = plt.subplots(3,1)
sns.set_style(style='ticks')
for i,job in enumerate(top_3_job):
    mean_work_setting_plot = mean_work_setting[mean_work_setting['job_category'] == job]
    sns.barplot(data=mean_work_setting_plot, y='work_setting', x='mean_salary', legend=False, ax=ax[i], palette='Set2', hue='mean_salary')
    ax[i].set_ylabel('')
    ax[i].set_xlim(0,200000)
    ax[i].set_xlabel('')
    ax[i].set_title(job)
    ax[i].xaxis.set_major_formatter(plt.FuncFormatter(lambda x,pos: f'${int(x/1000)}K'))
plt.suptitle('Mean Salary of Top 3 Data Jobs Based on Work Setting in the United States')
plt.tight_layout()
plt.show()
```
![Data Visualization](images\mean_work_setting.png)

### Insights
- Data Science & Research: In-person positions lead with a peak mean salary of ~$175K. Remote and hybrid models follow closely, showing strong salary resilience despite increased flexibility.
- Data Engineering: Demonstrates a distinct market preference for decentralized talent, with Remote work settings outperforming all other modalities in average pay.
- Data Analysis: Consistently places at the lower end of the compensation spectrum, showing minimal variation across different work settings.
- Cross-Category Hybrid Trend: Interestingly, Hybrid environments register the lowest mean compensation across the entire cohort, suggesting unique market forces affecting split-schedule roles.

### Deep dive on to what causing why Hybrid work setting mean salary to be the lowest  
Many job seekers prefer Hybrid work setting due to its convenience and cost-efficiency. We'll dive deeper on to the reason why Hybrid work setting has the lowest mean salary in all data job categories.

We'll compute the distribution of the experience level in the top 3 data job categories for every work settings. (For detailed steps, please look at the notebook [using this link](notebook.ipynb).

### Visualize the data
```python
df_US_distribution = df_US_top3.groupby(['job_category', 'work_setting', 'experience_level'],dropna=False).size()
df_US_distribution = df_US_distribution.reset_index(name='job_count')
df_US_distribution = df_US_distribution[df_US_distribution['job_category'].isin(top_3_job)]
setting_work = ['Hybrid', 'In-person', 'Remote']
for work in setting_work:
    df_US_dist_plot = df_US_distribution[df_US_distribution['work_setting'] == work]
    df_US_dist_plot = df_US_dist_plot.set_index('job_category')

    fig,ax = plt.subplots(1,3, figsize=(20,15))
    for i,job in enumerate(top_3_job):
        ax[i].pie(data=df_US_dist_plot.loc[job], x='job_count', labels=df_US_dist_plot.loc[job]['experience_level'], autopct='%1.1f%%', textprops={'fontsize': 13})
        ax[i].set_title(job, fontsize=15)
        ax[i].legend(loc='upper right', bbox_to_anchor=(1.2,0.5))
    plt.suptitle(f'Distribution of Experience Level in {work} Work Setting for Top 3 Data Job Titles', y=0.75, fontsize=20)
```
![Wahah](images/df_dist_hybrid.png)
![Wahah](images/df_dist_inperson.png)
![Wahah](images/df_dist_remote.png)

### Insights
- Hybrid models are heavily populated by junior-level experience tiers, dragging down the segment's overall mean salary.
- Data Analysis functions primarily as an entry-level category compared to the other two domains, and it relies heavily on hybrid schedules.
- Remote and In-Person settings are anchored by Senior-level professionals. Companies grant these modalities to tenured employees who operate independently and mentor junior staff.
- The high concentration of junior talent in hybrid frameworks indicates that companies prefer utilizing split schedules for foundational training before transitioning employees into higher-autonomy work settings.

## 2.) Mean salary for top 3 data job categories based on experience level

Some applicants care not much about the work setting. In this case, they might want to consider how well different job categories pay based on experience level.

To find the mean salary of top 3 data job categories, I filtered the 'job_category' column to the said categories, then utilized pandas groupby to finally calculate for the mean salary for different experience levels.

For detailed steps, please check my notebook [using this link](notebook.ipynb).


### Visualize the data
```python
fig, ax = plt.subplots(3,1)
sns.set_style(style='ticks')
for i,job in enumerate(top_3_job):
    mean_exp_level_plot = mean_exp_level[mean_exp_level['job_category'] == job]
    sns.barplot(data=mean_exp_level_plot, y='experience_level', x='mean_salary', legend=False, ax=ax[i], palette='Set2', hue='mean_salary')
    ax[i].set_ylabel('')
    ax[i].set_xlim(0,225000)
    ax[i].set_xlabel('')
    ax[i].set_title(job)
    ax[i].xaxis.set_major_formatter(plt.FuncFormatter(lambda x,pos: f'${int(x/1000)}K'))
plt.suptitle('Average Salary of Top 3 Data Job Categories Based on Experience Level')
plt.tight_layout()
plt.show()
```
![Mean for different experience level](images\mean_exp_level.png)

### Insights
- The Data Analysis track holds the lowest average salary at every career stage (Entry, Mid, Senior, Executive).
    - This trend aligns with industry realities. Positions designated as "Entry-Level" within advanced categories like Data Science are not entry-level in the traditional sense. Because these roles demand foundational expertise right from the start, their baseline compensation begins at a significantly higher tier compared to standard early-career roles.
- Data Analysis category has the lowest mean salary in every experience level.
    - Interestingly, Senior analysts out-earn Executive-level profiles in this dataset.
    - Industry research suggests this trend stems from title compression. Many companies rely on Senior Analysts to absorb executive-level strategic ownership and cross-functional leadership while retaining an individual contributor title, driving their specialized compensation past early managerial or executive baselines.
- Data Analysis category has also lowest salary growth.
    - Job seekers currently in this category might want to consider shifting into different category as salary progression is much higher.

## Are these categories paid well?
While all the shown graphs reflect the average salary, sometimes they do not reflect the whole picture. Here, we'll explore how well the top 3 data job categories paid based on experience level.

For detailed steps, please check my notebook [using this link](notebook.ipynb).

### Visualize the data
```python
experience = ['Entry-level', 'Mid-level', 'Senior', 'Executive']
sns.set_style('ticks')
for job in top_3_job:
    df_US_DA = df_US_top3[df_US_top3['job_category'] == job]
    DA_list = [df_US_DA[df_US_DA['experience_level'] == exp]['salary_in_usd'] for exp in experience]
    plt.boxplot(DA_list, tick_labels=experience, vert=False)
    plt.title(f'Salary Distribution of {job} in the United States')
    ax = plt.gca()
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x,pos: f'${int(x/1000)}K'))
    plt.show()
```
![whah](images\salary_dist_DS.png)
![whah](images\salary_dist_DE.png)
![whah](images\salary_dist_DA.png)

### Insights
- Data Science and Research
    - The distribution shows an absence of extreme low-end outliers, indicating high industry baselines where early-career professionals are well-compensated out of the gate.
    - Executive roles claim the highest median salary, though they sit only marginally above the Senior-level bracket.
    - Select Senior-level outliers surpass the absolute maximum of the Executive distribution. This points to the existence of highly specialized individual contributor (IC) roles that command premium market rates for technical expertise.
    - A subset of Mid-level professionals out-earns the Senior-level median, suggesting that corporate titles in this domain do not always perfectly align with a candidate's actual scope of responsibility or leverage.
    - Both Entry and Mid-level brackets exhibit compact distributions, signaling highly structured and standardized corporate pay scales for non-senior talent.

- Data Engineering
    - While compensation remains strong across most brackets, the Senior cohort features a unique downward outlier anomaly—revealing a small subset of senior positions paying significantly below market norms.
    - Executives yield the highest median salary. Unlike the Data Science track, the Executive distribution here contains no outliers, indicating a stable and predictable ceiling for management compensation.
    - Despite its low-end anomalies, the Senior bracket contains the highest volume of upward outliers, frequently eclipsing Executive-level pay scales.
    - The Mid-level tier displays a highly balanced, symmetric distribution, flanked by strong positive outliers on the upper extremity.
    - Although Entry-level median pay rests on the lower end, the overall distribution is heavily skewed toward the higher side, indicating strong upward mobility and aggressive negotiation potential for top junior talent.

- Data Analysis
    - Similar to Data Science, the lack of extreme low-end datapoints suggests an industry-wide baseline that prevents severe underpayment at any tier.
    - The boxplots across Entry, Mid, and Senior levels are noticeably compact. This tight variance indicates highly rigid, standardized salary bands across the analytics vertical.
    - Executive median pay is only fractionally higher than Senior-level pay, resulting in a massive distributional overlap where a significant portion of Senior Analysts out-earn their Executive counterparts. This underscores scenarios where experienced individual contributors absorb strategic leadership without a formal title change.
    - Select Mid-level outliers stretch past the Senior median, highlighting specialized analytics skill sets (e.g., domain-specific data modeling) that command a distinct market premium.
    - Entry-level roles register the lowest overall compensation and the most compressed distribution, indicating strict corporate salary ceilings for baseline analytics onboarding.