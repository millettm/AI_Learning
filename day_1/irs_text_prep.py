import nltk
import pandas as pd
nltk.download('punkt')  # downloads tokenizer
nltk.download('punkt_tab')

# Sample text
text = """
6.430.1.5 (01-01-2007)
Key Performance Management Activities
In effective organizations, managers and employees have been practicing good performance management naturally all their 
lives, executing each key component process well. Goals are set and work is planned routinely. Progress toward 
those goals is measured and employees get feedback. High standards are set, but care is also taken to develop the 
skills needed to reach them. Formal and informal rewards are used to recognize the behavior and results that 
accomplish the mission. All four component processes working together and supporting each other achieve natural, 
effective performance management.
The four steps of IRS performance management are:
Planning expectations
Monitoring progress
Evaluating performance
Recognizing performance

6.430.1.5.1 (01-01-2007)
Planning Expectations
Planning means setting performance expectations and goals for groups and individuals to channel their efforts 
toward achieving organizational objectives. Getting employees involved in the planning process will help them 
understand the goals of the organization, what needs to be done, why it needs to be done, and how well it should 
be done. In an effective organization, work is planned out in advance.
The regulatory requirements for planning employees' performance include establishing the critical elements and 
performance standards of their performance plans.
Performance elements and standards should be measurable, understandable, verifiable, equitable, and achievable. 
Through critical elements, employees are held accountable as individuals for work assignments or responsibilities.
Employee performance plans should be flexible so that they can be adjusted for changing program objectives and 
work requirements. When used effectively, these plans can be beneficial working documents that are discussed often, 
and not merely paperwork that is filed in a drawer and seen only when ratings of record are required.
6.430.1.5.2 (01-01-2007)
Monitoring Progress
Monitoring well means consistently measuring performance and providing ongoing feedback to employees and work groups on 
their progress toward reaching their goals. In an effective organization, assignments and projects are monitored 
continually.
Regulatory requirements for monitoring performance include conducting progress reviews with employees where their 
performance is compared against their elements and standards.
Ongoing monitoring provides the opportunity to check how well employees are meeting predetermined standards and to 
make changes to unrealistic or problematic standards. And by monitoring continually, unacceptable performance can 
be identified at any time during the appraisal period and assistance provided to address such performance rather 
than wait until the end of the period when summary rating levels are assigned.
6.430.1.5.3 (01-01-2007)
Evaluating Performance
From time to time, organizations find it useful to summarize employee performance. This can be helpful for looking 
at and comparing performance over time or among various employees. Organizations need to know who their best 
performers are.
Within the context of formal performance appraisal requirements, rating means evaluating employee or group 
performance against the critical elements and performance standards in an employee's performance plan and 
assigning a summary rating of record. The rating of record is assigned according to procedures included in the 
Service’s appraisal program and is based on work performed during an entire appraisal period.
The rating of record has a bearing on various other personnel actions, such as granting within-grade pay increases 
and determining additional retention service credit in a reduction in force.
6.430.1.5.4 (01-01-2007)
Recognizing Performance
Rewarding means recognizing employees, individually and as members of groups, for their performance and 
acknowledging their contributions to the agency's mission. In an effective organization, rewards are used well.
Another critical component of recognizing performance is the supervisor's obligation to address employee 
performance that does not meet performance expectations. Most performance problems can be resolved through 
effective communication and counseling between the supervisor and the employee. Addressing the need for 
improvement as it is observed will often correct poor performance before a performance-based action is necessary.
Good performance is recognized without waiting for nominations for formal awards to be solicited. Recognition is 
an ongoing, natural part of day-to-day experience. A lot of the actions that reward good performance — like 
saying "Thank you" — do not require a specific regulatory authority.
Award regulations provide a broad range of forms that more formal rewards can take, such as cash, time off, and 
many nonmonetary items. The regulations also cover a variety of contributions that can be rewarded, from 
suggestions to group accomplishments.
"""

# Tokenize
sentences = nltk.sent_tokenize(text)
words = nltk.word_tokenize(text)

df = pd.DataFrame({'sentences': sentences, 'word_count': [len(nltk.word_tokenize(s)) for s in sentences]})
print(df)

df.to_csv('C:/Users/the-s/PycharmProjects/AI_Learning/data/perf_mgt_sample.csv', index=False)