# GCIC-Background-Behavior
The objective of this study is to investigate whether behavioral or background variables can predict student success better. While universities are putting resources into helping students succeed, they are collecting limited insights of those factors. This research employs OULAD - Open University Learning Analytics Dataset as the primary source for data since they provide a public dataset containing data from 32,593 students in 22 online course presentations. On one hand, background factors include student age, level of education, disability status and neighbourhood socioeconomic deprivation. On the other hand, behavioral factors were derived from more than 10 million data records in the dataset including students' activities such as online participation, active study days, early participation, assignment submission timing, and assessment scores. Eventually, success was defined as the passing or distinction in a course. This analysis considers three different classification techniques (logistic regression, random forest, and XGBoost), which were trained on either background variables or behavioral variables or their combination. SHAP values and a decision tree were also used to identify the factors that are the most influential ones and provide interpretation of the results as success patterns in common language. K-means cluster analysis is performed to test whether the trajectory of the successful students is different. Moreover, models that were trained using the early course data are employed in order to test when successful outcomes can be predicted. We also check whether the engagement bridges the gap between the students from areas with different socioeconomic statuses.


**Overview
**This project tests whether what students do (study behavior) predicts course success better than who they are (background), using 32,593 real online students from the Open University (UK).
Working title: Background or Behavior? A Machine Learning Profile of Successful Students
Research questions
Which student characteristics are most strongly linked to passing a course?
Do behavior factors (engagement, submission habits, early scores) predict success better than background factors (age, prior education, neighborhood deprivation)?
Are there distinct "paths to success" among students who pass?
How early in a course can success be predicted with reasonable accuracy?
**Hypotheses
H1: A behavior-only model will predict success more accurately than a background-only model.
H2: Students from more deprived areas pass at lower rates, but the gap shrinks among students with similar engagement.
H3: Engagement in the first four weeks already separates most successful students from unsuccessful ones.
Why it matters: If behavior outweighs background, colleges can focus support on habits that students and advisors can change, and flag students early for help.
The dataset
The Open University Learning Analytics Dataset (OULAD) covers 32,593 students across 22 course presentations (7 modules, 2013–2014), with 10.6 million daily click records. It is free under a CC BY 4.0 license, anonymized, and comes as 7 linked CSV files (Kuzilek et al., 2017).
