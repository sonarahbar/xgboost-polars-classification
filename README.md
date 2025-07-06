# Maximizing Client Well-being: Proactive Funding Engagement Through Machine Learning

Within wellbeing-focused service organisations, clients are often allocated funding to support their mental health, physical wellbeing, or personal development. Despite the availability of these resources, a considerable proportion of allocated funds remains unspent, frequently due to a lack of timely engagement rather than an absence of need. This underutilisation not only limits the effectiveness of support programs but also represents a missed opportunity for clients to access services that could significantly improve their quality of life.

To address this challenge, we developed a machine learning model designed to identify clients who may require a timely reminder regarding their available funding. The model flags individuals based on their historical funding utilisation and service engagement patterns, enabling targeted and proactive outreach to those most at risk of leaving funds unutilised. This data-driven approach supports service providers in facilitating more effective use of allocated resources and enhancing overall client outcomes.

---

# 1. Methodology

To develop a reliable and ethically responsible solution for identifying clients at risk of leaving allocated funds unspent, we employed a comprehensive methodology grounded in **data simulation, machine learning**, and **privacy-conscious design**.

## 1.1 Data Simulation and Ethical Safeguards

Given the sensitivity of client data, a **representative mock dataset** was created to emulate the structure and behavioral patterns of real-world data, including trends in funding allocation, utilization, and service engagement.

* **Privacy and Security:** No identifiable or sensitive client data was used. By relying exclusively on simulated data, we ensured compliance with data protection standards and maintained client confidentiality throughout the model development lifecycle.

* **Representativeness:** The mock dataset was carefully engineered to preserve the statistical characteristics and anomalies of the original data, enabling accurate model training and realistic scenario testing.

* **Reproducibility:** Working with synthetic data enables consistent model evaluation and reproducibility, while allowing for iterative experimentation without risking data breaches.

## 1.2 Machine Learning Model Development

The central objective was to build a model capable of predicting which clients are likely to leave their funding unspent, enabling timely and targeted interventions.

* **Feature Engineering:** Key features were derived to capture behavioral and funding-related signals, including historical utilization trends, frequency of service access, recency of engagement, and available fund balance.

* **Model Selection:** Various classification models were tested, including **Logistic Regression**, **Random Forests**, and **Gradient Boosting Machines** (e.g., LightGBM and XGBoost). Model selection was guided by predictive accuracy, interpretability, and computational efficiency.

* **Training and Evaluation:** Models were trained on the mock dataset and evaluated using a hold-out validation set. Performance metrics included **precision**, **recall**, **F1-score**, and **AUC**, ensuring robust generalization and minimizing overfitting.

## 1.3 Technology Stack

* **Data Processing:** The **Polars** library was used for efficient and scalable data manipulation. Its speed and memory efficiency significantly accelerated feature engineering and model iteration.

* **Modeling Tools:** Python-based libraries including **scikit-learn**, **XGBoost**, and **LightGBM** were utilized for algorithm development, while **matplotlib** and **seaborn** supported exploratory data analysis and result visualization.

## 1.4 Targeted Outreach Strategy

The output of the model—a risk score or classification flag—enables targeted action by service providers:

* **Proactive Reminders:** High-risk clients can be proactively informed of their available funding and encouraged to engage with relevant services.

* **Personalized Engagement:** Communication strategies can be tailored based on client-specific behavior patterns, improving the likelihood of response and service uptake.

* **Impact Tracking:** The system facilitates monitoring of post-intervention outcomes, helping to assess and refine the effectiveness of outreach strategies over time.

---

# 2. Business Objective

A machine learning model was developed to help reduce unspent funds by identifying clients who are likely to begin using a new service. It not only predicts which clients are most likely to engage, but also determines the optimal timing to initiate contact. This enables weekly prioritisation of outreach, allowing staff to engage clients in timely, targeted conversations about services that could benefit them. Ultimately, the model supports more effective use of available funding by promoting proactive and personalised service delivery.

## 2.1 Go-Live Threshold Recommendation

We recommend setting the operational threshold at 0.7 for prediction probability scores to maintain a high level of confidence in identifying clients likely to uptake a new service. This threshold strikes a balance between precision and workload, helping advisors focus on the most promising opportunities while minimising the impact of false positives.

---

# 3. Target Definition

The prediction target for this model is a binary variable, **`new_service_flag`**, which indicates whether a client is likely to uptake a *new* service in the following week (`1 = new service introduced`, `0 = no new service`). This variable is derived from the **`billing_category_service_type`** column, which categorises the services each client receives.

## 3.1 Definition of "New Service"

A **new service** is defined as any service appearing in the client’s weekly service data that did *not* exist in their unique set of services over the previous four weeks.

* If all services in the current week have been seen in the past four weeks → `new_service_flag = 0`
* If at least one service is new → `new_service_flag = 1`

We opted for a binary classification approach due to data limitations that made more complex recommendation models impractical. Our goal is to predict **who** to engage and, more importantly, **when** to engage them, rather than predicting the exact service they will take up.

## 3.2 Step-by-Step Summary

1. **Load Weekly Data**: Include client IDs, week-start dates, and service lists for each week.
2. **Extract Weekly Services**: Identify unique services for each client-week.
3. **Aggregate Historical Services**: For each client, gather all unique services from the past 4 weeks.
4. **Compare Services**:

   * If all current services were present in the previous 4 weeks → set `new_service_flag = 0`
   * If any current service is new → set `new_service_flag = 1`
5. **Create Binary Target**: Add the result as a new column called `target`.
6. **Save Data**: Export the enriched dataset for model training.

---

# 4. Experimental Setup

## 4.1 Dataset Construction

To support model development, we first constructed an event-level dataset where each row captured a client interaction along with its timestamp. These events were then aggregated into a **weekly log per client**, representing each client’s service journey over time.

## 4.2 Temporal Split

The dataset was split into training and testing sets to reflect a real-world deployment scenario and prevent data leakage.

* **Training period**: January 2023 to August 2024
* **Testing period**: September 2024 to March 2025

This setup allowed the model to focus on **identifying timely upsell opportunities**, helping to guide proactive engagement based on the likelihood of new service uptake.

---

# 5. Baseline Model and Insights

## 5.1 Algorithm

We began with a baseline model using the default configuration of the XGBoost binary classification algorithm, without any hyperparameter tuning. This simple model served as an initial benchmark to assess the predictive signal within the data.

## 5.2 Initial Metrics

The baseline model demonstrated a reasonable ability to identify clients likely to uptake a new service, providing a useful starting point for further iteration. While the results were encouraging, they also highlighted opportunities for improvement, particularly in optimising precision, recall, and the balance between false positives and false negatives.

## 5.3 Learnings

* Even without advanced feature engineering or hyperparameter tuning, the baseline model achieved reasonable predictive performance.
* The model was effective at identifying **who** should be targeted for service upsell conversations, but less effective at predicting **when** those interactions should take place.
* To enhance the model, the next steps included incorporating time-based features (e.g. lagged service data), tuning model parameters, and training multiple models across different time horizons to better capture the timing of client engagement opportunities.

---

# 6. Final Model

## 6.1 Feature Enhancements

In the final model, we introduced a set of engineered features designed to capture temporal patterns and strengthen the predictive signal:

* **Lagged Features**: These included trailing aggregates over recent time windows—such as the number and types of services accessed, frequency of visits, and total spend.

## 6.2 Algorithm and Tuning

We selected a gradient-boosted decision tree model using the XGBoost framework, known for its strong performance on structured datasets and ability to model complex, non-linear relationships.

To optimise performance, we used FLAML’s AutoML framework, which automatically tuned the model’s hyperparameters using a cost-aware optimisation strategy.

The final solution consisted of an ensemble of models trained over different future timeframes. This **multi-horizon modelling** approach enabled the system not only to predict who is likely to engage, but also to recommend the best timing for outreach.

## 6.3 Cross-Validation Strategy

To ensure stability and generalisability, we used cross-validation across multiple time-based slices. This helped verify that model performance held up consistently as client behaviour evolved over time.

We also monitored **data drift** using the Population Stability Index (PSI). PSI values remained within acceptable limits, suggesting overall model stability. A slightly elevated PSI observed during the final testing window indicated a modest shift in patterns, which may require further monitoring or eventual retraining if the trend continues.

---

# 7. Model Evaluation

## 7.1 Multi-Horizon Modeling

Each model was trained with a **shifted target**, asking:

> *“If we were to engage a client this week, how likely are they to take up a new service in the next few weeks?”*

This enabled both **reactive** and **proactive** engagement strategies.

## 7.2 Calibration Output

To evaluate how well the model’s predicted probabilities align with actual outcomes, we assessed its **calibration** using:

* **Brier Score**: Measures the accuracy of probabilistic predictions.
* **Expected Calibration Error (ECE)**: Captures the difference between predicted probabilities and actual outcomes.

The model predicting immediate service uptake exhibited the strongest calibration, meaning its probability estimates closely reflected real-world outcomes. As the time horizon extended, calibration performance declined slightly but remained within acceptable bounds. This indicates that even over longer forecast windows, the model’s confidence levels continued to provide meaningful guidance for decision-making.

## 7.3 Interpretability

We employed **SHAP (SHapley Additive exPlanations)** to ensure the model remains transparent and explainable:

* **Global insights** show top contributing features across the dataset.
* **Local explanations** reveal why a particular client was flagged.
* **Integration with XGBoost** allows seamless interpretation.

---

# 8. Robustness Tests

For robustness testing, we conducted cross-validation (Section 6.3) to assess how well the model performs over different time periods. However, more in-depth testing—such as evaluating edge cases, analysing performance across different client groups, or detecting subtle data shifts—would require access to a larger and more reliable dataset.

This extended testing would help ensure the model remains stable and accurate under a wider range of real-world conditions and client scenarios.

