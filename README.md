# xgboost-polars-classification

## Maximizing Client Well-being: Proactive Funding Engagement Through Machine Learning
Many clients are allocated funding to support their wellbeing, health, or personal development, but a significant portion of these funds often goes unspent due to a lack of timely engagement. This underutilization not only reduces the impact of these programs but also represents a missed opportunity for clients to access valuable services.

To address this issue, we developed a machine learning model to identify clients who may require a timely reminder about their available funding. The model flags individuals based on their historical funding usage and service engagement patterns—enabling targeted outreach to those most at risk of leaving funds unused. For instance, a wellbeing provider like Smiling Mind (a mental health and mindfulness organization) might use such a system to ensure clients make full use of their allocated support resources.

To support safe and ethical model development, we created a representative mock dataset that mirrors the structure and behavioural dynamics of the original data. This approach ensures that no sensitive client information is exposed, while still enabling robust model training and evaluation.

### 1. Business Objective
A machine learning model was developed to help reduce unspent funds by identifying clients who are likely to begin using a new service. It not only predicts which clients are most likely to engage, but also determines the optimal timing to initiate contact. This enables weekly prioritisation of outreach, allowing staff to engage clients in timely, targeted conversations about services that could benefit them. Ultimately, the model supports more effective use of available funding by promoting proactive and personalised service delivery.

### Go-Live Threshold Recommendation
We recommend setting the operational threshold at 0.7 for prediction probability scores to maintain a high level of confidence in identifying clients likely to uptake a new service. This threshold strikes a balance between precision and workload, helping advisors focus on the most promising opportunities while minimising the impact of false positives.


### 2. Target Definition

The prediction target for this model is a binary variable, **`new_service_flag`**, which indicates whether a client is likely to uptake a *new* service in the following week (`1 = new service introduced`, `0 = no new service`). This variable is derived from the **`billing_category_service_type`** column, which categorises the services each client receives.

To construct the target, we aggregated service data into weekly snapshots per client. For each week, the client’s service list is compared to their service history over the preceding four weeks. If the current week's services include at least one not seen in the prior four weeks, the week is flagged as containing a new service.

#### Definition of "New Service"

A **new service** is defined as any service appearing in the client’s weekly service data that did *not* exist in their unique set of services over the previous four weeks.

* If all services in the current week have been seen in the past four weeks → `new_service_flag = 0`
* If at least one service is new → `new_service_flag = 1`

We opted for a binary classification approach due to data limitations that made more complex recommendation models impractical. Our goal is to predict **who** to engage and, more importantly, **when** to engage them, rather than predicting the exact service they will take up.

#### Step-by-Step Summary

1. **Load Weekly Data**: Include client IDs, week-start dates, and service lists for each week.
2. **Extract Weekly Services**: Identify unique services for each client-week.
3. **Aggregate Historical Services**: For each client, gather all unique services from the past 4 weeks.
4. **Compare Services**:

   * If all current services were present in the previous 4 weeks → set `new_service_flag = 0`
   * If any current service is new → set `new_service_flag = 1`
5. **Create Binary Target**: Add the result as a new column called `target`.
6. **Save Data**: Export the enriched dataset for model training.

Here's a refined version of your **3.1 Experimental Setup** section. The flow has been improved for clarity, conciseness, and professional tone while retaining all the technical details:

---

### 3. Experimental Setup

To support model development, we first constructed an event-level dataset where each row captured a client interaction along with its timestamp. These events were then aggregated into a **weekly log per client**, representing each client’s service journey over time.

The dataset was split into training and testing sets to reflect a real-world deployment scenario and prevent data leakage.

* **Training period**: January 2023 to August 2024
* **Testing period**: September 2024 to March 2025

This temporal split ensures that the model is evaluated on future data and not exposed to information that would not be available at the time of prediction.

The problem was framed as a **binary classification task**, with the target variable indicating whether a client would start using a **new service** in the following week. For each client-week, we compared the services used in the next week against those used in the previous four weeks (28 days).

* If at least one service in the following week was **new** (i.e. not present in the prior 4-week window), the target was set to **1**.
* If all services were already known or the client used no services, the target was set to **0**.

This setup allowed the model to focus on **identifying timely upsell opportunities**, helping to guide proactive engagement based on the likelihood of new service uptake.

Here’s a refined version of **3.3 Algorithm** and **3.4 Initial Metrics**, with improved clarity and tone, and all numerical values removed as requested:

---

### 3.1 Algorithm

We began with a baseline model using the default configuration of the XGBoost binary classification algorithm, without any hyperparameter tuning. This simple model served as an initial benchmark to assess the predictive signal within the data. Although basic, it allowed us to establish a reference point for evaluating the value of more advanced modeling techniques, feature engineering, and optimisation strategies in later stages.

---

### 3.2 Initial Metrics

The baseline model demonstrated a reasonable ability to identify clients likely to uptake a new service, providing a useful starting point for further iteration. While the results were encouraging, they also highlighted opportunities for improvement, particularly in optimising precision, recall, and the balance between false positives and false negatives. These insights informed the next phase of model refinement and feature enhancement.

Here’s a refined version of your **3.5 Learnings** and **Final Model (4.1–4.4)** sections with improved structure, clarity, and tone—*and all numerical values removed as requested*:

---

### 3.3 Learnings

From the initial modelling phase, we gathered several key insights:

* Even without advanced feature engineering or hyperparameter tuning, the baseline model achieved reasonable predictive performance.
* The model was effective at identifying **who** should be targeted for service upsell conversations, but less effective at predicting **when** those interactions should take place.
* To enhance the model, the next steps included incorporating time-based features (e.g. lagged service data), tuning model parameters, and training multiple models across different time horizons to better capture the timing of client engagement opportunities.

---

## 4. Final Model

### 4.1 Feature Enhancements

In the final model, we introduced a set of engineered features designed to capture temporal patterns and strengthen the predictive signal:

* **Lagged Features**: These included trailing aggregates over recent time windows—such as the number and types of services accessed, frequency of visits, and total spend. By summarising recent client activity, these features helped the model detect behavioural shifts that may signal readiness for new services.

---

### 4.2 Algorithm and Tuning

We selected a gradient-boosted decision tree model using the XGBoost framework, known for its strong performance on structured datasets and ability to model complex, non-linear relationships.
To optimise performance, we used FLAML’s AutoML framework, which automatically tuned the model’s hyperparameters using a cost-aware optimisation strategy.

The final solution consisted of an ensemble of models trained over different future timeframes. This **multi-horizon modelling** approach enabled the system not only to predict who is likely to engage, but also to recommend the best timing for outreach. It allowed us to align client engagement strategies with meaningful behavioural windows.

---

### 4.3 Cross-Validation Strategy

To ensure stability and generalisability, we used cross-validation across multiple time-based slices. This helped verify that model performance held up consistently as client behaviour evolved over time.

We also monitored **data drift** using the Population Stability Index (PSI), a metric that quantifies changes in data distribution. Across the different time windows, PSI values remained within acceptable limits, suggesting overall model stability. A slightly elevated PSI observed during the final testing window indicated a modest shift in patterns, which may require further monitoring or eventual retraining if the trend continues.

Despite natural variation in the data, the model consistently demonstrated strong predictive power across all time periods tested.

---

### 4.4 Final Model Metrics

To better inform the timing of client engagement, we developed a series of models trained to predict service uptake across multiple future horizons. Rather than focusing solely on immediate next-week outcomes, each model was trained with a **shifted target**—effectively asking:

> *“If we were to engage a client this week, how likely are they to take up a new service in the next few weeks?”*

This approach allowed us to distinguish between clients who are ready for **immediate outreach** and those who may be more responsive to **early planning or follow-up** in the coming weeks.

While the model for immediate uptake delivered the strongest performance, models targeting further horizons also maintained solid predictive ability. This combination supports both reactive and proactive strategies—enabling staff to prioritise the right clients at the right time and maximise service engagement outcomes.

---
Here's a refined version of **4.5 Calibration Output** and **4.6 Interpretability**, with improved clarity, flow, and professionalism—*and all numerical values removed as requested*:

---

### 4.5 Calibration Output

To evaluate how well the model’s predicted probabilities align with actual outcomes, we assessed its **calibration**. Calibration examines whether the model’s confidence levels reflect real-world probabilities—for instance, when the model predicts a certain likelihood of service uptake, that outcome should occur at a similar rate in practice.

We used two key metrics to assess calibration quality:

* **Brier Score**: Measures the accuracy of probabilistic predictions, with lower values indicating better performance.
* **Expected Calibration Error (ECE)**: Measures the difference between predicted probabilities and observed outcomes; lower values indicate better calibration.

The model predicting immediate service uptake exhibited the strongest calibration, meaning its probability estimates closely reflected real-world outcomes. As the time horizon extended, calibration performance declined slightly but remained within acceptable bounds. This indicates that even over longer forecast windows, the model’s confidence levels continued to provide meaningful guidance for decision-making.

---

### 4.6 Interpretability

Model interpretability plays a vital role in building stakeholder trust and enabling frontline advisors to act on predictions with confidence.

To achieve this, we employed **SHAP (SHapley Additive exPlanations)**—a leading framework for model explainability. SHAP provides:

* **Feature-level insights** into how much each input contributes to a prediction.
* **Global explanations** that identify which features are most influential across the entire dataset.
* **Local explanations** that reveal why a specific prediction was made for an individual client.
* **Seamless integration** with XGBoost, the final production model used in this project.

These capabilities allow us to clearly communicate what drives model predictions, support informed advisor decisions, and ensure alignment with service delivery goals.

---

### 4.7 Robustness Tests
For robustness testing, we were able to conduct cross-validation (Section 4.3) to assess how well the model performs over different time periods. However, more in-depth testing—such as evaluating edge cases, analysing performance across different client groups, or detecting subtle data shifts—would require access to a larger and more reliable dataset. This extended testing would help ensure the model remains stable and accurate under a wider range of real-world conditions and client scenarios.


