# Data-Analysis-Argentic-AI
An agentic data analysis system using Python, LangGraph, and Google Gemini to orchestrate dynamic, tool-based analytical workflows.

# Agentic Data Analysis Pipeline

**Agentic Data Analysis Pipeline** built with **Python**, **LangGraph**, and **Google Gemini API**.

This project implements an **agentic AI system** that performs end-to-end data analysis through a **decision-aware pipeline**. Instead of a fixed script, the system uses **LangGraph** to orchestrate analysis steps as a **stateful graph**, enabling dynamic planning, tool execution, and iterative reasoning.

## Capabilities

The agent is capable of:

- Interpreting a business question  
- Inspecting dataset schema and samples  
- Planning analytical steps  
- Executing Python-based data analysis (e.g., `pandas`)  
- Translating results into business-oriented insights  

## Architecture Overview

The system is implemented entirely in **Python**, uses **LangGraph** for agent orchestration and workflow control, and integrates **Google Gemini** as the underlying large language model to enable **low-cost experimentation** with modern LLM capabilities.

## Project Goal

The goal of this project is to build an **agentic machine learning pipeline** that can automatically perform **predictive modeling** based on a user’s intent and dataset context.

Given a **CSV dataset** and a **high-level dataset description**, the system is designed to:
- Infer whether the analytical task should be treated as **regression** or **classification**
- Identify a suitable **target variable** based on the dataset context
- Train a **baseline predictive model** using appropriate preprocessing and evaluation metrics
- Provide **interpretable insights** into how input variables influence the model
- Suggest **actionable next steps** for improving model performance and data quality

Rather than optimizing for model performance alone, this project focuses on **decision-aware automation**, emphasizing transparency, interpretability, and reproducibility in applied machine learning workflows.

---

## Success Criteria

This project is considered successful if the system can:

- Accept a CSV file and a short dataset description as input
- Correctly distinguish between **regression** and **classification** tasks with a clear rationale
- Produce a **working baseline model** with relevant evaluation metrics
- Generate a concise summary of **feature importance or variable impact**
- Identify potential data or modeling issues (e.g., missing values, imbalance, leakage)
- Output a structured, human-readable report that explains:
  - What model was built
  - Why it was chosen
  - What the results mean
  - How the analysis could be improved in future iterations

The emphasis is on building a **reliable analytical foundation** that can be extended with more advanced models, feature engineering, or evaluation techniques over time.

rather than traditional static data pipelines.
<img width="1536" height="1024" alt="image" src="https://github.com/user-attachments/assets/c8221038-2b2f-4c4d-a3af-bf918c96c4de" />

