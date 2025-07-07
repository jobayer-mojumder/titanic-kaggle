import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import os

#  exp1: Baseline data for all models
baseline_evaluation = [
    {"Model": "Decision Tree", "Kaggle": 0.73205, "Local": 0.75754},
    {"Model": "XGBoost", "Kaggle": 0.7512, "Local": 0.78547},
    {"Model": "Random Forests", "Kaggle": 0.76555, "Local": 0.79106},
    {"Model": "LightGBM", "Kaggle": 0.76555, "Local": 0.78994},
    {"Model": "CatBoost", "Kaggle": 0.77751, "Local": 0.80671},
]

#  exp2: Single feature data for all models
single_feature_data_untuned = [
    {
        "Feature": 1,
        "Model": "Decision Tree",
        "Kaggle": 0.78708,
        "Kaggle_Improvement": 0.05503,
        "Local": 0.80559,
        "Local_Improvement": 0.04805,
    },
    {
        "Feature": 1,
        "Model": "XGBoost",
        "Kaggle": 0.78708,
        "Kaggle_Improvement": 0.03588,
        "Local": 0.80894,
        "Local_Improvement": 0.02347,
    },
    {
        "Feature": 1,
        "Model": "Random Forests",
        "Kaggle": 0.78708,
        "Kaggle_Improvement": 0.02153,
        "Local": 0.80447,
        "Local_Improvement": 0.01341,
    },
    {
        "Feature": 1,
        "Model": "LightGBM",
        "Kaggle": 0.78708,
        "Kaggle_Improvement": 0.02153,
        "Local": 0.8,
        "Local_Improvement": 0.01006,
    },
    {
        "Feature": 1,
        "Model": "CatBoost",
        "Kaggle": 0.78708,
        "Kaggle_Improvement": 0.00956,
        "Local": 0.80782,
        "Local_Improvement": -0.00224,
    },
    {
        "Feature": 2,
        "Model": "Decision Tree",
        "Kaggle": 0.76555,
        "Kaggle_Improvement": 0.03349,
        "Local": 0.77654,
        "Local_Improvement": 0.019,
    },
    {
        "Feature": 2,
        "Model": "XGBoost",
        "Kaggle": 0.76555,
        "Kaggle_Improvement": 0.01435,
        "Local": 0.76313,
        "Local_Improvement": -0.02234,
    },
    {
        "Feature": 2,
        "Model": "Random Forests",
        "Kaggle": 0.75358,
        "Kaggle_Improvement": -0.01196,
        "Local": 0.76425,
        "Local_Improvement": -0.02681,
    },
    {
        "Feature": 2,
        "Model": "LightGBM",
        "Kaggle": 0.76555,
        "Kaggle_Improvement": 0.0,
        "Local": 0.76983,
        "Local_Improvement": -0.02011,
    },
    {
        "Feature": 2,
        "Model": "CatBoost",
        "Kaggle": 0.76794,
        "Kaggle_Improvement": -0.00957,
        "Local": 0.76201,
        "Local_Improvement": -0.04805,
    },
    {
        "Feature": 3,
        "Model": "Decision Tree",
        "Kaggle": 0.77751,
        "Kaggle_Improvement": 0.04546,
        "Local": 0.77765,
        "Local_Improvement": 0.02011,
    },
    {
        "Feature": 3,
        "Model": "XGBoost",
        "Kaggle": 0.77751,
        "Kaggle_Improvement": 0.02632,
        "Local": 0.78212,
        "Local_Improvement": -0.00335,
    },
    {
        "Feature": 3,
        "Model": "Random Forests",
        "Kaggle": 0.76555,
        "Kaggle_Improvement": 0.0,
        "Local": 0.78212,
        "Local_Improvement": -0.00894,
    },
    {
        "Feature": 3,
        "Model": "LightGBM",
        "Kaggle": 0.77751,
        "Kaggle_Improvement": 0.01196,
        "Local": 0.78212,
        "Local_Improvement": -0.00782,
    },
    {
        "Feature": 3,
        "Model": "CatBoost",
        "Kaggle": 0.77751,
        "Kaggle_Improvement": 0.0,
        "Local": 0.78212,
        "Local_Improvement": -0.02794,
    },
    {
        "Feature": 4,
        "Model": "Decision Tree",
        "Kaggle": 0.78468,
        "Kaggle_Improvement": 0.05263,
        "Local": 0.8,
        "Local_Improvement": 0.04246,
    },
    {
        "Feature": 4,
        "Model": "XGBoost",
        "Kaggle": 0.78468,
        "Kaggle_Improvement": 0.03349,
        "Local": 0.79553,
        "Local_Improvement": 0.01006,
    },
    {
        "Feature": 4,
        "Model": "Random Forests",
        "Kaggle": 0.78468,
        "Kaggle_Improvement": 0.01913,
        "Local": 0.8,
        "Local_Improvement": 0.00894,
    },
    {
        "Feature": 4,
        "Model": "LightGBM",
        "Kaggle": 0.78468,
        "Kaggle_Improvement": 0.01913,
        "Local": 0.7933,
        "Local_Improvement": 0.00336,
    },
    {
        "Feature": 4,
        "Model": "CatBoost",
        "Kaggle": 0.78468,
        "Kaggle_Improvement": 0.00717,
        "Local": 0.8,
        "Local_Improvement": -0.01006,
    },
    {
        "Feature": 5,
        "Model": "Decision Tree",
        "Kaggle": 0.75837,
        "Kaggle_Improvement": 0.02632,
        "Local": 0.79665,
        "Local_Improvement": 0.03911,
    },
    {
        "Feature": 5,
        "Model": "XGBoost",
        "Kaggle": 0.76315,
        "Kaggle_Improvement": 0.01195,
        "Local": 0.78994,
        "Local_Improvement": 0.00447,
    },
    {
        "Feature": 5,
        "Model": "Random Forests",
        "Kaggle": 0.77272,
        "Kaggle_Improvement": 0.00717,
        "Local": 0.79553,
        "Local_Improvement": 0.00447,
    },
    {
        "Feature": 5,
        "Model": "LightGBM",
        "Kaggle": 0.77511,
        "Kaggle_Improvement": 0.00956,
        "Local": 0.7933,
        "Local_Improvement": 0.00336,
    },
    {
        "Feature": 5,
        "Model": "CatBoost",
        "Kaggle": 0.76076,
        "Kaggle_Improvement": -0.01675,
        "Local": 0.79777,
        "Local_Improvement": -0.01229,
    },
    {
        "Feature": 6,
        "Model": "Decision Tree",
        "Kaggle": 0.77751,
        "Kaggle_Improvement": 0.04546,
        "Local": 0.79106,
        "Local_Improvement": 0.03352,
    },
    {
        "Feature": 6,
        "Model": "XGBoost",
        "Kaggle": 0.77751,
        "Kaggle_Improvement": 0.02632,
        "Local": 0.79106,
        "Local_Improvement": 0.00559,
    },
    {
        "Feature": 6,
        "Model": "Random Forests",
        "Kaggle": 0.77751,
        "Kaggle_Improvement": 0.01196,
        "Local": 0.79106,
        "Local_Improvement": 0.0,
    },
    {
        "Feature": 6,
        "Model": "LightGBM",
        "Kaggle": 0.77751,
        "Kaggle_Improvement": 0.01196,
        "Local": 0.79106,
        "Local_Improvement": 0.00112,
    },
    {
        "Feature": 6,
        "Model": "CatBoost",
        "Kaggle": 0.77751,
        "Kaggle_Improvement": 0.0,
        "Local": 0.79106,
        "Local_Improvement": -0.019,
    },
    {
        "Feature": 7,
        "Model": "Decision Tree",
        "Kaggle": 0.77751,
        "Kaggle_Improvement": 0.04546,
        "Local": 0.79106,
        "Local_Improvement": 0.03352,
    },
    {
        "Feature": 7,
        "Model": "XGBoost",
        "Kaggle": 0.77751,
        "Kaggle_Improvement": 0.02632,
        "Local": 0.79106,
        "Local_Improvement": 0.00559,
    },
    {
        "Feature": 7,
        "Model": "Random Forests",
        "Kaggle": 0.77751,
        "Kaggle_Improvement": 0.01196,
        "Local": 0.79106,
        "Local_Improvement": 0.0,
    },
    {
        "Feature": 7,
        "Model": "LightGBM",
        "Kaggle": 0.77751,
        "Kaggle_Improvement": 0.01196,
        "Local": 0.79106,
        "Local_Improvement": 0.00112,
    },
    {
        "Feature": 7,
        "Model": "CatBoost",
        "Kaggle": 0.77751,
        "Kaggle_Improvement": 0.0,
        "Local": 0.79106,
        "Local_Improvement": -0.019,
    },
    {
        "Feature": 8,
        "Model": "Decision Tree",
        "Kaggle": 0.77751,
        "Kaggle_Improvement": 0.04546,
        "Local": 0.79218,
        "Local_Improvement": 0.03464,
    },
    {
        "Feature": 8,
        "Model": "XGBoost",
        "Kaggle": 0.77751,
        "Kaggle_Improvement": 0.02632,
        "Local": 0.79218,
        "Local_Improvement": 0.00671,
    },
    {
        "Feature": 8,
        "Model": "Random Forests",
        "Kaggle": 0.77751,
        "Kaggle_Improvement": 0.01196,
        "Local": 0.79218,
        "Local_Improvement": 0.00112,
    },
    {
        "Feature": 8,
        "Model": "LightGBM",
        "Kaggle": 0.77751,
        "Kaggle_Improvement": 0.01196,
        "Local": 0.79106,
        "Local_Improvement": 0.00112,
    },
    {
        "Feature": 8,
        "Model": "CatBoost",
        "Kaggle": 0.77751,
        "Kaggle_Improvement": 0.0,
        "Local": 0.79218,
        "Local_Improvement": -0.01788,
    },
    {
        "Feature": 9,
        "Model": "Decision Tree",
        "Kaggle": 0.78468,
        "Kaggle_Improvement": 0.05263,
        "Local": 0.8,
        "Local_Improvement": 0.04246,
    },
    {
        "Feature": 9,
        "Model": "XGBoost",
        "Kaggle": 0.78468,
        "Kaggle_Improvement": 0.03349,
        "Local": 0.79553,
        "Local_Improvement": 0.01006,
    },
    {
        "Feature": 9,
        "Model": "Random Forests",
        "Kaggle": 0.78468,
        "Kaggle_Improvement": 0.01913,
        "Local": 0.8,
        "Local_Improvement": 0.00894,
    },
    {
        "Feature": 9,
        "Model": "LightGBM",
        "Kaggle": 0.78468,
        "Kaggle_Improvement": 0.01913,
        "Local": 0.79218,
        "Local_Improvement": 0.00224,
    },
    {
        "Feature": 9,
        "Model": "CatBoost",
        "Kaggle": 0.78468,
        "Kaggle_Improvement": 0.00717,
        "Local": 0.8,
        "Local_Improvement": -0.01006,
    },
    {
        "Feature": 10,
        "Model": "Decision Tree",
        "Kaggle": 0.78468,
        "Kaggle_Improvement": 0.05263,
        "Local": 0.80559,
        "Local_Improvement": 0.04805,
    },
    {
        "Feature": 10,
        "Model": "XGBoost",
        "Kaggle": 0.78468,
        "Kaggle_Improvement": 0.03349,
        "Local": 0.80112,
        "Local_Improvement": 0.01565,
    },
    {
        "Feature": 10,
        "Model": "Random Forests",
        "Kaggle": 0.78468,
        "Kaggle_Improvement": 0.01913,
        "Local": 0.80559,
        "Local_Improvement": 0.01453,
    },
    {
        "Feature": 10,
        "Model": "LightGBM",
        "Kaggle": 0.78468,
        "Kaggle_Improvement": 0.01913,
        "Local": 0.80223,
        "Local_Improvement": 0.01229,
    },
    {
        "Feature": 10,
        "Model": "CatBoost",
        "Kaggle": 0.78468,
        "Kaggle_Improvement": 0.00717,
        "Local": 0.80559,
        "Local_Improvement": -0.00447,
    },
    {
        "Feature": 11,
        "Model": "Decision Tree",
        "Kaggle": 0.78229,
        "Kaggle_Improvement": 0.05024,
        "Local": 0.78324,
        "Local_Improvement": 0.0257,
    },
    {
        "Feature": 11,
        "Model": "XGBoost",
        "Kaggle": 0.78229,
        "Kaggle_Improvement": 0.0311,
        "Local": 0.78659,
        "Local_Improvement": 0.00112,
    },
    {
        "Feature": 11,
        "Model": "Random Forests",
        "Kaggle": 0.78229,
        "Kaggle_Improvement": 0.01674,
        "Local": 0.78659,
        "Local_Improvement": -0.00447,
    },
    {
        "Feature": 11,
        "Model": "LightGBM",
        "Kaggle": 0.78468,
        "Kaggle_Improvement": 0.01913,
        "Local": 0.78883,
        "Local_Improvement": -0.00111,
    },
    {
        "Feature": 11,
        "Model": "CatBoost",
        "Kaggle": 0.78229,
        "Kaggle_Improvement": 0.00478,
        "Local": 0.79106,
        "Local_Improvement": -0.019,
    },
    {
        "Feature": 12,
        "Model": "Decision Tree",
        "Kaggle": 0.76315,
        "Kaggle_Improvement": 0.0311,
        "Local": 0.78771,
        "Local_Improvement": 0.03017,
    },
    {
        "Feature": 12,
        "Model": "XGBoost",
        "Kaggle": 0.76315,
        "Kaggle_Improvement": 0.01195,
        "Local": 0.78659,
        "Local_Improvement": 0.00112,
    },
    {
        "Feature": 12,
        "Model": "Random Forests",
        "Kaggle": 0.76315,
        "Kaggle_Improvement": -0.00239,
        "Local": 0.78547,
        "Local_Improvement": -0.00559,
    },
    {
        "Feature": 12,
        "Model": "LightGBM",
        "Kaggle": 0.77511,
        "Kaggle_Improvement": 0.00956,
        "Local": 0.78994,
        "Local_Improvement": 0.0,
    },
    {
        "Feature": 12,
        "Model": "CatBoost",
        "Kaggle": 0.76315,
        "Kaggle_Improvement": -0.01436,
        "Local": 0.78994,
        "Local_Improvement": -0.02012,
    },
]

#  exp3: Top 10 feature combinations for all models
top10_feature_combinations_untuned = [
    {
        "Model": "Decision Tree",
        "Feature_Combination": [1, 3, 4, 6, 7, 9, 10, 11],
        "Kaggle": 0.80143,
        "Improvement": 0.06938,
        "Local": 0.8,
        "Local_Improvement": 0.04246,
    },
    {
        "Model": "Decision Tree",
        "Feature_Combination": [1, 3, 4, 6, 9, 10, 11],
        "Kaggle": 0.80143,
        "Improvement": 0.06938,
        "Local": 0.8,
        "Local_Improvement": 0.04246,
    },
    {
        "Model": "Decision Tree",
        "Feature_Combination": [1, 3, 4, 6, 7, 10, 11],
        "Kaggle": 0.80143,
        "Improvement": 0.06938,
        "Local": 0.79777,
        "Local_Improvement": 0.04023,
    },
    {
        "Model": "Decision Tree",
        "Feature_Combination": [1, 3, 4, 7, 9, 10, 11],
        "Kaggle": 0.80143,
        "Improvement": 0.06938,
        "Local": 0.79888,
        "Local_Improvement": 0.04134,
    },
    {
        "Model": "Decision Tree",
        "Feature_Combination": [1, 3, 4, 9, 10, 11],
        "Kaggle": 0.80143,
        "Improvement": 0.06938,
        "Local": 0.79888,
        "Local_Improvement": 0.04134,
    },
    {
        "Model": "Decision Tree",
        "Feature_Combination": [1, 3, 4, 6, 10, 11],
        "Kaggle": 0.80143,
        "Improvement": 0.06938,
        "Local": 0.79777,
        "Local_Improvement": 0.04023,
    },
    {
        "Model": "Decision Tree",
        "Feature_Combination": [1, 3, 6, 11],
        "Kaggle": 0.79904,
        "Improvement": 0.06698,
        "Local": 0.79665,
        "Local_Improvement": 0.03911,
    },
    {
        "Model": "Decision Tree",
        "Feature_Combination": [3, 4, 6, 7, 8, 11],
        "Kaggle": 0.79904,
        "Improvement": 0.06698,
        "Local": 0.79777,
        "Local_Improvement": 0.04023,
    },
    {
        "Model": "Decision Tree",
        "Feature_Combination": [3, 7, 10, 11],
        "Kaggle": 0.79904,
        "Improvement": 0.06698,
        "Local": 0.79553,
        "Local_Improvement": 0.03799,
    },
    {
        "Model": "Decision Tree",
        "Feature_Combination": [1, 3, 4, 7, 8, 9, 10, 11],
        "Kaggle": 0.79904,
        "Improvement": 0.06698,
        "Local": 0.8,
        "Local_Improvement": 0.04246,
    },
    {
        "Model": "XGBoost",
        "Feature_Combination": [1, 3, 4, 7, 9, 10, 11, 12],
        "Kaggle": 0.80382,
        "Improvement": 0.05262,
        "Local": 0.79665,
        "Local_Improvement": 0.01118,
    },
    {
        "Model": "XGBoost",
        "Feature_Combination": [1, 3, 4, 9, 10, 11, 12],
        "Kaggle": 0.80382,
        "Improvement": 0.05262,
        "Local": 0.79665,
        "Local_Improvement": 0.01118,
    },
    {
        "Model": "XGBoost",
        "Feature_Combination": [1, 3, 4, 10, 11, 12],
        "Kaggle": 0.80382,
        "Improvement": 0.05262,
        "Local": 0.79665,
        "Local_Improvement": 0.01118,
    },
    {
        "Model": "XGBoost",
        "Feature_Combination": [1, 3, 4, 7, 10, 11, 12],
        "Kaggle": 0.80382,
        "Improvement": 0.05262,
        "Local": 0.79665,
        "Local_Improvement": 0.01118,
    },
    {
        "Model": "XGBoost",
        "Feature_Combination": [3, 4, 6, 7, 8, 9, 11, 12],
        "Kaggle": 0.80143,
        "Improvement": 0.05023,
        "Local": 0.79553,
        "Local_Improvement": 0.01006,
    },
    {
        "Model": "XGBoost",
        "Feature_Combination": [3, 4, 6, 7, 8, 11, 12],
        "Kaggle": 0.80143,
        "Improvement": 0.05023,
        "Local": 0.79441,
        "Local_Improvement": 0.00894,
    },
    {
        "Model": "XGBoost",
        "Feature_Combination": [3, 4, 8, 9, 11, 12],
        "Kaggle": 0.80143,
        "Improvement": 0.05023,
        "Local": 0.79665,
        "Local_Improvement": 0.01118,
    },
    {
        "Model": "XGBoost",
        "Feature_Combination": [3, 4, 8, 11, 12],
        "Kaggle": 0.80143,
        "Improvement": 0.05023,
        "Local": 0.79665,
        "Local_Improvement": 0.01118,
    },
    {
        "Model": "XGBoost",
        "Feature_Combination": [3, 4, 7, 8, 11, 12],
        "Kaggle": 0.80143,
        "Improvement": 0.05023,
        "Local": 0.79665,
        "Local_Improvement": 0.01118,
    },
    {
        "Model": "XGBoost",
        "Feature_Combination": [3, 4, 6, 8, 9, 11, 12],
        "Kaggle": 0.80143,
        "Improvement": 0.05023,
        "Local": 0.79553,
        "Local_Improvement": 0.01006,
    },
    {
        "Model": "Random Forests",
        "Feature_Combination": [1, 3, 4, 6, 7, 11],
        "Kaggle": 0.80861,
        "Improvement": 0.04306,
        "Local": 0.79665,
        "Local_Improvement": 0.00559,
    },
    {
        "Model": "Random Forests",
        "Feature_Combination": [1, 3, 4, 6, 7, 10, 11],
        "Kaggle": 0.80861,
        "Improvement": 0.04306,
        "Local": 0.7933,
        "Local_Improvement": 0.00224,
    },
    {
        "Model": "Random Forests",
        "Feature_Combination": [1, 3, 4, 10, 11],
        "Kaggle": 0.80861,
        "Improvement": 0.04306,
        "Local": 0.79553,
        "Local_Improvement": 0.00447,
    },
    {
        "Model": "Random Forests",
        "Feature_Combination": [1, 3, 4, 9, 11],
        "Kaggle": 0.80861,
        "Improvement": 0.04306,
        "Local": 0.79553,
        "Local_Improvement": 0.00447,
    },
    {
        "Model": "Random Forests",
        "Feature_Combination": [1, 3, 4, 8, 11],
        "Kaggle": 0.80861,
        "Improvement": 0.04306,
        "Local": 0.79553,
        "Local_Improvement": 0.00447,
    },
    {
        "Model": "Random Forests",
        "Feature_Combination": [1, 3, 4, 6, 11],
        "Kaggle": 0.80861,
        "Improvement": 0.04306,
        "Local": 0.79665,
        "Local_Improvement": 0.00559,
    },
    {
        "Model": "Random Forests",
        "Feature_Combination": [1, 3, 4, 6, 10, 11],
        "Kaggle": 0.80861,
        "Improvement": 0.04306,
        "Local": 0.7933,
        "Local_Improvement": 0.00224,
    },
    {
        "Model": "Random Forests",
        "Feature_Combination": [1, 3, 4, 7, 8, 11],
        "Kaggle": 0.80861,
        "Improvement": 0.04306,
        "Local": 0.79553,
        "Local_Improvement": 0.00447,
    },
    {
        "Model": "Random Forests",
        "Feature_Combination": [1, 3, 4, 7, 9, 11],
        "Kaggle": 0.80861,
        "Improvement": 0.04306,
        "Local": 0.79553,
        "Local_Improvement": 0.00447,
    },
    {
        "Model": "Random Forests",
        "Feature_Combination": [1, 3, 4, 7, 10, 11],
        "Kaggle": 0.80861,
        "Improvement": 0.04306,
        "Local": 0.79553,
        "Local_Improvement": 0.00447,
    },
    {
        "Model": "LightGBM",
        "Feature_Combination": [1, 3, 4, 6, 8, 9, 11, 12],
        "Kaggle": 0.80382,
        "Improvement": 0.03827,
        "Local": 0.78883,
        "Local_Improvement": -0.00111,
    },
    {
        "Model": "LightGBM",
        "Feature_Combination": [1, 3, 4, 6, 7, 8, 11, 12],
        "Kaggle": 0.80382,
        "Improvement": 0.03827,
        "Local": 0.78883,
        "Local_Improvement": -0.00111,
    },
    {
        "Model": "LightGBM",
        "Feature_Combination": [1, 3, 4, 6, 8, 11, 12],
        "Kaggle": 0.80382,
        "Improvement": 0.03827,
        "Local": 0.78883,
        "Local_Improvement": -0.00111,
    },
    {
        "Model": "LightGBM",
        "Feature_Combination": [1, 3, 4, 6, 7, 8, 9, 11, 12],
        "Kaggle": 0.80382,
        "Improvement": 0.03827,
        "Local": 0.78883,
        "Local_Improvement": -0.00111,
    },
    {
        "Model": "LightGBM",
        "Feature_Combination": [1, 3, 4, 7, 11],
        "Kaggle": 0.80143,
        "Improvement": 0.03588,
        "Local": 0.78883,
        "Local_Improvement": -0.00111,
    },
    {
        "Model": "LightGBM",
        "Feature_Combination": [1, 3, 4, 9, 11],
        "Kaggle": 0.80143,
        "Improvement": 0.03588,
        "Local": 0.78883,
        "Local_Improvement": -0.00111,
    },
    {
        "Model": "LightGBM",
        "Feature_Combination": [1, 3, 4, 11],
        "Kaggle": 0.80143,
        "Improvement": 0.03588,
        "Local": 0.78883,
        "Local_Improvement": -0.00111,
    },
    {
        "Model": "LightGBM",
        "Feature_Combination": [1, 3, 4, 7, 9, 11],
        "Kaggle": 0.80143,
        "Improvement": 0.03588,
        "Local": 0.78883,
        "Local_Improvement": -0.00111,
    },
    {
        "Model": "LightGBM",
        "Feature_Combination": [1, 3, 4, 8, 11],
        "Kaggle": 0.79904,
        "Improvement": 0.03349,
        "Local": 0.79106,
        "Local_Improvement": 0.00112,
    },
    {
        "Model": "LightGBM",
        "Feature_Combination": [1, 3, 7, 8, 9, 11],
        "Kaggle": 0.79904,
        "Improvement": 0.03349,
        "Local": 0.79218,
        "Local_Improvement": 0.00224,
    },
    {
        "Model": "CatBoost",
        "Feature_Combination": [1, 3, 5, 8, 10, 11, 12],
        "Kaggle": 0.80143,
        "Improvement": 0.02391,
        "Local": 0.80894,
        "Local_Improvement": -0.00112,
    },
    {
        "Model": "CatBoost",
        "Feature_Combination": [1, 3, 5, 7, 8, 10, 11, 12],
        "Kaggle": 0.80143,
        "Improvement": 0.02391,
        "Local": 0.80894,
        "Local_Improvement": -0.00112,
    },
    {
        "Model": "CatBoost",
        "Feature_Combination": [1, 5, 6, 7, 8, 10, 11, 12],
        "Kaggle": 0.80143,
        "Improvement": 0.02391,
        "Local": 0.81229,
        "Local_Improvement": 0.00223,
    },
    {
        "Model": "CatBoost",
        "Feature_Combination": [1, 5, 6, 8, 10, 11, 12],
        "Kaggle": 0.80143,
        "Improvement": 0.02391,
        "Local": 0.81229,
        "Local_Improvement": 0.00223,
    },
    {
        "Model": "CatBoost",
        "Feature_Combination": [1, 3, 4, 8, 9, 11, 12],
        "Kaggle": 0.79904,
        "Improvement": 0.02152,
        "Local": 0.80223,
        "Local_Improvement": -0.00783,
    },
    {
        "Model": "CatBoost",
        "Feature_Combination": [3, 5, 8, 10, 11, 12],
        "Kaggle": 0.79904,
        "Improvement": 0.02152,
        "Local": 0.8067,
        "Local_Improvement": -0.00336,
    },
    {
        "Model": "CatBoost",
        "Feature_Combination": [3, 5, 8, 9, 10, 11],
        "Kaggle": 0.79904,
        "Improvement": 0.02152,
        "Local": 0.80223,
        "Local_Improvement": -0.00783,
    },
    {
        "Model": "CatBoost",
        "Feature_Combination": [1, 4, 5, 8, 11, 12],
        "Kaggle": 0.79904,
        "Improvement": 0.02152,
        "Local": 0.8067,
        "Local_Improvement": -0.00336,
    },
    {
        "Model": "CatBoost",
        "Feature_Combination": [1, 4, 5, 9, 11, 12],
        "Kaggle": 0.79904,
        "Improvement": 0.02152,
        "Local": 0.80559,
        "Local_Improvement": -0.00447,
    },
    {
        "Model": "CatBoost",
        "Feature_Combination": [3, 4, 5, 6, 8, 10, 11],
        "Kaggle": 0.79904,
        "Improvement": 0.02152,
        "Local": 0.80112,
        "Local_Improvement": -0.00894,
    },
]

#  exp4: Best feature combinations for all models
best_feature_combinations_untuned = [
    {
        "Model": "Decision Tree",
        "Feature_Combination": [1, 3, 4, 6, 7, 9, 10, 11],
        "Kaggle": 0.80143,
        "Improvement": 0.06938,
        "Local": 0.8,
        "Local_Improvement": 0.04246,
    },
    {
        "Model": "XGBoost",
        "Feature_Combination": [1, 3, 4, 7, 9, 10, 11, 12],
        "Kaggle": 0.80382,
        "Improvement": 0.05262,
        "Local": 0.79665,
        "Local_Improvement": 0.01118,
    },
    {
        "Model": "Random Forests",
        "Feature_Combination": [1, 3, 4, 6, 7, 11],
        "Kaggle": 0.80861,
        "Improvement": 0.04306,
        "Local": 0.79665,
        "Local_Improvement": 0.00559,
    },
    {
        "Model": "LightGBM",
        "Feature_Combination": [1, 3, 4, 6, 8, 9, 11, 12],
        "Kaggle": 0.80382,
        "Improvement": 0.03827,
        "Local": 0.78883,
        "Local_Improvement": -0.00111,
    },
    {
        "Model": "CatBoost",
        "Feature_Combination": [1, 3, 5, 8, 10, 11, 12],
        "Kaggle": 0.80143,
        "Improvement": 0.02391,
        "Local": 0.80894,
        "Local_Improvement": -0.00112,
    },
]

#  exp5: All features combined for all models
all_features_combined_untuned = [
    {
        "Model": "Decision Tree",
        "Feature_Combination": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
        "Kaggle": 0.7488,
        "Improvement": 0.01675,
        "Local": 0.8,
        "Local_Improvement": 0.04246,
    },
    {
        "Model": "XGBoost",
        "Feature_Combination": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
        "Kaggle": 0.74401,
        "Improvement": -0.00718,
        "Local": 0.79553,
        "Local_Improvement": 0.01006,
    },
    {
        "Model": "Random Forests",
        "Feature_Combination": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
        "Kaggle": 0.73444,
        "Improvement": -0.0311,
        "Local": 0.7933,
        "Local_Improvement": 0.00224,
    },
    {
        "Model": "LightGBM",
        "Feature_Combination": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
        "Kaggle": 0.75358,
        "Improvement": -0.01196,
        "Local": 0.81229,
        "Local_Improvement": 0.02235,
    },
    {
        "Model": "CatBoost",
        "Feature_Combination": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
        "Kaggle": 0.7799,
        "Improvement": 0.00239,
        "Local": 0.80335,
        "Local_Improvement": -0.00671,
    },
]

#  exp6: Top 3 features per model
top3_features_per_model = [
    {
        "Model": "Decision Tree",
        "Feature No.": 7,
        "Importance (%)": 10.6,
        "Feature Name": "Deck",
    },
    {
        "Model": "Decision Tree",
        "Feature No.": 8,
        "Importance (%)": 9.27,
        "Feature Name": "IsMother",
    },
    {
        "Model": "Decision Tree",
        "Feature No.": 9,
        "Importance (%)": 9.27,
        "Feature Name": "IsChild",
    },
    {
        "Model": "XGBoost",
        "Feature No.": 7,
        "Importance (%)": 10.6,
        "Feature Name": "Deck",
    },
    {
        "Model": "XGBoost",
        "Feature No.": 3,
        "Importance (%)": 9.27,
        "Feature Name": "IsAlone",
    },
    {
        "Model": "XGBoost",
        "Feature No.": 9,
        "Importance (%)": 9.27,
        "Feature Name": "IsChild",
    },
    {
        "Model": "Random Forests",
        "Feature No.": 7,
        "Importance (%)": 10.6,
        "Feature Name": "Deck",
    },
    {
        "Model": "Random Forests",
        "Feature No.": 10,
        "Importance (%)": 9.27,
        "Feature Name": "WomenChildrenFirst",
    },
    {
        "Model": "Random Forests",
        "Feature No.": 12,
        "Importance (%)": 9.27,
        "Feature Name": "HasCabin",
    },
    {
        "Model": "LightGBM",
        "Feature No.": 5,
        "Importance (%)": 16.58,
        "Feature Name": "FarePerPerson",
    },
    {
        "Model": "LightGBM",
        "Feature No.": 2,
        "Importance (%)": 12.79,
        "Feature Name": "FamilySize",
    },
    {
        "Model": "LightGBM",
        "Feature No.": 3,
        "Importance (%)": 11.08,
        "Feature Name": "IsAlone",
    },
    {
        "Model": "CatBoost",
        "Feature No.": 7,
        "Importance (%)": 10.6,
        "Feature Name": "Deck",
    },
    {
        "Model": "CatBoost",
        "Feature No.": 10,
        "Importance (%)": 9.27,
        "Feature Name": "WomenChildrenFirst",
    },
    {
        "Model": "CatBoost",
        "Feature No.": 5,
        "Importance (%)": 9.27,
        "Feature Name": "FarePerPerson",
    },
]

#  exp7: Baseline model data tuned for all models
baseline_model_tuned = [
    {
        "Model": "Decision Tree",
        "Feature_Combination": [],
        "Kaggle": 0.76315,
        "Improvement": 0.0311,
        "Local": 0.80335,
        "Local_Improvement": 0.04581,
    },
    {
        "Model": "XGBoost",
        "Feature_Combination": [],
        "Kaggle": 0.77033,
        "Improvement": 0.01913,
        "Local": 0.79106,
        "Local_Improvement": 0.00559,
    },
    {
        "Model": "Random Forests",
        "Feature_Combination": [],
        "Kaggle": 0.77033,
        "Improvement": 0.00478,
        "Local": 0.81006,
        "Local_Improvement": 0.019,
    },
    {
        "Model": "LightGBM",
        "Feature_Combination": [],
        "Kaggle": 0.75598,
        "Improvement": -0.00956,
        "Local": 0.79888,
        "Local_Improvement": 0.00894,
    },
    {
        "Model": "CatBoost",
        "Feature_Combination": [],
        "Kaggle": 0.75119,
        "Improvement": -0.02632,
        "Local": 0.81453,
        "Local_Improvement": 0.00447,
    },
]

#  exp8: tuned Single feature data for all models
single_feature_data_tuned = [
    {
        "Feature": 1,
        "Model": "Decision Tree",
        "Kaggle": 0.78229,
        "Improvement": 0.05024,
        "Local": 0.80559,
        "Local_Improvement": 0.04805,
    },
    {
        "Feature": 1,
        "Model": "XGBoost",
        "Kaggle": 0.78708,
        "Improvement": 0.03588,
        "Local": 0.81006,
        "Local_Improvement": 0.02459,
    },
    {
        "Feature": 1,
        "Model": "Random Forests",
        "Kaggle": 0.78708,
        "Improvement": 0.02153,
        "Local": 0.79888,
        "Local_Improvement": 0.00782,
    },
    {
        "Feature": 1,
        "Model": "LightGBM",
        "Kaggle": 0.78708,
        "Improvement": 0.02153,
        "Local": 0.80335,
        "Local_Improvement": 0.01341,
    },
    {
        "Feature": 1,
        "Model": "CatBoost",
        "Kaggle": 0.78708,
        "Improvement": 0.00956,
        "Local": 0.80894,
        "Local_Improvement": -0.00112,
    },
    {
        "Feature": 2,
        "Model": "Decision Tree",
        "Kaggle": 0.77511,
        "Improvement": 0.04305,
        "Local": 0.77654,
        "Local_Improvement": 0.019,
    },
    {
        "Feature": 2,
        "Model": "XGBoost",
        "Kaggle": 0.77511,
        "Improvement": 0.02391,
        "Local": 0.78212,
        "Local_Improvement": -0.00335,
    },
    {
        "Feature": 2,
        "Model": "Random Forests",
        "Kaggle": 0.77272,
        "Improvement": 0.00717,
        "Local": 0.77654,
        "Local_Improvement": -0.01452,
    },
    {
        "Feature": 2,
        "Model": "LightGBM",
        "Kaggle": 0.77751,
        "Improvement": 0.01196,
        "Local": 0.78994,
        "Local_Improvement": 0.0,
    },
    {
        "Feature": 2,
        "Model": "CatBoost",
        "Kaggle": 0.75837,
        "Improvement": -0.01914,
        "Local": 0.76313,
        "Local_Improvement": -0.04693,
    },
    {
        "Feature": 3,
        "Model": "Decision Tree",
        "Kaggle": 0.77751,
        "Improvement": 0.04546,
        "Local": 0.78212,
        "Local_Improvement": 0.02458,
    },
    {
        "Feature": 3,
        "Model": "XGBoost",
        "Kaggle": 0.77751,
        "Improvement": 0.02632,
        "Local": 0.79106,
        "Local_Improvement": 0.00559,
    },
    {
        "Feature": 3,
        "Model": "Random Forests",
        "Kaggle": 0.77751,
        "Improvement": 0.01196,
        "Local": 0.78659,
        "Local_Improvement": -0.00447,
    },
    {
        "Feature": 3,
        "Model": "LightGBM",
        "Kaggle": 0.77751,
        "Improvement": 0.01196,
        "Local": 0.78324,
        "Local_Improvement": -0.0067,
    },
    {
        "Feature": 3,
        "Model": "CatBoost",
        "Kaggle": 0.77751,
        "Improvement": 0.0,
        "Local": 0.78324,
        "Local_Improvement": -0.02682,
    },
    {
        "Feature": 4,
        "Model": "Decision Tree",
        "Kaggle": 0.78468,
        "Improvement": 0.05263,
        "Local": 0.80447,
        "Local_Improvement": 0.04693,
    },
    {
        "Feature": 4,
        "Model": "XGBoost",
        "Kaggle": 0.78468,
        "Improvement": 0.03349,
        "Local": 0.79777,
        "Local_Improvement": 0.0123,
    },
    {
        "Feature": 4,
        "Model": "Random Forests",
        "Kaggle": 0.78468,
        "Improvement": 0.01913,
        "Local": 0.8067,
        "Local_Improvement": 0.01564,
    },
    {
        "Feature": 4,
        "Model": "LightGBM",
        "Kaggle": 0.78468,
        "Improvement": 0.01913,
        "Local": 0.79777,
        "Local_Improvement": 0.00783,
    },
    {
        "Feature": 4,
        "Model": "CatBoost",
        "Kaggle": 0.78468,
        "Improvement": 0.00717,
        "Local": 0.8,
        "Local_Improvement": -0.01006,
    },
    {
        "Feature": 5,
        "Model": "Decision Tree",
        "Kaggle": 0.76794,
        "Improvement": 0.03588,
        "Local": 0.80223,
        "Local_Improvement": 0.04469,
    },
    {
        "Feature": 5,
        "Model": "XGBoost",
        "Kaggle": 0.77033,
        "Improvement": 0.01913,
        "Local": 0.78994,
        "Local_Improvement": 0.00447,
    },
    {
        "Feature": 5,
        "Model": "Random Forests",
        "Kaggle": 0.75358,
        "Improvement": -0.01196,
        "Local": 0.79106,
        "Local_Improvement": 0.0,
    },
    {
        "Feature": 5,
        "Model": "LightGBM",
        "Kaggle": 0.77751,
        "Improvement": 0.01196,
        "Local": 0.78994,
        "Local_Improvement": 0.0,
    },
    {
        "Feature": 5,
        "Model": "CatBoost",
        "Kaggle": 0.76315,
        "Improvement": -0.01436,
        "Local": 0.79777,
        "Local_Improvement": -0.01229,
    },
    {
        "Feature": 6,
        "Model": "Decision Tree",
        "Kaggle": 0.77751,
        "Improvement": 0.04546,
        "Local": 0.79106,
        "Local_Improvement": 0.03352,
    },
    {
        "Feature": 6,
        "Model": "XGBoost",
        "Kaggle": 0.77751,
        "Improvement": 0.02632,
        "Local": 0.79106,
        "Local_Improvement": 0.00559,
    },
    {
        "Feature": 6,
        "Model": "Random Forests",
        "Kaggle": 0.77751,
        "Improvement": 0.01196,
        "Local": 0.79106,
        "Local_Improvement": 0.0,
    },
    {
        "Feature": 6,
        "Model": "LightGBM",
        "Kaggle": 0.77751,
        "Improvement": 0.01196,
        "Local": 0.79106,
        "Local_Improvement": 0.00112,
    },
    {
        "Feature": 6,
        "Model": "CatBoost",
        "Kaggle": 0.77751,
        "Improvement": 0.0,
        "Local": 0.79106,
        "Local_Improvement": -0.019,
    },
    {
        "Feature": 7,
        "Model": "Decision Tree",
        "Kaggle": 0.77751,
        "Improvement": 0.04546,
        "Local": 0.79106,
        "Local_Improvement": 0.03352,
    },
    {
        "Feature": 7,
        "Model": "XGBoost",
        "Kaggle": 0.77751,
        "Improvement": 0.02632,
        "Local": 0.79106,
        "Local_Improvement": 0.00559,
    },
    {
        "Feature": 7,
        "Model": "Random Forests",
        "Kaggle": 0.77751,
        "Improvement": 0.01196,
        "Local": 0.79106,
        "Local_Improvement": 0.0,
    },
    {
        "Feature": 7,
        "Model": "LightGBM",
        "Kaggle": 0.77751,
        "Improvement": 0.01196,
        "Local": 0.79106,
        "Local_Improvement": 0.00112,
    },
    {
        "Feature": 7,
        "Model": "CatBoost",
        "Kaggle": 0.77751,
        "Improvement": 0.0,
        "Local": 0.79106,
        "Local_Improvement": -0.019,
    },
    {
        "Feature": 8,
        "Model": "Decision Tree",
        "Kaggle": 0.77751,
        "Improvement": 0.04546,
        "Local": 0.79106,
        "Local_Improvement": 0.03352,
    },
    {
        "Feature": 8,
        "Model": "XGBoost",
        "Kaggle": 0.77751,
        "Improvement": 0.02632,
        "Local": 0.78994,
        "Local_Improvement": 0.00447,
    },
    {
        "Feature": 8,
        "Model": "Random Forests",
        "Kaggle": 0.77751,
        "Improvement": 0.01196,
        "Local": 0.78994,
        "Local_Improvement": -0.00112,
    },
    {
        "Feature": 8,
        "Model": "LightGBM",
        "Kaggle": 0.77751,
        "Improvement": 0.01196,
        "Local": 0.79106,
        "Local_Improvement": 0.00112,
    },
    {
        "Feature": 8,
        "Model": "CatBoost",
        "Kaggle": 0.77751,
        "Improvement": 0.0,
        "Local": 0.7933,
        "Local_Improvement": -0.01676,
    },
    {
        "Feature": 9,
        "Model": "Decision Tree",
        "Kaggle": 0.78468,
        "Improvement": 0.05263,
        "Local": 0.8,
        "Local_Improvement": 0.04246,
    },
    {
        "Feature": 9,
        "Model": "XGBoost",
        "Kaggle": 0.78468,
        "Improvement": 0.03349,
        "Local": 0.79777,
        "Local_Improvement": 0.0123,
    },
    {
        "Feature": 9,
        "Model": "Random Forests",
        "Kaggle": 0.78468,
        "Improvement": 0.01913,
        "Local": 0.80223,
        "Local_Improvement": 0.01117,
    },
    {
        "Feature": 9,
        "Model": "LightGBM",
        "Kaggle": 0.78468,
        "Improvement": 0.01913,
        "Local": 0.79888,
        "Local_Improvement": 0.00894,
    },
    {
        "Feature": 9,
        "Model": "CatBoost",
        "Kaggle": 0.78468,
        "Improvement": 0.00717,
        "Local": 0.8,
        "Local_Improvement": -0.01006,
    },
    {
        "Feature": 10,
        "Model": "Decision Tree",
        "Kaggle": 0.78229,
        "Improvement": 0.05024,
        "Local": 0.80559,
        "Local_Improvement": 0.04805,
    },
    {
        "Feature": 10,
        "Model": "XGBoost",
        "Kaggle": 0.78468,
        "Improvement": 0.03349,
        "Local": 0.79777,
        "Local_Improvement": 0.0123,
    },
    {
        "Feature": 10,
        "Model": "Random Forests",
        "Kaggle": 0.78468,
        "Improvement": 0.01913,
        "Local": 0.80223,
        "Local_Improvement": 0.01117,
    },
    {
        "Feature": 10,
        "Model": "LightGBM",
        "Kaggle": 0.78468,
        "Improvement": 0.01913,
        "Local": 0.80223,
        "Local_Improvement": 0.01229,
    },
    {
        "Feature": 10,
        "Model": "CatBoost",
        "Kaggle": 0.78468,
        "Improvement": 0.00717,
        "Local": 0.80559,
        "Local_Improvement": -0.00447,
    },
    {
        "Feature": 11,
        "Model": "Decision Tree",
        "Kaggle": 0.77751,
        "Improvement": 0.04546,
        "Local": 0.79106,
        "Local_Improvement": 0.03352,
    },
    {
        "Feature": 11,
        "Model": "XGBoost",
        "Kaggle": 0.77751,
        "Improvement": 0.02632,
        "Local": 0.79106,
        "Local_Improvement": 0.00559,
    },
    {
        "Feature": 11,
        "Model": "Random Forests",
        "Kaggle": 0.78229,
        "Improvement": 0.01674,
        "Local": 0.78994,
        "Local_Improvement": -0.00112,
    },
    {
        "Feature": 11,
        "Model": "LightGBM",
        "Kaggle": 0.77511,
        "Improvement": 0.00956,
        "Local": 0.79218,
        "Local_Improvement": 0.00224,
    },
    {
        "Feature": 11,
        "Model": "CatBoost",
        "Kaggle": 0.78229,
        "Improvement": 0.00478,
        "Local": 0.78771,
        "Local_Improvement": -0.02235,
    },
    {
        "Feature": 12,
        "Model": "Decision Tree",
        "Kaggle": 0.77751,
        "Improvement": 0.04546,
        "Local": 0.78994,
        "Local_Improvement": 0.0324,
    },
    {
        "Feature": 12,
        "Model": "XGBoost",
        "Kaggle": 0.77751,
        "Improvement": 0.02632,
        "Local": 0.79106,
        "Local_Improvement": 0.00559,
    },
    {
        "Feature": 12,
        "Model": "Random Forests",
        "Kaggle": 0.76315,
        "Improvement": -0.00239,
        "Local": 0.79665,
        "Local_Improvement": 0.00559,
    },
    {
        "Feature": 12,
        "Model": "LightGBM",
        "Kaggle": 0.77751,
        "Improvement": 0.01196,
        "Local": 0.79106,
        "Local_Improvement": 0.00112,
    },
    {
        "Feature": 12,
        "Model": "CatBoost",
        "Kaggle": 0.76315,
        "Improvement": -0.01436,
        "Local": 0.78883,
        "Local_Improvement": -0.02123,
    },
]

#  exp9: Top 10 feature combinations for all models tuned
top10_feature_combinations_tuned = [
    {
        "Model": "Decision Tree",
        "Feature_Combination": [1, 3, 4, 6, 7, 9, 10, 11],
        "Kaggle": 0.78229,
        "Improvement": 0.05024,
        "Local": 0.78994,
        "Local_Improvement": 0.0324,
    },
    {
        "Model": "Decision Tree",
        "Feature_Combination": [1, 3, 4, 6, 9, 10, 11],
        "Kaggle": 0.78229,
        "Improvement": 0.05024,
        "Local": 0.78994,
        "Local_Improvement": 0.0324,
    },
    {
        "Model": "Decision Tree",
        "Feature_Combination": [1, 3, 4, 6, 7, 10, 11],
        "Kaggle": 0.78229,
        "Improvement": 0.05024,
        "Local": 0.78994,
        "Local_Improvement": 0.0324,
    },
    {
        "Model": "Decision Tree",
        "Feature_Combination": [1, 3, 4, 7, 9, 10, 11],
        "Kaggle": 0.78229,
        "Improvement": 0.05024,
        "Local": 0.78994,
        "Local_Improvement": 0.0324,
    },
    {
        "Model": "Decision Tree",
        "Feature_Combination": [1, 3, 4, 9, 10, 11],
        "Kaggle": 0.78229,
        "Improvement": 0.05024,
        "Local": 0.78994,
        "Local_Improvement": 0.0324,
    },
    {
        "Model": "Decision Tree",
        "Feature_Combination": [1, 3, 4, 6, 10, 11],
        "Kaggle": 0.78229,
        "Improvement": 0.05024,
        "Local": 0.78994,
        "Local_Improvement": 0.0324,
    },
    {
        "Model": "Decision Tree",
        "Feature_Combination": [1, 3, 6, 11],
        "Kaggle": 0.78229,
        "Improvement": 0.05024,
        "Local": 0.7933,
        "Local_Improvement": 0.03576,
    },
    {
        "Model": "Decision Tree",
        "Feature_Combination": [3, 4, 6, 7, 8, 11],
        "Kaggle": 0.78468,
        "Improvement": 0.05263,
        "Local": 0.7933,
        "Local_Improvement": 0.03576,
    },
    {
        "Model": "Decision Tree",
        "Feature_Combination": [3, 7, 10, 11],
        "Kaggle": 0.79186,
        "Improvement": 0.05981,
        "Local": 0.79553,
        "Local_Improvement": 0.03799,
    },
    {
        "Model": "Decision Tree",
        "Feature_Combination": [1, 3, 4, 7, 8, 9, 10, 11],
        "Kaggle": 0.78229,
        "Improvement": 0.05024,
        "Local": 0.78994,
        "Local_Improvement": 0.0324,
    },  # XGBoost
    {
        "Model": "XGBoost",
        "Feature_Combination": [1, 3, 4, 7, 9, 10, 11, 12],
        "Kaggle": 0.7799,
        "Improvement": 0.02871,
        "Local": 0.79441,
        "Local_Improvement": 0.00894,
    },
    {
        "Model": "XGBoost",
        "Feature_Combination": [1, 3, 4, 9, 10, 11, 12],
        "Kaggle": 0.7799,
        "Improvement": 0.02871,
        "Local": 0.79441,
        "Local_Improvement": 0.00894,
    },
    {
        "Model": "XGBoost",
        "Feature_Combination": [1, 3, 4, 10, 11, 12],
        "Kaggle": 0.78947,
        "Improvement": 0.03827,
        "Local": 0.79665,
        "Local_Improvement": 0.01118,
    },
    {
        "Model": "XGBoost",
        "Feature_Combination": [1, 3, 4, 7, 10, 11, 12],
        "Kaggle": 0.78947,
        "Improvement": 0.03827,
        "Local": 0.79665,
        "Local_Improvement": 0.01118,
    },
    {
        "Model": "XGBoost",
        "Feature_Combination": [3, 4, 6, 7, 8, 9, 11, 12],
        "Kaggle": 0.78468,
        "Improvement": 0.03349,
        "Local": 0.79106,
        "Local_Improvement": 0.00559,
    },
    {
        "Model": "XGBoost",
        "Feature_Combination": [3, 4, 6, 7, 8, 11, 12],
        "Kaggle": 0.79186,
        "Improvement": 0.04066,
        "Local": 0.79218,
        "Local_Improvement": 0.00671,
    },
    {
        "Model": "XGBoost",
        "Feature_Combination": [3, 4, 8, 9, 11, 12],
        "Kaggle": 0.78468,
        "Improvement": 0.03349,
        "Local": 0.80223,
        "Local_Improvement": 0.01676,
    },
    {
        "Model": "XGBoost",
        "Feature_Combination": [3, 4, 8, 11, 12],
        "Kaggle": 0.79904,
        "Improvement": 0.04784,
        "Local": 0.7933,
        "Local_Improvement": 0.00783,
    },
    {
        "Model": "XGBoost",
        "Feature_Combination": [3, 4, 7, 8, 11, 12],
        "Kaggle": 0.79904,
        "Improvement": 0.04784,
        "Local": 0.7933,
        "Local_Improvement": 0.00783,
    },
    {
        "Model": "XGBoost",
        "Feature_Combination": [3, 4, 6, 8, 9, 11, 12],
        "Kaggle": 0.78468,
        "Improvement": 0.03349,
        "Local": 0.79106,
        "Local_Improvement": 0.00559,
    },
    # Random Forests
    {
        "Model": "Random Forests",
        "Feature_Combination": [1, 3, 4, 6, 7, 11],
        "Kaggle": 0.79186,
        "Improvement": 0.02631,
        "Local": 0.79441,
        "Local_Improvement": 0.00335,
    },
    {
        "Model": "Random Forests",
        "Feature_Combination": [1, 3, 4, 6, 7, 10, 11],
        "Kaggle": 0.78947,
        "Improvement": 0.02392,
        "Local": 0.79218,
        "Local_Improvement": 0.00112,
    },
    {
        "Model": "Random Forests",
        "Feature_Combination": [1, 3, 4, 10, 11],
        "Kaggle": 0.78947,
        "Improvement": 0.02392,
        "Local": 0.79218,
        "Local_Improvement": 0.00112,
    },
    {
        "Model": "Random Forests",
        "Feature_Combination": [1, 3, 4, 9, 11],
        "Kaggle": 0.79186,
        "Improvement": 0.02631,
        "Local": 0.79441,
        "Local_Improvement": 0.00335,
    },
    {
        "Model": "Random Forests",
        "Feature_Combination": [1, 3, 4, 8, 11],
        "Kaggle": 0.79186,
        "Improvement": 0.02631,
        "Local": 0.79777,
        "Local_Improvement": 0.00671,
    },
    {
        "Model": "Random Forests",
        "Feature_Combination": [1, 3, 4, 6, 11],
        "Kaggle": 0.79186,
        "Improvement": 0.02631,
        "Local": 0.79441,
        "Local_Improvement": 0.00335,
    },
    {
        "Model": "Random Forests",
        "Feature_Combination": [1, 3, 4, 6, 10, 11],
        "Kaggle": 0.78947,
        "Improvement": 0.02392,
        "Local": 0.79218,
        "Local_Improvement": 0.00112,
    },
    {
        "Model": "Random Forests",
        "Feature_Combination": [1, 3, 4, 7, 8, 11],
        "Kaggle": 0.79186,
        "Improvement": 0.02631,
        "Local": 0.79777,
        "Local_Improvement": 0.00671,
    },
    {
        "Model": "Random Forests",
        "Feature_Combination": [1, 3, 4, 7, 9, 11],
        "Kaggle": 0.79186,
        "Improvement": 0.02631,
        "Local": 0.79441,
        "Local_Improvement": 0.00335,
    },
    {
        "Model": "Random Forests",
        "Feature_Combination": [1, 3, 4, 7, 10, 11],
        "Kaggle": 0.78947,
        "Improvement": 0.02392,
        "Local": 0.79218,
        "Local_Improvement": 0.00112,
    },
    # LightGBM
    {
        "Model": "LightGBM",
        "Feature_Combination": [1, 3, 4, 6, 8, 9, 11, 12],
        "Kaggle": 0.78947,
        "Improvement": 0.02392,
        "Local": 0.80559,
        "Local_Improvement": 0.01565,
    },
    {
        "Model": "LightGBM",
        "Feature_Combination": [1, 3, 4, 6, 7, 8, 11, 12],
        "Kaggle": 0.78947,
        "Improvement": 0.02392,
        "Local": 0.80447,
        "Local_Improvement": 0.01453,
    },
    {
        "Model": "LightGBM",
        "Feature_Combination": [1, 3, 4, 6, 8, 11, 12],
        "Kaggle": 0.78947,
        "Improvement": 0.02392,
        "Local": 0.80447,
        "Local_Improvement": 0.01453,
    },
    {
        "Model": "LightGBM",
        "Feature_Combination": [1, 3, 4, 6, 7, 8, 9, 11, 12],
        "Kaggle": 0.78947,
        "Improvement": 0.02392,
        "Local": 0.80559,
        "Local_Improvement": 0.01565,
    },
    {
        "Model": "LightGBM",
        "Feature_Combination": [1, 3, 4, 7, 11],
        "Kaggle": 0.78708,
        "Improvement": 0.02153,
        "Local": 0.8,
        "Local_Improvement": 0.01006,
    },
    {
        "Model": "LightGBM",
        "Feature_Combination": [1, 3, 4, 9, 11],
        "Kaggle": 0.78708,
        "Improvement": 0.02153,
        "Local": 0.8,
        "Local_Improvement": 0.01006,
    },
    {
        "Model": "LightGBM",
        "Feature_Combination": [1, 3, 4, 11],
        "Kaggle": 0.78708,
        "Improvement": 0.02153,
        "Local": 0.8,
        "Local_Improvement": 0.01006,
    },
    {
        "Model": "LightGBM",
        "Feature_Combination": [1, 3, 4, 7, 9, 11],
        "Kaggle": 0.78708,
        "Improvement": 0.02153,
        "Local": 0.8,
        "Local_Improvement": 0.01006,
    },
    {
        "Model": "LightGBM",
        "Feature_Combination": [1, 3, 4, 8, 11],
        "Kaggle": 0.78708,
        "Improvement": 0.02153,
        "Local": 0.8,
        "Local_Improvement": 0.01006,
    },
    {
        "Model": "LightGBM",
        "Feature_Combination": [1, 3, 7, 8, 9, 11],
        "Kaggle": 0.78708,
        "Improvement": 0.02153,
        "Local": 0.79888,
        "Local_Improvement": 0.00894,
    },
    # CatBoost
    {
        "Model": "CatBoost",
        "Feature_Combination": [1, 3, 5, 8, 10, 11, 12],
        "Kaggle": 0.77272,
        "Improvement": -0.00479,
        "Local": 0.8067,
        "Local_Improvement": -0.00336,
    },
    {
        "Model": "CatBoost",
        "Feature_Combination": [1, 3, 5, 7, 8, 10, 11, 12],
        "Kaggle": 0.77272,
        "Improvement": -0.00479,
        "Local": 0.8067,
        "Local_Improvement": -0.00336,
    },
    {
        "Model": "CatBoost",
        "Feature_Combination": [1, 5, 6, 7, 8, 10, 11, 12],
        "Kaggle": 0.78468,
        "Improvement": 0.00717,
        "Local": 0.80447,
        "Local_Improvement": -0.00559,
    },
    {
        "Model": "CatBoost",
        "Feature_Combination": [1, 5, 6, 8, 10, 11, 12],
        "Kaggle": 0.78468,
        "Improvement": 0.00717,
        "Local": 0.80447,
        "Local_Improvement": -0.00559,
    },
    {
        "Model": "CatBoost",
        "Feature_Combination": [1, 3, 4, 8, 9, 11, 12],
        "Kaggle": 0.79665,
        "Improvement": 0.01913,
        "Local": 0.79777,
        "Local_Improvement": -0.01229,
    },
    {
        "Model": "CatBoost",
        "Feature_Combination": [3, 5, 8, 10, 11, 12],
        "Kaggle": 0.78708,
        "Improvement": 0.00956,
        "Local": 0.81341,
        "Local_Improvement": 0.00335,
    },
    {
        "Model": "CatBoost",
        "Feature_Combination": [3, 5, 8, 9, 10, 11],
        "Kaggle": 0.78468,
        "Improvement": 0.00717,
        "Local": 0.81229,
        "Local_Improvement": 0.00223,
    },
    {
        "Model": "CatBoost",
        "Feature_Combination": [1, 4, 5, 8, 11, 12],
        "Kaggle": 0.77511,
        "Improvement": -0.0024,
        "Local": 0.81006,
        "Local_Improvement": 0.0,
    },
    {
        "Model": "CatBoost",
        "Feature_Combination": [1, 4, 5, 9, 11, 12],
        "Kaggle": 0.77751,
        "Improvement": 0.0,
        "Local": 0.80559,
        "Local_Improvement": -0.00447,
    },
    {
        "Model": "CatBoost",
        "Feature_Combination": [3, 4, 5, 6, 8, 10, 11],
        "Kaggle": 0.78708,
        "Improvement": 0.00956,
        "Local": 0.8067,
        "Local_Improvement": -0.00336,
    },
]

#  exp10: Best feature combinations for all models tuned
best_feature_combinations_tuned = [
    {
        "Model": "Decision Tree",
        "Feature_Combination": [1, 3, 4, 6, 7, 9, 10, 11],
        "Kaggle": 0.78229,
        "Improvement": 0.05024,
        "Local": 0.78994,
        "Local_Improvement": 0.0324,
    },
    {
        "Model": "XGBoost",
        "Feature_Combination": [1, 3, 4, 7, 9, 10, 11, 12],
        "Kaggle": 0.7799,
        "Improvement": 0.02871,
        "Local": 0.79441,
        "Local_Improvement": 0.00894,
    },
    {
        "Model": "Random Forests",
        "Feature_Combination": [1, 3, 4, 6, 7, 11],
        "Kaggle": 0.79186,
        "Improvement": 0.02631,
        "Local": 0.79441,
        "Local_Improvement": 0.00335,
    },
    {
        "Model": "LightGBM",
        "Feature_Combination": [1, 3, 4, 6, 8, 9, 11, 12],
        "Kaggle": 0.78947,
        "Improvement": 0.02392,
        "Local": 0.80559,
        "Local_Improvement": 0.01565,
    },
    {
        "Model": "CatBoost",
        "Feature_Combination": [1, 3, 5, 8, 10, 11, 12],
        "Kaggle": 0.77272,
        "Improvement": -0.00479,
        "Local": 0.8067,
        "Local_Improvement": -0.00336,
    },
]

#  exp11: All features combined for all models tuned
all_features_combined_tuned = [
    {
        "Model": "Decision Tree",
        "Feature_Combination": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
        "Kaggle": 0.78229,
        "Improvement": 0.05024,
        "Local": 0.80335,
        "Local_Improvement": 0.04581,
    },
    {
        "Model": "XGBoost",
        "Feature_Combination": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
        "Kaggle": 0.78468,
        "Improvement": 0.03349,
        "Local": 0.80447,
        "Local_Improvement": 0.019,
    },
    {
        "Model": "Random Forests",
        "Feature_Combination": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
        "Kaggle": 0.78708,
        "Improvement": 0.02153,
        "Local": 0.80894,
        "Local_Improvement": 0.01788,
    },
    {
        "Model": "LightGBM",
        "Feature_Combination": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
        "Kaggle": 0.77751,
        "Improvement": 0.01196,
        "Local": 0.8067,
        "Local_Improvement": 0.01676,
    },
    {
        "Model": "CatBoost",
        "Feature_Combination": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
        "Kaggle": 0.76076,
        "Improvement": -0.01675,
        "Local": 0.8,
        "Local_Improvement": -0.01006,
    },
]

fixed_palette = {
    "Decision Tree": "#1f77b4",
    "XGBoost": "#ff7f0e",
    "Random Forests": "#2ca02c",
    "LightGBM": "#d62728",
    "CatBoost": "#9467bd",
}

# Common plot style
sns.set(style="whitegrid")
plt.rcParams.update(
    {
        "axes.titlesize": 12,
        "axes.labelsize": 10,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 9,
    }
)


def generate_fixed_palette(keys):
    palette_colors = sns.color_palette("tab10", n_colors=len(set(keys)))
    return dict(zip(sorted(set(keys)), palette_colors))


def annotate_bars(ax):
    for p in ax.patches:
        height = p.get_height()
        ax.annotate(
            f"{height:.3f}",
            (p.get_x() + p.get_width() / 2.0, height),
            ha="center",
            va="bottom",
            fontsize=8,
            xytext=(0, 3),
            textcoords="offset points",
        )


def exp_1_plot_baseline_comparison(
    data, title="Experiment 1: Baseline Model Accuracy", save_path=None
):
    df = pd.DataFrame(data)
    fig, ax = plt.subplots(figsize=(10, 6))
    bar_width = 0.35
    x = range(len(df))
    kaggle_colors = [fixed_palette[m] for m in df["Model"]]
    local_colors = kaggle_colors

    bars1 = ax.bar(
        [i - bar_width / 2 for i in x],
        df["Kaggle"],
        width=bar_width,
        label="Kaggle",
        color=kaggle_colors,
    )
    bars2 = ax.bar(
        [i + bar_width / 2 for i in x],
        df["Local"],
        width=bar_width,
        label="Local",
        color=local_colors,
        alpha=0.75,
    )

    ax.set_xlabel("Model")
    ax.set_ylabel("Accuracy")
    ax.set_title(title + "\n(Deterministic run with fixed seed = 42)")
    ax.set_xticks(list(x))
    ax.set_xticklabels(df["Model"])
    ax.set_ylim(0.6, 0.9)
    ax.yaxis.grid(True, linestyle="--", alpha=0.5)
    ax.legend()
    annotate_bars(ax)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300)
    else:
        plt.show()


def exp_2_and_8_plot_single_feature_accuracy(data, tuned=False):
    df = pd.DataFrame(data)
    exp_num = 8 if tuned else 2
    suffix = " (Tuned)" if tuned else " (Untuned)"

    for metric in ["Local", "Kaggle"]:
        plt.figure(figsize=(12, 6))
        ax = sns.barplot(
            data=df, x="Feature", y=metric, hue="Model", palette=fixed_palette
        )
        ax.set_title(
            f"Experiment {exp_num}: {metric} Accuracy per Feature{suffix}\n(Seed=42, deterministic)"
        )
        ax.set_ylabel(f"{metric} Accuracy")
        ax.set_xlabel("Feature Number")
        ax.set_ylim(0.6, 0.9)
        ax.yaxis.grid(True, linestyle="--", alpha=0.5)
        plt.legend(title="Model", bbox_to_anchor=(1.02, 1), loc="upper left")
        plt.tight_layout()
        plt.show()


def exp_3_and_9_plot_top10_combination_accuracy(data, tuned=False):
    df = pd.DataFrame(data)
    df["Combination_ID"] = df.index.astype(str)
    exp_num = 9 if tuned else 3
    suffix = " (Tuned)" if tuned else " (Untuned)"

    for metric in ["Local", "Kaggle"]:
        plt.figure(figsize=(18, 7))
        ax = sns.barplot(
            data=df, x="Combination_ID", y=metric, hue="Model", palette=fixed_palette
        )
        ax.set_title(
            f"Experiment {exp_num}: {metric} Accuracy - Top 10 Feature Combinations{suffix}\n(Seed=42, deterministic)"
        )
        ax.set_ylabel(f"{metric} Accuracy")
        ax.set_xlabel("Feature Combination")
        ax.set_ylim(0.6, 0.9)
        ax.set_xticklabels(
            df["Feature_Combination"], rotation=60, ha="right", fontsize=7
        )
        ax.yaxis.grid(True, linestyle="--", alpha=0.5)
        plt.legend(title="Model", bbox_to_anchor=(1.02, 1), loc="upper left")
        plt.tight_layout()
        plt.show()


def exp_4_and_10_plot_best_combinations_accuracy(data, tuned=False):
    df = pd.DataFrame(data)
    exp_num = 10 if tuned else 4
    suffix = " (Tuned)" if tuned else " (Untuned)"

    for metric in ["Local", "Kaggle"]:
        plt.figure(figsize=(10, 6))
        ax = sns.barplot(data=df, x="Model", y=metric, palette=fixed_palette)
        ax.set_title(
            f"Experiment {exp_num}: Best Feature Combination per Model{suffix} - {metric} Accuracy\n(Seed=42, deterministic)"
        )
        ax.set_ylabel(f"{metric} Accuracy")
        ax.set_xlabel("Model")
        ax.set_ylim(0.6, 0.9)
        ax.yaxis.grid(True, linestyle="--", alpha=0.5)
        annotate_bars(ax)
        plt.tight_layout()
        plt.show()


def exp_5_and_11_plot_all_features_combined_accuracy(data, tuned=False):
    df = pd.DataFrame(data)
    exp_num = 11 if tuned else 5
    suffix = " (Tuned)" if tuned else " (Untuned)"

    for metric in ["Local", "Kaggle"]:
        plt.figure(figsize=(10, 6))
        ax = sns.barplot(data=df, x="Model", y=metric, palette=fixed_palette)
        ax.set_title(
            f"Experiment {exp_num}: All Features Combined {suffix} - {metric} Accuracy\n(Seed=42, deterministic)"
        )
        ax.set_ylabel(f"{metric} Accuracy")
        ax.set_xlabel("Model")
        ax.set_ylim(0.6, 0.9)
        ax.yaxis.grid(True, linestyle="--", alpha=0.5)
        annotate_bars(ax)
        plt.tight_layout()
        plt.show()


def exp_6_top3_feature_each_model(data):
    df = pd.DataFrame(data)
    df["Label"] = df["Model"] + " - " + df["Feature Name"]
    fixed_palette = generate_fixed_palette(df["Label"])

    plt.figure(figsize=(14, 6))
    ax = sns.barplot(data=df, x="Label", y="Importance (%)", palette=fixed_palette)
    plt.title(
        "Experiment 6: Top 3 Features per Model (Feature Importance)\nStatic single-run result with seed = 42"
    )
    plt.ylabel("Importance (%)")
    plt.xlabel("Model - Feature")
    plt.xticks(rotation=45, ha="right")
    ax.yaxis.grid(True, linestyle="--", alpha=0.5)
    annotate_bars(ax)
    plt.tight_layout()
    plt.show()


def exp_7_baseline_model_tuned(
    data, title="Experiment 7: Baseline Model (Tuned) Accuracy", save_path=None
):
    df = pd.DataFrame(data)
    fig, ax = plt.subplots(figsize=(10, 6))
    bar_width = 0.35
    x = range(len(df))

    kaggle_colors = [fixed_palette[m] for m in df["Model"]]
    local_colors = [fixed_palette[m] for m in df["Model"]]

    bars1 = ax.bar(
        [i - bar_width / 2 for i in x],
        df["Kaggle"],
        width=bar_width,
        label="Kaggle",
        color=kaggle_colors,
    )
    bars2 = ax.bar(
        [i + bar_width / 2 for i in x],
        df["Local"],
        width=bar_width,
        label="Local",
        color=local_colors,
        alpha=0.75,
    )

    ax.set_xlabel("Model")
    ax.set_ylabel("Accuracy")
    ax.set_title(title + "\n(Deterministic run with fixed seed = 42)")
    ax.set_xticks(list(x))
    ax.set_xticklabels(df["Model"])
    ax.set_ylim(0.6, 0.9)
    ax.yaxis.grid(True, linestyle="--", alpha=0.5)
    ax.legend()

    for bar in bars1 + bars2:
        height = bar.get_height()
        ax.annotate(
            f"{height:.3f}",
            xy=(bar.get_x() + bar.get_width() / 2, height),
            xytext=(0, 3),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=8,
        )

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300)
        print(f"✅ Plot saved to {save_path}")
    else:
        plt.show()


def plot_combined_accuracy(data, experiment_number, title_suffix):
    df = pd.DataFrame(data)
    fig, ax = plt.subplots(figsize=(10, 6))
    bar_width = 0.35
    x = range(len(df))

    kaggle_colors = [fixed_palette[m] for m in df["Model"]]
    local_colors = kaggle_colors

    bars1 = ax.bar(
        [i - bar_width / 2 for i in x],
        df["Kaggle"],
        width=bar_width,
        label="Kaggle",
        color=kaggle_colors,
    )
    bars2 = ax.bar(
        [i + bar_width / 2 for i in x],
        df["Local"],
        width=bar_width,
        label="Local",
        color=local_colors,
        alpha=0.75,
    )

    ax.set_xlabel("Model")
    ax.set_ylabel("Accuracy")
    ax.set_title(
        f"Experiment {experiment_number}: {title_suffix}\n(Local & Kaggle Accuracy - Seed=42)"
    )
    ax.set_xticks(list(x))
    ax.set_xticklabels(df["Model"])
    ax.set_ylim(0.6, 0.9)
    ax.yaxis.grid(True, linestyle="--", alpha=0.5)
    ax.legend()
    annotate_bars(ax)
    plt.tight_layout()
    plt.show()


# Wrappers for Experiments 4, 5, 10, 11
def exp_4_plot_combined_accuracy(data):
    plot_combined_accuracy(data, 4, "Best Feature Combination per Model (Untuned)")


def exp_5_plot_combined_accuracy(data):
    plot_combined_accuracy(data, 5, "All Features Combined (Untuned)")


def exp_10_plot_combined_accuracy(data):
    plot_combined_accuracy(data, 10, "Best Feature Combination per Model (Tuned)")


def exp_11_plot_combined_accuracy(data):
    plot_combined_accuracy(data, 11, "All Features Combined (Tuned)")


# exp_1_plot_baseline_comparison(baseline_evaluation)
# exp_2_and_8_plot_single_feature_accuracy(single_feature_data_untuned)
# exp_3_and_9_plot_top10_combination_accuracy(top10_feature_combinations_untuned)
# exp_4_plot_combined_accuracy(best_feature_combinations_untuned)
# exp_5_plot_combined_accuracy(all_features_combined_untuned)
# exp_6_top3_feature_each_model(top3_features_per_model)
# exp_7_baseline_model_tuned(baseline_model_tuned)
# exp_2_and_8_plot_single_feature_accuracy(single_feature_data_tuned, tuned=True)
# exp_3_and_9_plot_top10_combination_accuracy(
#     top10_feature_combinations_tuned, tuned=True
# )
exp_10_plot_combined_accuracy(best_feature_combinations_tuned)
exp_11_plot_combined_accuracy(all_features_combined_tuned)
