def select_model_key():
    print("\n🤖 Choose a model:")
    print("1. Decision Tree")
    print("2. XGBoost")
    print("3. Random Forest")
    print("4. LightGBM")
    print("5. CatBoost")
    model_choice = input("Enter model number (1–5): ").strip()
    return {"1": "dt", "2": "xgb", "3": "rf", "4": "lgbm", "5": "cb"}.get(model_choice)

def prompt_all_or_one():
    return input("Run [a]ll models or [o]ne model? ").strip().lower()