import pandas as pd
import statsmodels.formula.api as smf

df = pd.read_csv("austin_home_sales.csv")

model = smf.ols(
    "Sale_Price ~ Sq_Footage + Bedrooms + Home_Age_Years + Dist_Downtown_Mi",
    data=df,
).fit()

model_2 = smf.ols(
    "Sale_Price ~ Sq_Footage + Home_Age_Years + Dist_Downtown_Mi",
    data=df,
).fit()


def predict_price(sq_footage, bedrooms, home_age_years, dist_downtown_mi):
    features = pd.DataFrame(
        {
            "Sq_Footage": [sq_footage],
            "Bedrooms": [bedrooms],
            "Home_Age_Years": [home_age_years],
            "Dist_Downtown_Mi": [dist_downtown_mi],
        }
    )
    return float(model.predict(features).iloc[0])


def should_buy(list_price, sq_footage, bedrooms, home_age_years, dist_downtown_mi):
    predicted = predict_price(
        sq_footage, bedrooms, home_age_years, dist_downtown_mi
    )
    return list_price < predicted


def predict_price_2(sq_footage, home_age_years, dist_downtown_mi):
    features = pd.DataFrame(
        {
            "Sq_Footage": [sq_footage],
            "Home_Age_Years": [home_age_years],
            "Dist_Downtown_Mi": [dist_downtown_mi],
        }
    )
    return float(model_2.predict(features).iloc[0])


def should_buy_2(list_price, sq_footage, home_age_years, dist_downtown_mi):
    predicted = predict_price_2(sq_footage, home_age_years, dist_downtown_mi)
    return list_price < predicted


print(model.summary())

for name, coefficient in model.params.items():
    print(f"{name} coefficient: {coefficient:.0f}")

for name, p_value in model.pvalues.items():
    print(f"{name} p-value: {p_value:.4f}")

print(f"R-squared: {model.rsquared:.4f}")
print(f"Adjusted R-squared: {model.rsquared_adj:.4f}")

sq_footage = 2000
bedrooms = 3
home_age_years = 12
dist_downtown_mi = 2.5
list_price = 380000

predicted_price = predict_price(
    sq_footage, bedrooms, home_age_years, dist_downtown_mi
)
decision = should_buy(
    list_price, sq_footage, bedrooms, home_age_years, dist_downtown_mi
)

print(f"Predicted price: {predicted_price}")
print(f"List price: {list_price}")
print(f"Difference: {predicted_price - list_price}")
print(f"Decision: {decision}")

print("Second model")

for name, coefficient in model_2.params.items():
    print(f"{name} coefficient: {coefficient:.0f}")

for name, p_value in model_2.pvalues.items():
    print(f"{name} p-value: {p_value:.4f}")

print(f"R-squared: {model_2.rsquared:.4f}")
print(f"Adjusted R-squared: {model_2.rsquared_adj:.4f}")

predicted_price_2 = predict_price_2(sq_footage, home_age_years, dist_downtown_mi)
decision_2 = should_buy_2(list_price, sq_footage, home_age_years, dist_downtown_mi)

print(f"Predicted price: {predicted_price_2}")
print(f"List price: {list_price}")
print(f"Difference: {predicted_price_2 - list_price}")
print(f"Decision: {decision_2}")
