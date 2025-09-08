# نصب کتابخانه‌ها (فقط یک بار لازم است)
!pip install pyomo
!apt-get install -y -qq glpk-utils

# آپلود فایل اکسل
from google.colab import files
uploaded = files.upload()

# بارگذاری فایل اکسل
import pandas as pd
file_name = list(uploaded.keys())[0]
xls = pd.ExcelFile(file_name)
df = xls.parse(xls.sheet_names[0])

# ترانهاده داده‌ها: زمان روی ردیف، دارایی‌ها روی ستون
df = df.T

# جدا کردن شاخص کل (ستون آخر) و حذفش از دارایی‌ها
market_prices = df.iloc[:, -1]
df = df.iloc[:, :-1]

# محاسبه بازده‌ها
returns = df.pct_change().dropna().reset_index(drop=True)                        #محاسبه بازده برای هریک از شاخص ها در هر بازه زمانی
market_return = market_prices.pct_change().dropna().reset_index(drop=True)       #حاسبه بازده برای هریک از شاخص های بازار  ها در هر بازه زمانی
# آماده‌سازی مدل Pyomo
from pyomo.environ import *
import numpy as np
#لیست کردن تمامی داده ها
assets = list(returns.columns)
T = returns.shape[0]                #مشخص کردن تمامی بازه های زمانی
n = len(assets)                     #مشخص کردن تمام دارایی ها
K_value = 0.02                      # حداکثر underperformance قابل‌قبول

model = ConcreteModel()
model.Assets = RangeSet(0, n-1)     #تعداد n تا دارایی
model.Times = RangeSet(0, T-1)      #تعداد T تا تایم استپ داریم
r = returns.values
r_market = market_return.values     #تبدیل کردن بازده های مختلف یک شاخص به آرایه
model.x = Var(model.Assets, domain=NonNegativeReals)                                          # متغیر های تصمیم  (از هر سهم چقدر بخره که  پرتفو بسازه ))

def objective_rule(m):
    return sum(sum(r[t, i] * m.x[i] for i in m.Assets) - r_market[t] for t in m.Times) / T    #معادله تابع هدف
model.objective = Objective(rule=objective_rule, sense=maximize)                              # نوشتن تابع هدف ماکزیموم سازی سود میانگین  شاخص ها

def underperformance_constraint(m, t):
    return sum(r[t, i] * m.x[i] for i in m.Assets) >= r_market[t] - K_value                   # نوشتن تابع هدف دوم که هدف آن مینیموم کردن بدترین عملکرد بوده
model.Underperformance = Constraint(model.Times, rule=underperformance_constraint)            #در مقاله گفته شده که به جای تابع دو هدفه از تابع تک هدفه استفاده میشود  و تابع هدف دوم بصورت قید نوشته می شود

model.WeightSum = Constraint(expr=sum(model.x[i] for i in model.Assets) == 1)                 # درصدی از هر شاخص می توان خرید که مجمو100% سهم مارا تشکیل میدهند

# حل مدل با GLPK
solver = SolverFactory('glpk')
results = solver.solve(model, tee=True)

# استخراج وزن‌های بهینه
weights = np.array([value(model.x[i]) for i in model.Assets])
nonzero_weights = {assets[i]: w for i, w in enumerate(weights) if w > 1e-4}

# نمایش وزن‌های غیرصفر
print("📊 وزن دارایی‌های فعال در پرتفوی:")
for asset, weight in nonzero_weights.items():
    print(f"{asset}: {weight:.4f}")

# مقایسه عملکرد پرتفوی با بازار
portfolio_returns = returns.values @ weights   #بازده هر شاخص را در وزن شاخص ضرب میکنیم  ضرب به صورت ضرب ماتریسی انجام می شود
comparison_df = pd.DataFrame({                 #یک دیتا فریم از ریترن های شاخص وهر پرتفو و شاخص کل می سازد که بتونه اونا رو مقایسه کنه
    "Portfolio": portfolio_returns,            #اطلاعات پرتفولیو و مارکت رو جدا میکنه
    "Market": market_return.values
})
comparison_df["Portfolio_CumReturn"] = (1 + comparison_df["Portfolio"]).cumprod()   # ضرب تجمعی رو برای مارکت و پرتفو محاسبه محاسبه میکنیم و یک ستون به دیتا فریم اضافه می کنیم  تا با هم مقایسه بشن
comparison_df["Market_CumReturn"] = (1 + comparison_df["Market"]).cumprod()         #از ضرب تجمعی استفاده شده چون سود روز های قبل برداشت نمیشه و مجدد سرمایه گذاری میشه

# نمایش بازده تجمعی اولیه
comparison_df[["Portfolio_CumReturn", "Market_CumReturn"]].head()

# نمودار بازده تجمعی
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 6))
plt.plot(comparison_df["Portfolio_CumReturn"], label="Portfolio")
plt.plot(comparison_df["Market_CumReturn"], label="Market")
plt.title("Cumulative Return: Portfolio vs Market")
plt.xlabel("Time Period")
plt.ylabel("Cumulative Return")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
