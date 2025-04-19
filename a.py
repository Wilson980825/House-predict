import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

# 讀取資料
file_path = "Taipei_house.csv"
df = pd.read_csv(file_path)

# 選擇幾個比較關鍵的欄位
features = ['建物總面積', '屋齡', '房數', '廳數', '衛數']
target = '總價'

# 建立特徵與目標變數
X = df[features]
y = df[target]

# 分割資料集（訓練 80%，測試 20%）
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 建立並訓練模型
model = LinearRegression()
model.fit(X_train, y_train)

# 整合特色功能：讓使用者輸入資訊進行預測
print("歡迎使用台北市房價預測系統")
print("請依照提示輸入你家的資訊：")

try:
    building_area = float(input("建物總面積（坪）: "))
    age = float(input("屋齡（年）: "))
    rooms = int(input("房數: "))
    living_rooms = int(input("廳數: "))
    bathrooms = int(input("衛數: "))

    # 預測價格
    user_input = [[building_area, age, rooms, living_rooms, bathrooms]]
    predicted_price = model.predict(user_input)
    print(f"\n✨ 預測你家的價格大約是：{int(predicted_price[0])} 萬台幣")

except Exception as e:
    print("輸入錯誤，請確認輸入的數值格式正確。")