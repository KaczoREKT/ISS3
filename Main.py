import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# Dane
pwm_values = np.array([50, 100, 150, 200, 255])
speed_values = np.array([1.2, 2.5, 3.8, 5.0, 6.1])

# Dopasowanie modelu regresji liniowej
model = LinearRegression()
pwm_values_reshaped = pwm_values.reshape(-1, 1)
model.fit(pwm_values_reshaped, speed_values)

# Wartości przewidywane przez model
predicted_speeds = model.predict(pwm_values_reshaped)

# Rysowanie wykresu
plt.scatter(pwm_values, speed_values, color='blue', label='Pomiary')
plt.plot(pwm_values, predicted_speeds, color='red', label='Regresja liniowa')
plt.xlabel('PWM')
plt.ylabel('Prędkość (cm/s)')
plt.legend()
plt.title('Zależność prędkości od PWM')
plt.show()

# Współczynniki regresji
print("Współczynnik kierunkowy (a):", model.coef_[0])
print("Wyraz wolny (b):", model.intercept_)