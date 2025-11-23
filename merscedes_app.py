import streamlit as st
import pandas as pd
import numpy as np
import pickle

# Загрузка сохраненной модели
@st.cache_resource
def load_model():
    with open('simple_mercedes_model.pkl', 'rb') as f:
        model_data = pickle.load(f)
    return model_data

# Функция для предсказания цены
def predict_car_price(model, year, mileage, model_data):
    """
    Предсказывает цену автомобиля Mercedes
    """
    # Получаем компоненты модели
    loaded_model = model_data['model']
    scaler = model_data['scaler']
    features = model_data['features']
    model_prices = model_data['model_prices']

    # Создаем DataFrame с входными данными
    input_data = {
        'model': model,
        'year': year,
        'mileage': mileage,
        'age': 2025 - year,
    }

    # Вычисляем model_price
    if model in model_prices:
        input_data['model_price'] = model_prices[model]
    else:
        input_data['model_price'] = np.mean(list(model_prices.values()))

    # Создаем признаки
    input_data['log_mileage'] = np.log1p(mileage)
    input_data['is_new'] = 1 if (2025 - year) == 0 else 0
    input_data['high_mileage'] = 1 if mileage > 100000 else 0
    input_data['premium'] = 1 if input_data['model_price'] > 10000000 else 0

    # Создаем DataFrame и предсказываем
    df_input = pd.DataFrame([input_data])
    X_input = df_input[features]
    X_scaled = scaler.transform(X_input)
    log_prediction = loaded_model.predict(X_scaled)[0]
    predicted_price = np.expm1(log_prediction)

    return round(predicted_price)

# Настройка страницы
st.set_page_config(
    page_title="Mercedes Price Predictor",
    page_icon="🚗",
    layout="centered"
)

# Заголовок приложения
st.title("🚗 Mercedes-Benz Price Predictor")
st.markdown("### Предсказание стоимости автомобилей Mercedes-Benz")

# Загрузка модели
try:
    model_data = load_model()
    available_models = list(model_data['model_prices'].keys())
    
    # Создаем колонки для ввода данных
    col1, col2 = st.columns(2)
    
    with col1:
        # Выбор модели из выпадающего списка
        selected_model = st.selectbox(
            "Выберите модель Mercedes:",
            options=sorted(available_models),
            help="Выберите модель из списка доступных"
        )
        
        # Ввод года выпуска с валидацией
        selected_year = st.number_input(
            "Год выпуска:",
            min_value=1990,
            max_value=2025,
            value=2023,
            step=1,
            help="Год выпуска от 1990 до 2025"
        )
    
    with col2:
        # Ввод пробега с валидацией
        selected_mileage = st.number_input(
            "Пробег (км):",
            min_value=0,
            max_value=500000,
            value=10000,
            step=1000,
            help="Пробег от 0 до 500,000 км"
        )
        
        # Отображаем возраст автомобиля
        car_age = 2025 - selected_year
        st.info(f"Возраст автомобиля: {car_age} лет")

    # Кнопка для предсказания
    if st.button("🎯 Предсказать цену", type="primary"):
        with st.spinner("Вычисляем стоимость..."):
            try:
                predicted_price = predict_car_price(
                    selected_model, 
                    selected_year, 
                    selected_mileage, 
                    model_data
                )
                
                # Отображаем результат
                st.success("### Результат предсказания")
                
                # Красивое отображение цены
                col_a, col_b, col_c = st.columns([1, 2, 1])
                with col_b:
                    st.metric(
                        label=f"{selected_model} {selected_year}",
                        value=f"{predicted_price:,.0f} ₽",
                        help=f"Пробег: {selected_mileage:,} км"
                    )
                
                # Дополнительная информация
                with st.expander("📊 Детали расчета"):
                    st.write(f"**Модель:** {selected_model}")
                    st.write(f"**Год выпуска:** {selected_year}")
                    st.write(f"**Пробег:** {selected_mileage:,} км")
                    st.write(f"**Возраст:** {car_age} лет")
                    st.write(f"**Тип:** {'Премиум' if model_data['model_prices'][selected_model] > 10000000 else 'Стандарт'}")
                    st.write(f"**Состояние:** {'Новый' if car_age == 0 else 'С пробегом'}")
                    
            except Exception as e:
                st.error(f"Ошибка при предсказании: {str(e)}")

    # Раздел с примерами
    st.markdown("---")
    st.subheader("📋 Примеры для тестирования")
    
    example_col1, example_col2, example_col3 = st.columns(3)
    
    examples = [
        {"model": "CLA-класс", "year": 2019, "mileage": 48994},
        {"model": "GLE-класс", "year": 2025, "mileage": 10},
        {"model": "S-класс", "year": 2021, "mileage": 68000}
    ]
    
    for i, example in enumerate(examples):
        with [example_col1, example_col2, example_col3][i]:
            if st.button(f"Пример {i+1}", key=f"example_{i}"):
                try:
                    price = predict_car_price(
                        example["model"], 
                        example["year"], 
                        example["mileage"], 
                        model_data
                    )
                    st.success(f"{example['model']} {example['year']}\n"
                              f"Пробег: {example['mileage']:,} км\n"
                              f"Цена: {price:,.0f} ₽")
                except Exception as e:
                    st.error(f"Ошибка: {str(e)}")

    # Информация о модели
    st.markdown("---")
    with st.expander("ℹ️ О модели"):
        st.write("""
        **Метрики модели:**
        - R² = 0.8732
        - Средняя ошибка: 6.8%
        - Медианная ошибка: 4.4%
        
        **Используемые признаки:**
        - Средняя цена модели
        - Год выпуска
        - Логарифм пробега
        - Возраст автомобиля
        - Новый/с пробегом
        - Высокий пробег
        - Премиум класс
        """)

except FileNotFoundError:
    st.error("❌ Файл модели 'simple_mercedes_model.pkl' не найден!")
    st.info("Убедитесь, что файл модели находится в той же директории, что и это приложение")

except Exception as e:
    st.error(f"❌ Ошибка при загрузке модели: {str(e)}")