import streamlit as st
from PIL import Image
from pathlib import Path
import torch
from backbone import UNet, check_img_size, make_prediction, load_model



st.header("Рассчёт площади застройки территории по снимку", divider="gray", width="content")
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = UNet().to(DEVICE)
checkpoints = load_model()
model.load_state_dict(checkpoints['model_state_dict'])
model.eval()

if 'input_image' not in st.session_state:
    st.session_state['input_image'] = None
if 'image_dimension' not in st.session_state:
    st.session_state['image_dimension'] = None

uploaded_file = st.file_uploader("Загрузите орторектифицированный аэрофотоснимок, размеры которого \
                                по ширине и высоте должны быть кратны 250 пикселям. В случае несоответствия исходного \
                                изображения указанным требованиям, будет выполнена центральная обрезка до ближайших \
                                подходящих размеров, кратных 250.")
dimension = st.number_input("Введите разрешение (м/пиксель)", min_value=0.0, value=None, placeholder="Введите значение...")

if (uploaded_file is not None) and (st.session_state['input_image'] is not uploaded_file):
    st.session_state['input_image'] = uploaded_file
if (dimension is not None) and (st.session_state['image_dimension'] is not dimension):
    st.session_state['image_dimension'] = dimension

if (st.session_state['input_image'] is not None) and (st.session_state['image_dimension'] is not None):
    img = Image.open(st.session_state['input_image'])
    img = check_img_size(img)
    st.image(img, caption='Входное изображение', use_container_width=True)
    if st.button("Рассчитать площадь застройки"):
        with st.spinner("Wait for it...", show_time=True):
            all_area, mask = make_prediction(model, img, st.session_state['image_dimension'])
        st.write(f"Площадь застройки: {all_area:,.2f} м²")
        st.image(mask, caption='Маска застройки', use_container_width=True)

