#!/usr/bin/env python
# coding: utf-8

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
# import networkx as nx # دیگر برای رسم گراف اصلی استفاده نمی‌شود، اما ممکن است برای تحلیل‌های دیگر مفید باشد
# import matplotlib.pyplot as plt # دیگر برای رسم گراف اصلی استفاده نمی‌شود
from PIL import Image
import numpy as np
import io
from collections import defaultdict
import arabic_reshaper
from bidi.algorithm import get_display
# import matplotlib.font_manager as fm # دیگر برای فونت گراف اصلی استفاده نمی‌شود
import os
import traceback
# from matplotlib.colors import LinearSegmentedColormap # دیگر برای گراف اصلی استفاده نمی‌شود
# import matplotlib as mpl # دیگر برای گراف اصلی استفاده نمی‌شود
# from matplotlib.lines import Line2D # دیگر برای لجند گراف اصلی استفاده نمی‌شود
import base64 # برای تبدیل تصویر به فرمت قابل استفاده در Plotly

# تنظیم عنوان و توضیحات داشبورد
st.set_page_config(page_title="داشبورد تحلیل ظرفیت مترو", layout="wide")
st.title("داشبورد تحلیل ظرفیت مترو")
st.markdown("این داشبورد برای تحلیل ظرفیت قطارهای مترو بین ایستگاه‌ها در ساعات مختلف طراحی شده است.")

# --- Font Configuration for Persian Text (Used for Plotly titles, labels if needed elsewhere) ---
# Plotly معمولا از فونت‌های وب یا سیستم استفاده می‌کند.  برای کاراکترها کافی است.
# persian_font_name = 'nazanin' 
# persian_font_path = 'E:/dashbord/B-NAZANIN.ttf'
# persian_font = None
# font_display_name = "پیش‌فرض"

# try:
#     # این بخش برای Matplotlib بود، در Plotly فونت‌ها متفاوت مدیریت می‌شوند.
#     # اگر نیاز به تنظیم فونت خاصی در Plotly دارید، باید در layout.font انجام شود.
#     pass
# except Exception as e:
#     st.sidebar.warning(f"خطا در بارگذاری فونت: {e}")


# --- Data Loading Functions ---
def load_data(file_path, file_type='csv', sheet_name=0):
    try:
        if file_type == 'csv':
            return pd.read_csv(file_path)
        elif file_type == 'excel':
            return pd.read_excel(file_path, sheet_name=sheet_name)
    except FileNotFoundError:
        st.error(f"فایل در مسیر '{file_path}' یافت نشد.")
    except Exception as e:
        st.error(f"خطا در بارگذاری فایل '{file_path}': {e}")
    return None

def load_network_data(uploaded_file, default_path, file_type='csv', sheet_name=0):
    if uploaded_file is not None:
        try:
            if file_type == 'csv':
                return pd.read_csv(uploaded_file)
            elif file_type == 'excel':
                return pd.read_excel(uploaded_file, sheet_name=sheet_name)
        except Exception as e:
            st.error(f"خطا در بارگذاری فایل آپلود شده: {e}")
            return load_data(default_path, file_type, sheet_name)
    return load_data(default_path, file_type, sheet_name)

def load_positions_and_image(uploaded_pos_file, default_pos_path, uploaded_img_file, default_img_path):
    positions_df = load_network_data(uploaded_pos_file, default_pos_path, 'excel')
    img = None
    if uploaded_img_file is not None:
        try:
            img = Image.open(uploaded_img_file)
        except Exception as e:
            st.error(f"خطا در بارگذاری تصویر آپلود شده: {e}")
            try:
                img = Image.open(default_img_path)
            except Exception:
                st.warning(f"تصویر پیش‌فرض در مسیر '{default_img_path}' نیز یافت نشد.")
    else:
        try:
            img = Image.open(default_img_path)
        except FileNotFoundError:
            st.warning(f"تصویر پیش‌فرض در مسیر '{default_img_path}' یافت نشد.")
        except Exception as e:
            st.error(f"خطا در بارگذاری تصویر پیش‌فرض: {e}")
    return positions_df, img

# --- Sidebar for File Uploads ---
st.sidebar.header(("بارگذاری فایل‌ها"))
uploaded_capacity_file = st.sidebar.file_uploader(("فایل ظرفیت (CSV)"), type="csv")
uploaded_line_file = st.sidebar.file_uploader(("فایل خطوط (Excel)"), type="xlsx")
uploaded_positions_file = st.sidebar.file_uploader(("فایل مختصات (Excel)"), type="xlsx")
uploaded_background_image = st.sidebar.file_uploader(("تصویر پس‌زمینه (PNG/JPG)"), type=["png", "jpg", "jpeg"])

# Define default paths (modify if necessary)
DEFAULT_CAPACITY_PATH = 'new_capacity_data_with_passengers.csv'
DEFAULT_LINE_PATH = 'Book1.xlsx'
DEFAULT_POSITIONS_PATH = 'liness.xlsx'
DEFAULT_IMAGE_PATH = 'background_image.png' # نام تصویر پس‌زمینه خود را اینجا قرار دهید

data = load_network_data(uploaded_capacity_file, DEFAULT_CAPACITY_PATH, 'csv')
line_df = load_network_data(uploaded_line_file, DEFAULT_LINE_PATH, 'excel')
positions_df, background_image_pil = load_positions_and_image(uploaded_positions_file, DEFAULT_POSITIONS_PATH, uploaded_background_image, DEFAULT_IMAGE_PATH)


# --- Helper function for Plotly graph ---
def calculate_perpendicular_offset(p1, p2, offset_distance):
    dx = p2[0] - p1[0]
    dy = p2[1] - p1[1]
    length = np.hypot(dx, dy)
    if length == 0: return (0, 0)
    # Normalized perpendicular vector
    offset_x_norm = -dy / length
    offset_y_norm = dx / length
    return offset_x_norm * offset_distance, offset_y_norm * offset_distance

# --- New Plotly Graph Drawing Function ---
def draw_detailed_metro_map_plotly(capacity_df, line_data_df, pos_df, img_pil, selected_hour=None):
    if capacity_df is None or capacity_df.empty:
        print("خطا: داده‌های ظرفیت برای رسم نقشه دقیق موجود نیست.")
        return go.Figure()
    if line_data_df is None or line_data_df.empty:
        print("خطا: داده‌های خطوط برای رسم نقشه دقیق موجود نیست.")
        return go.Figure()
    if pos_df is None or pos_df.empty:
        print("خطا: داده‌های مختصات ایستگاه‌ها برای رسم نقشه دقیق موجود نیست.")
        return go.Figure()

    required_cap_cols = ['from_station', 'to_station', 'line', 'direction', 'capacity_utilization_percent', 'hour', 'passenger_count']
    if not all(col in capacity_df.columns for col in required_cap_cols):
        missing_cols = [col for col in required_cap_cols if col not in capacity_df.columns]
        print(f"خطا: ستون‌های ضروری ({', '.join(missing_cols)}) در دیتافریم ظرفیت یافت نشد.")
        return go.Figure()

    required_pos_cols = ['station', 'x', 'y', 'line number', 'name']
    if not all(col in pos_df.columns for col in required_pos_cols):
        missing_cols = [col for col in required_pos_cols if col not in pos_df.columns]
        print(f"خطا: ستون‌های ضروری ({', '.join(missing_cols)}) در دیتافریم مختصات (فایل liness.xlsx) یافت نشد. لطفاً اطمینان حاصل کنید ستون 'name' برای اسامی ایستگاه‌ها موجود است.")
        return go.Figure()

    # =========================================================================
    # تعریف ایستگاه‌های پایانی قطعی مورد نظر شما
    # این ایستگاه‌ها اجازه نخواهند داشت که در مسیر اصلی خط به ایستگاه دیگری متصل شوند.
    HARD_TERMINAL_STATIONS = {92, 202} 
    # می‌توانید ایستگاه‌های دیگری را نیز به این مجموعه اضافه کنید.
    # =========================================================================

    edge_details = {}
    filtered_df = capacity_df.copy()
    if selected_hour is not None:
        try:
            current_hour = int(selected_hour)
            if pd.api.types.is_numeric_dtype(filtered_df['hour']):
                filtered_df = filtered_df[filtered_df['hour'] == current_hour]
            else:
                print("هشدار: ستون 'hour' عددی نیست. فیلتر ساعت اعمال نشد.")
        except (ValueError, TypeError):
            print(f"خطا: مقدار ساعت انتخاب شده ({selected_hour}) نامعتبر است.")

    for _, row in filtered_df.iterrows():
        if pd.isna(row['from_station']) or pd.isna(row['to_station']) or pd.isna(row['line']) or \
           pd.isna(row['direction']) or pd.isna(row['capacity_utilization_percent']) or pd.isna(row['passenger_count']):
            continue
        try:
            from_station = int(row['from_station'])
            to_station = int(row['to_station'])
            line = int(row['line'])
            direction = str(row['direction']).lower().strip()
            capacity_util = float(row['capacity_utilization_percent'])
            passengers = int(row['passenger_count'])
            key = (from_station, to_station, line, direction)
            edge_details[key] = {'capacity': capacity_util, 'passengers': passengers}
        except ValueError as ve:
            print(f"هشدار: خطا در تبدیل داده برای ردیف: {row}. {ve}")
            continue

    station_coords_from_pos_df = {}
    for _, row in pos_df.dropna(subset=['station', 'x', 'y', 'line number', 'name']).iterrows():
        try:
            station_id = int(row['station'])
            station_name = str(row['name'])
            x, y = float(row['x']), float(row['y'])
            line_str = str(row['line number']).strip()
            if station_id not in station_coords_from_pos_df:
                station_coords_from_pos_df[station_id] = {
                    'x': x, 'y': y,
                    'lines': line_str,
                    'id': station_id,
                    'name': station_name
                }
        except ValueError as ve:
            print(f"هشدار: خطا در تبدیل مختصات یا نام برای ایستگاه: {row.get('station', 'N/A')}. {ve}")
            continue
        except KeyError:
            print(f"هشدار: ستون ضروری (مانند 'station', 'x', 'y', 'line number', یا 'name') در ردیف اطلاعات ایستگاه یافت نشد: {row}")
            continue

    if not station_coords_from_pos_df:
        print("خطا: هیچ مختصات معتبری برای ایستگاه‌ها یافت نشد.")
        return go.Figure()

    fig = go.Figure()

    if img_pil:
        img_byte_arr = io.BytesIO()
        img_pil.save(img_byte_arr, format='PNG')
        img_base64 = base64.b64encode(img_byte_arr.getvalue()).decode('utf-8')
        fig.add_layout_image(
            dict(
                source=f"data:image/png;base64,{img_base64}",
                xref="x", yref="y", x=0, y=0,
                sizex=img_pil.width, sizey=img_pil.height,
                sizing="stretch", opacity=0.6, layer="below"
            )
        )
        fig.update_xaxes(range=[0, img_pil.width], showgrid=False, zeroline=False, visible=False)
        fig.update_yaxes(range=[img_pil.height, 0], showgrid=False, zeroline=False, visible=False)
    else:
        all_x = [info['x'] for info in station_coords_from_pos_df.values()]
        all_y = [info['y'] for info in station_coords_from_pos_df.values()]
        if all_x and all_y:
            padding_x = (max(all_x) - min(all_x)) * 0.1
            padding_y = (max(all_y) - min(all_y)) * 0.1
            fig.update_xaxes(range=[min(all_x) - padding_x, max(all_x) + padding_x], visible=False)
            fig.update_yaxes(range=[max(all_y) + padding_y, min(all_y) - padding_y], visible=False)

    edge_color_palette = {'green': '#2ECC71', 'yellow': '#F1C40F', 'orange': '#E67E22', 'red': '#E74C3C', 'over_capacity': '#000000', 'default': '#BDC3C7'}

    def get_edge_style_plotly(capacity_util_percent):
        width = 2
        if pd.isna(capacity_util_percent):
            return edge_color_palette['default'], width
        if capacity_util_percent > 100:
            color = edge_color_palette['over_capacity']
            width = min(7, 3 + ((capacity_util_percent - 100) / 20))
        elif capacity_util_percent > 75:
            color = edge_color_palette['red']
            width = 5
        elif capacity_util_percent > 50:
            color = edge_color_palette['orange']
            width = 4
        elif capacity_util_percent > 25:
            color = edge_color_palette['yellow']
            width = 3
        else:
            color = edge_color_palette['green']
            width = 2.5
        return color, width

    offset_distance = 7
    
    # این دیکشنری انشعاب از کد اصلی شماست. در صورت نیاز آن را به‌روز کنید.
    branching_original = {'1': {122: [123, 91]}, '4': {205: [203, 206]}}
    branching_int_keys = {}
    for line_key, branch_dict_val in branching_original.items():
        new_branch_dict = {}
        for station_key, destinations in branch_dict_val.items():
            try:
                new_branch_dict[int(station_key)] = [int(d) for d in destinations]
            except ValueError:
                 print(f"هشدار: کلید ایستگاه یا مقصد انشعاب '{station_key}' / '{destinations}' در خط '{line_key}' عدد معتبر نیست.")
                 new_branch_dict[station_key] = destinations
        branching_int_keys[line_key] = new_branch_dict
    branching = branching_int_keys # از این پس branching دارای کلیدهای عددی برای ایستگاه‌ها است

    processed_segments_for_legend = set()
    
    termini_stations_info = {}
    for line_col_name in line_data_df.columns:
        line_number_str = str(line_col_name).strip()
        try:
            line_number_int = int(line_number_str)
        except ValueError:
            print(f"هشدار: نام ستون خط '{line_number_str}' یک عدد معتبر نیست. از این خط صرف نظر می‌شود.")
            continue
            
        stations_on_line_for_termini = line_data_df[line_col_name].dropna().astype(int).tolist() # نام متغیر تغییر کرد تا تداخل نکند
        if stations_on_line_for_termini:
            first_station_id = stations_on_line_for_termini[0]
            last_station_id = stations_on_line_for_termini[-1]

            if first_station_id not in termini_stations_info:
                termini_stations_info[first_station_id] = set()
            termini_stations_info[first_station_id].add(f"خط {line_number_str} (ابتدا)")

            if last_station_id not in termini_stations_info:
                termini_stations_info[last_station_id] = set()
            termini_stations_info[last_station_id].add(f"خط {line_number_str} (انتها)")

    for line_col_name in line_data_df.columns:
        line_number_str = str(line_col_name).strip()
        try:
            line_number_int = int(line_number_str)
        except ValueError:
            continue # هشدار قبلاً داده شده است

        stations_on_line = line_data_df[line_col_name].dropna().astype(int).tolist()
        segments_to_process = [] # لیست سگمنت‌ها برای این خط خاص
        temp_path = []
        
        current_branching_for_line = branching.get(line_number_str, {})

        for i in range(len(stations_on_line)):
            s = stations_on_line[i] # s شناسه عددی ایستگاه است
            temp_path.append(s)
            
            if s in current_branching_for_line: # اگر s یک نقطه انشعاب باشد
                if len(temp_path) > 1: 
                    for j in range(len(temp_path) - 1): 
                        start_node = temp_path[j]
                        end_node = temp_path[j+1]
                        
                        # === تغییر اعمال شده ===
                        if start_node in HARD_TERMINAL_STATIONS:
                            continue 
                        # === پایان تغییر ===
                        segments_to_process.append((start_node, end_node))
                
                branch_destinations = current_branching_for_line[s]
                for dest_station in branch_destinations:
                    segments_to_process.append((s, dest_station))
                
                temp_path = [s]
            
            elif i == len(stations_on_line) - 1: 
                if len(temp_path) > 1:
                     for j in range(len(temp_path) - 1):
                        start_node = temp_path[j]
                        end_node = temp_path[j+1]
                        
                        # === تغییر اعمال شده ===
                        if start_node in HARD_TERMINAL_STATIONS:
                            continue
                        # === پایان تغییر ===
                        segments_to_process.append((start_node, end_node))
        
        unique_segments = []
        seen_segments = set()
        for seg_start, seg_end in segments_to_process:
            try:
                s_start = int(seg_start)
                s_end = int(seg_end)
                sorted_seg = tuple(sorted((s_start, s_end))) # برای جلوگیری از رسم دوباره سگمنت در جهت مخالف (اگر داده‌ها شامل هر دو جهت باشند)
                                                            # یا اگر نمی‌خواهید این کار را بکنید، این خط را بردارید و از (s_start, s_end) استفاده کنید.
                                                            # در کد اصلی شما، این مرتب‌سازی وجود داشت برای unique_segments، پس حفظ شده است.
                if sorted_seg not in seen_segments:
                    unique_segments.append((s_start, s_end)) # ذخیره به صورت اصلی برای حفظ جهت‌دهی احتمالی
                    seen_segments.add(sorted_seg)
            except ValueError:
                print(f"هشدار: شناسه ایستگاه نامعتبر در بخش ({seg_start}-{seg_end}) برای خط {line_number_str}")
                continue

        for from_s_id, to_s_id in unique_segments:
            if from_s_id not in station_coords_from_pos_df or to_s_id not in station_coords_from_pos_df:
                print(f"هشدار: مختصات برای ایستگاه {from_s_id} یا {to_s_id} در خط {line_number_str} یافت نشد.")
                continue

            p1_actual = (station_coords_from_pos_df[from_s_id]['x'], station_coords_from_pos_df[from_s_id]['y'])
            p2_actual = (station_coords_from_pos_df[to_s_id]['x'], station_coords_from_pos_df[to_s_id]['y'])

            # Forward direction (از from_s_id به to_s_id)
            key_fwd = (from_s_id, to_s_id, line_number_int, "forward") # یا هر جهتی که در داده‌هایتان دارید
            details_fwd = edge_details.get(key_fwd, {})
            util_fwd = details_fwd.get('capacity', np.nan)
            pass_fwd = details_fwd.get('passengers', 'N/A')
            color_fwd, width_fwd = get_edge_style_plotly(util_fwd)
            off_x_fwd, off_y_fwd = calculate_perpendicular_offset(p1_actual, p2_actual, offset_distance)
            p1_fwd_viz = (p1_actual[0] + off_x_fwd, p1_actual[1] + off_y_fwd)
            p2_fwd_viz = (p2_actual[0] + off_x_fwd, p2_actual[1] + off_y_fwd)
            
            from_station_display_edge = station_coords_from_pos_df[from_s_id].get('name', str(from_s_id))
            to_station_display_edge = station_coords_from_pos_df[to_s_id].get('name', str(to_s_id))

            hover_text_fwd = (f"خط: {line_number_str} (رفت)<br>"
                                         f"{from_station_display_edge} به {to_station_display_edge}<br>"
                                         f"مسافر: {pass_fwd}<br>ظرفیت: {util_fwd if pd.notna(util_fwd) else 'N/A'}{'%' if pd.notna(util_fwd) else ''}")
            legend_group_fwd = f"line_{line_number_str}_fwd_{from_s_id}_{to_s_id}" # برای یکتایی بیشتر
            show_legend_fwd = legend_group_fwd not in processed_segments_for_legend
            
            # بررسی اینکه آیا از to_s_id نباید به from_s_id متصل شویم (برای جلوگیری از رسم خطوط اضافی در unique_segments)
            # این بخش در کد اصلی شما برای نمایش خطوط رفت و برگشت جداگانه بود.
            fig.add_trace(go.Scatter(x=[p1_fwd_viz[0], p2_fwd_viz[0]], y=[p1_fwd_viz[1], p2_fwd_viz[1]],
                                     mode='lines', line=dict(color=color_fwd, width=width_fwd),
                                     hoverinfo='text', text=hover_text_fwd, name=(f"خط {line_number_str} ({from_station_display_edge} به {to_station_display_edge})"), # نامگذاری بهتر برای لجند
                                     legendgroup=legend_group_fwd, showlegend=show_legend_fwd))
            if show_legend_fwd: processed_segments_for_legend.add(legend_group_fwd)


            # Backward direction (از to_s_id به from_s_id)
            key_bwd = (to_s_id, from_s_id, line_number_int, "backward") # یا هر جهتی که در داده‌هایتان دارید
            details_bwd = edge_details.get(key_bwd, {})
            util_bwd = details_bwd.get('capacity', np.nan)
            pass_bwd = details_bwd.get('passengers', 'N/A')
            color_bwd, width_bwd = get_edge_style_plotly(util_bwd)
            off_x_bwd, off_y_bwd = calculate_perpendicular_offset(p1_actual, p2_actual, -offset_distance) # آفست منفی برای جهت مخالف
            p1_bwd_viz = (p1_actual[0] + off_x_bwd, p1_actual[1] + off_y_bwd)
            p2_bwd_viz = (p2_actual[0] + off_x_bwd, p2_actual[1] + off_y_bwd)
            hover_text_bwd = (f"خط: {line_number_str} (برگشت)<br>"
                                          f"{to_station_display_edge} به {from_station_display_edge}<br>"
                                          f"مسافر: {pass_bwd}<br>ظرفیت: {util_bwd if pd.notna(util_bwd) else 'N/A'}{'%' if pd.notna(util_bwd) else ''}")
            legend_group_bwd = f"line_{line_number_str}_bwd_{to_s_id}_{from_s_id}" # برای یکتایی بیشتر
            show_legend_bwd = legend_group_bwd not in processed_segments_for_legend

            fig.add_trace(go.Scatter(x=[p1_bwd_viz[0], p2_bwd_viz[0]], y=[p1_bwd_viz[1], p2_bwd_viz[1]],
                                     mode='lines', line=dict(color=color_bwd, width=width_bwd),
                                     hoverinfo='text', text=hover_text_bwd, name=(f"خط {line_number_str} ({to_station_display_edge} به {from_station_display_edge})"), # نامگذاری بهتر
                                     legendgroup=legend_group_bwd, showlegend=show_legend_bwd))
            if show_legend_bwd: processed_segments_for_legend.add(legend_group_bwd)

    station_nodes_x = []
    station_nodes_y = []
    station_nodes_colors = []
    station_nodes_hover_texts = []
    station_node_ids = []
    station_node_sizes = []
    default_station_size = 10
    terminus_station_size = 16 # اندازه بزرگتر برای ایستگاه‌های پایانی خطوط اصلی
    
    line_color_map_nodes = {'1': 'red', '2': 'blue', '3': 'skyblue', '4': 'yellow', '5': 'green', '6': 'pink', '7': 'purple', 'default': 'grey'}

    for station_id, info in station_coords_from_pos_df.items():
        station_nodes_x.append(info['x'])
        station_nodes_y.append(info['y'])
        station_node_ids.append(info['id'])

        # تشخیص اینکه آیا ایستگاه، پایانه یک خط اصلی است یا خیر (بر اساس termini_stations_info)
        # HARD_TERMINAL_STATIONS برای جلوگیری از ادامه مسیر است، نه لزوما برای تغییر ظاهر.
        # اگر می‌خواهید ظاهر این ایستگاه‌ها هم متفاوت باشد، می‌توانید شرطی برای آن اضافه کنید.
        if info['id'] in termini_stations_info:
            station_node_sizes.append(terminus_station_size)
        else:
            station_node_sizes.append(default_station_size)

        lines_for_station = str(info['lines']).split(',')[0].strip() # استفاده از اولین خط برای رنگ‌بندی
        station_nodes_colors.append(line_color_map_nodes.get(lines_for_station, line_color_map_nodes['default']))
        
        station_display_name = info.get('name', str(info['id']))
        hover_text_node = (f"ایستگاه: {station_display_name}<br>خطوط: {info['lines']}")
        station_nodes_hover_texts.append(hover_text_node)

    fig.add_trace(go.Scatter(
        x=station_nodes_x, y=station_nodes_y,
        mode='markers',
        marker=dict(
            size=station_node_sizes,
            color=station_nodes_colors,
            line=dict(width=1, color='DarkSlateGrey')
        ),
        hoverinfo='text', hoverlabel=dict(font_size=12), text=station_nodes_hover_texts,
        name=('ایستگاه‌ها'), legendgroup="stations", showlegend=True
    ))
    
    annotations_list = []
    for station_id_key, roles_set in termini_stations_info.items():
        if station_id_key in station_coords_from_pos_df:
            station_map_info = station_coords_from_pos_df[station_id_key]
            x_coord = station_map_info['x']
            y_coord = station_map_info['y']
            
            roles_text_parts = sorted(list(roles_set))
            roles_text_display = ", ".join(str(part) for part in roles_text_parts)
            
            station_display_name_annotation = station_map_info.get('name', str(station_map_info['id']))
            annotation_text = f"<b>ایستگاه {station_display_name_annotation}</b><br>{roles_text_display}"
            
            annotations_list.append(
                go.layout.Annotation(
                    x=x_coord,
                    y=y_coord,
                    xref="x",
                    yref="y",
                    text=annotation_text,
                    showarrow=False,
                    xanchor="left",
                    yanchor="bottom",
                    font=dict(family="Tahoma, sans-serif", size=10, color="#2c3e50"),
                    bgcolor="rgba(255, 255, 255, 0.75)",
                    borderpad=3,
                    borderwidth=0.5,
                    bordercolor="#7f8c8d"
                )
            )
    
    if annotations_list:
        if 'annotations' not in fig.layout or fig.layout.annotations is None:
            fig.layout.annotations = tuple()
        fig.layout.annotations += tuple(annotations_list)


    title_text_plotly = ("نقشه شبکه مترو با نمایش ظرفیت و تعداد مسافر")
    if selected_hour is not None:
        try:
            title_text_plotly += (f" - ساعت {int(selected_hour)}:00")
        except ValueError: pass

    fig.update_layout(
        title=dict(text=title_text_plotly, x=0.5, font=dict(size=18)),
        showlegend=False,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1, font=dict(size=10), itemsizing='constant'),
        dragmode='pan', margin=dict(l=20, r=20, t=50, b=20), hovermode='closest'
    )
    if not img_pil:
         fig.update_yaxes(autorange="reversed")
    return fig

# --- Main App Logic ---
if data is not None and not data.empty:
    if 'hour' not in data.columns:
        st.error(("ستون 'hour' در فایل ظرفیت یافت نشد. این ستون برای فیلتر ساعتی ضروری است."))
        data = None # Prevent further processing if essential column is missing
    else:
        # Ensure 'hour' is treated as integer for slider, handle potential errors
        try:
            data['hour'] = data['hour'].astype(int)
            available_hours = sorted(data['hour'].unique())
            if available_hours:
                 selected_hour_slider = st.sidebar.select_slider(
                     ("انتخاب ساعت برای نقشه و جداول"),
                     options=available_hours,
                     value=available_hours[0] if available_hours else None
                 )
                 hour_to_display = selected_hour_slider
            else:
                st.sidebar.warning(("ساعت معتبری در داده‌ها یافت نشد."))
                hour_to_display = None
        except ValueError:
            st.error(("ستون 'hour' شامل مقادیر غیرعددی است. امکان ایجاد اسلایدر ساعت وجود ندارد."))
            data = None # Mark data as unusable for hour-based filtering
            hour_to_display = None
else:
    st.sidebar.warning(("فایل داده ظرفیت بارگذاری نشده یا خالی است."))
    hour_to_display = None


tab1, tab2, tab3 = st.tabs([("نقشه شبکه دقیق"), ("نمودارها"), ("جداول داده")])

with tab1:
    st.header(("نقشه شبکه خطوط مترو (ظرفیت و مسافر)"))
    if data is not None and line_df is not None and positions_df is not None:
        if hour_to_display is not None:
            network_map_fig = draw_detailed_metro_map_plotly(
                capacity_df=data,
                line_data_df=line_df,
                pos_df=positions_df,
                img_pil=background_image_pil,
                selected_hour=hour_to_display
            )
            st.plotly_chart(network_map_fig, use_container_width=True, config={'scrollZoom': True})
        else:
            st.info(("لطفا یک ساعت از نوار کناری انتخاب کنید یا از بارگذاری صحیح داده‌ها اطمینان حاصل کنید."))
    else:
        st.warning(("لطفاً تمامی فایل‌های مورد نیاز (ظرفیت، خطوط، مختصات) را بارگذاری کنید."))


with tab2:
    st.header(("نمودارهای تحلیلی"))
    if data is not None and not data.empty:
        if 'hour' in data.columns and 'capacity_utilization_percent' in data.columns and 'line' in data.columns:
            
            # 1. Hourly Capacity Utilization Trend (Line Chart)
            st.subheader(("روند ساعتی استفاده از ظرفیت (میانگین کل شبکه)"))
            hourly_avg_capacity = data.groupby('hour')['capacity_utilization_percent'].mean().reset_index()
            fig_hourly_trend = px.line(hourly_avg_capacity, x='hour', y='capacity_utilization_percent',
                                       labels={'hour': ('ساعت'), 'capacity_utilization_percent': ('میانگین درصد ظرفیت')},
                                       title=("میانگین استفاده از ظرفیت در ساعات مختلف"))
            fig_hourly_trend.update_xaxes(type='category') # Treat hours as categories if they are discrete
            st.plotly_chart(fig_hourly_trend, use_container_width=True)

            # 2. Capacity Utilization by Line (Bar Chart for selected hour)
            if hour_to_display is not None:
                st.subheader((f"مقایسه استفاده از ظرفیت خطوط (ساعت {hour_to_display})"))
                hourly_data_for_bar = data[data['hour'] == hour_to_display]
                if not hourly_data_for_bar.empty:
                    line_avg_capacity_hour = hourly_data_for_bar.groupby('line')['capacity_utilization_percent'].mean().reset_index()
                    line_avg_capacity_hour['line'] = line_avg_capacity_hour['line'].astype(str) # Ensure line is string for categorical axis
                    fig_line_comparison = px.bar(line_avg_capacity_hour, x='line', y='capacity_utilization_percent',
                                                 labels={'line': ('شماره خط'), 'capacity_utilization_percent': ('میانگین درصد ظرفیت')},
                                                 title=(f"استفاده از ظرفیت بر اساس خط در ساعت {hour_to_display}"),
                                                 color='capacity_utilization_percent', color_continuous_scale=px.colors.sequential.YlOrRd)
                    st.plotly_chart(fig_line_comparison, use_container_width=True)
                else:
                    st.info((f"داده‌ای برای ساعت {hour_to_display} جهت مقایسه خطوط یافت نشد."))
            
            # 3. Heatmap of Capacity Utilization (Hour vs. Line)
            st.subheader(("نقشه حرارتی استفاده از ظرفیت (ساعت در مقابل خط)"))
            heatmap_data = data.groupby(['hour', 'line'])['capacity_utilization_percent'].mean().reset_index()
            if not heatmap_data.empty:
                heatmap_pivot = heatmap_data.pivot(index='line', columns='hour', values='capacity_utilization_percent')
                heatmap_pivot.index = heatmap_pivot.index.map(str) # Ensure line is string
                
                fig_heatmap = px.imshow(heatmap_pivot,
                                        labels=dict(x=("ساعت"), y=("شماره خط"), color=("درصد ظرفیت")),
                                        x=heatmap_pivot.columns, y=heatmap_pivot.index,
                                        text_auto=".0f", # Show values on heatmap cells, rounded
                                        aspect="auto",
                                        color_continuous_scale="YlOrRd")
                fig_heatmap.update_xaxes(side="top", type='category')
                st.plotly_chart(fig_heatmap, use_container_width=True)
            else:
                st.info(("داده‌ای برای ایجاد نقشه حرارتی یافت نشد."))
        else:
            st.warning(("ستون‌های 'hour'، 'capacity_utilization_percent'، یا 'line' برای ایجاد نمودارها یافت نشد."))
    else:
        st.info(("داده‌ای برای نمایش نمودارها بارگذاری نشده است."))


with tab3:
    st.header(("جداول داده و جستجو"))
    if data is not None and not data.empty:
        
        # Display Top 10 busiest routes for selected hour
        if hour_to_display is not None and 'passenger_count' in data.columns:
            st.subheader((f"۱۰ مسیر شلوغ بر اساس تعداد مسافر (ساعت {hour_to_display})"))
            hourly_data_for_table = data[data['hour'] == hour_to_display].copy() # Use .copy() to avoid SettingWithCopyWarning
            if not hourly_data_for_table.empty:
                # Create a 'route' column for better display
                hourly_data_for_table['route'] = hourly_data_for_table['from_station'].astype(str) + " -> " + hourly_data_for_table['to_station'].astype(str) + " (خط " + hourly_data_for_table['line'].astype(str) + ", " + hourly_data_for_table['direction'] + ")"
                top_10_routes = hourly_data_for_table.sort_values(by='passenger_count', ascending=False).head(10)
                # Select and rename columns for display
                display_cols_routes = {
                    'route': ('مسیر (ایستگاه مبدا -> مقصد، خط، جهت)'),
                    'passenger_count': ('تعداد مسافر'),
                    'capacity_utilization_percent': ('درصد استفاده از ظرفیت')
                }
                st.dataframe(top_10_routes[list(display_cols_routes.keys())].rename(columns=display_cols_routes), use_container_width=True)
            else:
                st.info((f"داده‌ای برای ساعت {hour_to_display} جهت نمایش مسیرهای شلوغ یافت نشد."))
        elif 'passenger_count' not in data.columns:
            st.warning(("ستون 'passenger_count' برای نمایش مسیرهای شلوغ یافت نشد."))


        # Advanced Search and Data Download
        st.subheader(("جستجوی پیشرفته و دانلود داده‌ها"))
        
        # Create filter widgets
        cols_to_filter = ['from_station', 'to_station', 'line', 'direction', 'hour']
        filters = {}
        
        # Prepare columns for filtering (ensure they exist)
        available_filter_cols = [col for col in cols_to_filter if col in data.columns]

        for col in available_filter_cols:
            unique_vals = sorted(data[col].unique())
            if unique_vals:
                # For 'hour', ensure it's int if possible
                if col == 'hour':
                    try: unique_vals = sorted(data[col].astype(int).unique())
                    except ValueError: pass # Keep as is if conversion fails
                
                selected_val = st.multiselect((f"فیلتر بر اساس {col}"), unique_vals, key=f"filter_{col}")
                if selected_val:
                    filters[col] = selected_val
        
        filtered_search_data = data.copy()
        for col, selected_values in filters.items():
            if selected_values: # If a filter is chosen for this column
                 # Ensure consistent data types for comparison, especially for numeric-like strings
                if data[col].dtype == 'object' and all(isinstance(x, (int, float)) or str(x).isnumeric() for x in selected_values):
                    try:
                        filtered_search_data = filtered_search_data[filtered_search_data[col].astype(float).isin([float(x) for x in selected_values])]
                    except ValueError: # Fallback to string comparison if astype float fails for column
                        filtered_search_data = filtered_search_data[filtered_search_data[col].isin([str(x) for x in selected_values])]
                else: # Standard isin for other types or when selected_values are not all numeric-like
                    filtered_search_data = filtered_search_data[filtered_search_data[col].isin(selected_values)]


        st.dataframe(filtered_search_data, height=400, use_container_width=True)

        if not filtered_search_data.empty:
            csv_export = filtered_search_data.to_csv(index=False).encode('utf-8-sig') # utf-8-sig for Excel compatibility
            st.download_button(
                label=("دانلود نتایج جستجو (CSV)"),
                data=csv_export,
                file_name=("نتایج_جستجوی_ظرفیت_مترو.csv"),
                mime='text/csv',
            )
        else:
            st.info(("هیچ داده‌ای با فیلترهای انتخابی مطابقت ندارد."))
            
    else:
        st.info(("داده‌ای برای نمایش جداول یا جستجو بارگذاری نشده است."))


# --- Footer / Usage Guide (Updated) ---
st.markdown("---")
st.markdown(("**راهنمای استفاده:**"))
st.markdown(("""
- **بارگذاری داده:** در نوار کناری صفحه، فایل‌های CSV (ظرفیت)، Excel (خطوط)، Excel (مختصات) و تصویر پس‌زمینه را آپلود کنید.
- **انتخاب ساعت:** از اسلایدر در نوار کناری برای انتخاب ساعت مورد نظر جهت نمایش در نقشه‌ها و جداول استفاده کنید.
- **تب نقشه شبکه دقیق:** نقشه گرافیکی خطوط مترو را با قابلیت زوم و هاور برای نمایش اطلاعات ایستگاه و مسیرها ارائه می‌دهد.
- **تب نمودارها:** نمودارهای تحلیلی روند ساعتی، مقایسه خطوط، و نقشه حرارتی ظرفیت را نمایش می‌دهد.
- **تب جداول داده:** جداول مسیرهای شلوغ و امکان جستجوی پیشرفته در داده‌ها با قابلیت دانلود نتایج را فراهم می‌کند.
"""))

# --- Error Handling for Main Script ---
# (This can be enhanced if specific top-level errors need custom messages)
# try:
#     # Main execution part of the script is implicitly here
#     pass
# except Exception as main_execution_error:
#     st.error((f"یک خطای کلی در اجرای برنامه رخ داده است: {main_execution_error}"))
#     st.error(traceback.format_exc())