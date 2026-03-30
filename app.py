import streamlit as st
import pandas as pd
import numpy as np

# 1. 페이지 설정
st.set_page_config(layout='wide', page_title='FREND Performance Simulator')
st.title('FREND Performance Simulator')

def get_csv(df): 
    return df.to_csv(index=False).encode('utf-8-sig')

tabs = st.tabs([
    'Comparison (EP09)', 
    'Limits (EP17)', 
    'Precision (EP05)', 
    'Accuracy (EP15)', 
    'Linearity (EP06)', 
    'Stability (EP25)'
])

# --- 1. Method Comparison (EP09-A3) : R-value 소수점 4자리 수정 ---
with tabs[0]:
    st.header('Method Comparison (EP09-A3)')
    with st.expander("설정 구성", expanded=True):
        c1, c2, c3 = st.columns(3)
        n = c1.number_input('검체 수 (N)', 10, 200, 40)
        x_min = c1.number_input('Min Concentration', value=0.00)
        x_max = c1.number_input('Max Concentration', value=500.00)
        
        y_s = c2.number_input('Y Slope', value=1.0000, format="%.4f")
        y_i = c2.number_input('Y Intercept', value=0.0000, format="%.4f", key='ep09_yi')
        # R-value 소수점 4자리 입력 가능하도록 수정
        y_r = c2.number_input('Y R-value', 0.0, 1.0, 0.9900, format="%.4f", step=0.0001)
        
        z_s = c3.number_input('Z Slope', value=1.0000, format="%.4f")
        z_i = c3.number_input('Z Intercept', value=0.0000, format="%.4f", key='ep09_zi')
        # R-value 소수점 4자리 입력 가능하도록 수정
        z_r = c3.number_input('Z R-value', 0.0, 1.0, 0.9900, format="%.4f", step=0.0001)

    if st.button('Generate EP09 Data'):
        x_raw = np.random.uniform(x_min, x_max, n)
        
        def gen_inst_with_reps(x, s, i, r):
            target = s * x + i
            # 상관관계(r)에 따른 노이즈 계산 로직 유지
            noise_std = np.sqrt(np.abs(np.var(target)*((1/r**2)-1))) if 0 < r < 1.0 else 0.001
            m1 = np.random.normal(target, noise_std)
            m2 = m1 * np.random.uniform(0.98, 1.02, len(x))
            return np.round(m1, 2), np.round(m2, 2)
        
        x1, x2 = gen_inst_with_reps(x_raw, 1.0, 0.0, 0.9999) 
        y1, y2 = gen_inst_with_reps(x_raw, y_s, y_i, y_r)
        z1, z2 = gen_inst_with_reps(x_raw, z_s, z_i, z_r)
        
        df = pd.DataFrame({
            'ID': range(1, n+1), 
            'X1': x1, 'X2': x2, 
            'Y1': y1, 'Y2': y2, 
            'Z1': z1, 'Z2': z2
        })
        st.dataframe(df)
        st.download_button('Download EP09 CSV', get_csv(df), 'ep09.csv')

# --- 2. Limits (EP17-A2) : 가로 정렬 유지 ---
with tabs[1]:
    st.header('LoB / LoD / LoQ (EP17-A2)')
    with st.expander("🎯 목표치 설정", expanded=True):
        c1, c2, c3, c4 = st.columns(4)
        t_lob = c1.number_input('Target LoB', value=0.010, format="%.3f")
        t_lod = c2.number_input('Target LoD', value=0.025, format="%.3f")
        t_loq = c3.number_input('Target LoQ', value=0.050, format="%.3f")
        t_te = c4.number_input('LoQ 기준 TE (%)', value=20.0)
    st.write("**Blank Concentration Input (B1~B5)**")
    b_cols = st.columns(5)
    b_concs = [b_cols[i].number_input(f'B{i+1}', value=0.002, format="%.3f", key=f'b_e17_{i}') for i in range(5)]
    st.write("**Low Concentration Input (L1~L5)**")
    l_cols = st.columns(5)
    l_concs = [l_cols[i].number_input(f'L{i+1}', value=0.030, format="%.3f", key=f'l_e17_{i}') for i in range(5)]
    if st.button('Generate EP17 Data'):
        data_dict, summary = {}, []
        for i in range(5):
            sd = max(0.001, (t_lob - b_concs[i]) / 1.645)
            vals = [round(max(0, v), 3) for v in np.random.normal(b_concs[i], sd, 12)]
            data_dict[f'Blank {i+1}'] = vals
            summary.append([f'Blank {i+1}', np.mean(vals), np.std(vals, ddof=1), "-"])
        for i in range(5):
            sd = (l_concs[i] * (t_te/100)) / 2.5
            vals = [round(max(0, v), 3) for v in np.random.normal(l_concs[i], sd, 12)]
            data_dict[f'Low {i+1}'] = vals
            m, s = np.mean(vals), np.std(vals, ddof=1)
            te = (abs(m - l_concs[i])/l_concs[i]*100) + (2*(s/m*100))
            summary.append([f'Low {i+1}', m, s, f"{round(te, 1)}%"])
        df_ep17 = pd.DataFrame(data_dict, index=pd.MultiIndex.from_product([[1,2,3],[1,2,3,4]], names=['Day','Rep'])).reset_index()
        st.dataframe(df_ep17)
        st.table(pd.DataFrame(summary, columns=['Sample','Mean','SD','Actual TE%']))
        st.download_button('Download EP17 CSV', get_csv(df_ep17), 'ep17.csv')

# --- 3. Precision (EP05-A3) : 기존 유지 ---
with tabs[2]:
    st.header('Precision Analysis (EP05-A3)')
    p_mode = st.radio("모드", ["Repeatability (20D)", "Reproducibility (3 Lot)"], horizontal=True)
    if p_mode == "Repeatability (20D)":
        num_lv = st.slider('레벨 수', 1, 5, 2, key='p20_lv')
        p_cols = st.columns(num_lv)
        confs, cvs = [], []
        for i in range(num_lv):
            with p_cols[i]:
                confs.append(st.number_input(f'Lv{i+1} Conc.', value=10.0, key=f'rc{i}'))
                cvs.append(st.number_input(f'Lv{i+1} CV%', value=3.0, key=f'rv{i}'))
        if st.button('Generate 20D Data'):
            rows = []
            for d in range(1, 21):
                for run in range(1, 3):
                    for rep in range(1, 3):
                        row = [d, run, rep]
                        for i in range(num_lv):
                            row.append(round(max(0, np.random.normal(confs[i], confs[i]*cvs[i]/100)), 3))
                        rows.append(row)
            df = pd.DataFrame(rows, columns=['Day', 'Run', 'Rep'] + [f'Level {i+1}' for i in range(num_lv)])
            st.dataframe(df); st.download_button('Download 20D CSV', get_csv(df), 'precision_20d.csv')
    else: # Reproducibility (3 Lot)
        num_lv = st.slider('레벨 수', 1, 5, 3, key='prp_lv')
        lot_configs = []
        for lot_id in range(1, 4):
            with st.expander(f"Lot {lot_id} 설정", expanded=True):
                t_cols = st.columns(num_lv)
                l_targets = [t_cols[i].number_input(f'Lv{i+1} Target', value=10.0*(10**i), key=f'l{lot_id}_t{i}') for i in range(num_lv)]
                v_cols = st.columns(num_lv)
                l_cvs = [v_cols[i].number_input(f'Lv{i+1} CV%', value=5.0, key=f'l{lot_id}_v{i}') for i in range(num_lv)]
                lot_configs.append({'targets': l_targets, 'cvs': l_cvs})
        if st.button('Generate 3 Lot Data'):
            all_data = []
            for lot_idx, config in enumerate(lot_configs):
                lot_rows = []
                for d in range(1, 6):
                    for r in range(1, 6):
                        row = [f'Lot {lot_idx+1}', d, r]
                        for i in range(num_lv):
                            target, cv = config['targets'][i], config['cvs'][i]
                            row.append(round(max(0, np.random.normal(target, target * (cv/100))), 3))
                        lot_rows.append(row)
                all_data.append(pd.DataFrame(lot_rows, columns=['Lot','Day','Rep'] + [f'Lv {i+1}' for i in range(num_lv)]))
            full_df = pd.concat(all_data, ignore_index=True)
            st.write("#### Total Combined Data"); st.dataframe(full_df)
            st.download_button('Download Reproducibility CSV', get_csv(full_df), 'reproducibility.csv')

# --- 4. Accuracy & User-Precision (EP15-A3) : 기존 유지 ---
with tabs[3]:
    st.header('Accuracy & User-Precision (EP15-A3)')
    num_lv_ep15 = st.slider('농도 레벨 수', 1, 5, 3, key='ep15_lv_s')
    ep15_confs, ep15_cvs = [], []
    cols_ep15 = st.columns(num_lv_ep15)
    for i in range(num_lv_ep15):
        with cols_ep15[i]:
            ep15_confs.append(st.number_input(f'Lv{i+1} Target', value=10.0*(10**i), key=f'ep15_t{i}'))
            ep15_cvs.append(st.number_input(f'Lv{i+1} CV%', value=4.0, key=f'ep15_v{i}'))
    if st.button('Generate EP15 Data'):
        all_lv_data = []
        for i in range(num_lv_ep15):
            target, cv = ep15_confs[i], ep15_cvs[i]
            sd = target * (cv / 100)
            lv_rows = []
            for d in range(1, 6):
                reps = [round(max(0, np.random.normal(target, sd)), 3) for _ in range(5)]
                lv_rows.append([f'Level {i+1}', d] + reps)
            df_lv = pd.DataFrame(lv_rows, columns=['Sample', 'Day', 'Rep 1', 'Rep 2', 'Rep 3', 'Rep 4', 'Rep 5'])
            st.write(f"#### Level {i+1} Data (5D x 5R)"); st.dataframe(df_lv)
            all_lv_data.append(df_lv)
        full_ep15 = pd.concat(all_lv_data, ignore_index=True)
        st.download_button('Download EP15 CSV', get_csv(full_ep15), 'ep15.csv')

# --- 5. Linearity (EP06-A2) : 가로 정렬 유지 ---
with tabs[4]:
    st.header('Linearity (EP06-A2)')
    n_s = st.number_input('농도 구간 수', 5, 15, 5, key='lin_n')
    lin_concs, lin_cvs = [], []
    for i in range(0, n_s, 5):
        cols = st.columns(5)
        for j in range(5):
            if i + j < n_s:
                with cols[j]:
                    lin_concs.append(st.number_input(f'Lv {i+j+1} Conc.', value=float((i+j)*100), key=f'ls_c{i+j}'))
                    lin_cvs.append(st.number_input(f'Lv {i+j+1} CV%', value=2.0, key=f'ls_v{i+j}'))
    if st.button('Generate Linearity'):
        rows = []
        for idx, lv in enumerate(lin_concs):
            sd = lv * (lin_cvs[idx] / 100)
            reps = [round(max(0, np.random.normal(lv, sd)), 2) for _ in range(7)]
            rows.append([lv] + reps)
        df_lin = pd.DataFrame(rows, columns=['Target'] + [f'Rep {i+1}' for i in range(7)])
        st.dataframe(df_lin); st.download_button('Download Linearity CSV', get_csv(df_lin), 'linearity_horizontal.csv')

# --- 6. Stability (EP25-A) : 기존 유지 ---
with tabs[5]:
    st.header('Stability (EP25-A)')
    dur = st.number_input('평가 기간 (Month)', 1, 36, 12, key='stab_d')
    if st.button('Generate Stability'):
        res = [[f'S_{v}', t, r, round(max(0, v*(1-t*0.001) + np.random.normal(0, v*0.03)), 3)] 
               for v in [10.0, 50.0, 100.0] for t in range(dur + 1) for r in range(1, 21)]
        df = pd.DataFrame(res, columns=['Sample', 'Month', 'Rep', 'Value'])
        st.dataframe(df); st.download_button('Download Stability CSV', get_csv(df), 'stability.csv')
        # --- 코드 맨 아래에 이 내용을 추가하세요 ---

footer_content = """
<style>
    .ryangeun-footer {
        position: fixed;
        right: 30px;         /* 오른쪽에서 좀 더 안쪽으로 (로그창 피하기) */
        bottom: 20px;        /* 바닥에서 좀 더 위쪽으로 */
        width: auto;
        color: rgba(0, 0, 0, 0.6); /* 조금 더 진하게 */
        font-size: 12px;
        font-family: sans-serif;
        z-index: 999999;     /* 숫자를 엄청 크게 해서 무조건 맨 위로! */
        pointer-events: none;
        background-color: rgba(255, 255, 255, 0.5); /* 혹시 모르니 살짝 배경색 추가 */
        padding: 2px 5px;
        border-radius: 5px;
    }
</style>
<div class="ryangeun-footer">
    Created by Ryangeun
</div>
"""
st.markdown(footer_content, unsafe_allow_html=True)
